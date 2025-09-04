# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from functools import partial

import torch
import torch.cuda.amp as amp
import torch.distributed as dist
from tqdm import tqdm

from .distributed.fsdp import shard_model
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae import WanVAE
from .utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
# from .train_free_utils import make_ar1_noise, temporal_align_step
import numpy as np

class WanT2V:

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
    ):
        r"""
        Initializes the Wan text-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_usp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of USP.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None)

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        logging.info(f"Creating WanModel from {checkpoint_dir}")
        self.model = WanModel.from_pretrained(checkpoint_dir)
        self.model.eval().requires_grad_(False)

        if use_usp:
            from xfuser.core.distributed import get_sequence_parallel_world_size

            from .distributed.xdit_context_parallel import (
                usp_attn_forward,
                usp_dit_forward,
            )
            for block in self.model.blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn)
            self.model.forward = types.MethodType(usp_dit_forward, self.model)
            self.sp_size = get_sequence_parallel_world_size()
        else:
            self.sp_size = 1

        if dist.is_initialized():
            dist.barrier()
        if dit_fsdp:
            self.model = shard_fn(self.model)
        else:
            self.model.to(self.device)

        self.sample_neg_prompt = config.sample_neg_prompt

    def generate(self,
                 input_prompt,
                 size=(1280, 720),
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True):
        r"""
        Generates video frames from text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation
            size (tupele[`int`], *optional*, defaults to (1280,720)):
                Controls video resolution, (width,height).
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed.
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N, H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from size)
                - W: Frame width from size)
        """
        # preprocess
        F = frame_num
        target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
                        size[1] // self.vae_stride[1],
                        size[0] // self.vae_stride[2])

        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        noise = [
            torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=self.device,
                generator=seed_g)
        ]






        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)
        epsilon = getattr(self, "tr_epsilon", 0.8)   # 对通道做L2后的阈值，可按实际分布调
        warmup_steps = getattr(self, "tr_warmup", 10) # 用前 6 步做判定(对应 timestep=0..5)
        frame_diffs_hist = []     # 每步一个 (T-1,H,W) tensor
        reuse_mask_frames = None  # (T-1,H,W)；True=该帧该像素点与f=0足够接近，可复用
        mask_ready = False

        # evaluation mode
        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            latents = noise


            xa, ya = 5, 5
            xb, yb = 5, 6 
            xc, yc= 30, 21
            all_steps_AB_L2 = []   # [num_steps, F]，每步每帧的 |A-B|_2
            all_steps_AB_vec = []  # 每步保存 [F, C] 的向量差（如需）
            #latents_init = latents.detach().clone()
            A0 = latents[0][10, :, ya, xa]  # [F]
            B0 = latents[0][10, :, yb, xb]  # [F]
            C0=latents[0][10, :, yc, xc]
            A_steps = [] 
            B_steps = []
            C_steps = []
            #print('init_0:',A0.shape())
      





            arg_c = {'context': context, 'seq_len': seq_len}
            arg_null = {'context': context_null, 'seq_len': seq_len}
            ref_frame = 0

            for step_idx, t in enumerate(tqdm(timesteps)):
                latent_model_input = latents
                timestep = [t]

                timestep = torch.stack(timestep)

                self.model.to(self.device)
                noise_pred_cond = self.model(
                    latent_model_input, t=timestep, **arg_c)[0]
                noise_pred_uncond = self.model(
                    latent_model_input, t=timestep, **arg_null)[0]

                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)
                A_point=noise_pred[10,:,ya,xa]
                A_steps.append(A_point.detach().cpu().numpy())
                B_point=noise_pred[10,:,yb,xb]
                B_steps.append(B_point.detach().cpu().numpy())
                C_point=noise_pred[10,:,yc,xc]
                C_steps.append(C_point.detach().cpu().numpy())

                
                if step_idx == 0:
                    ref0 = noise_pred[:, ref_frame:ref_frame+1, :, :]                # (C,1,H,W)
                    # expand到全部帧后与各帧做差，得到 (C,T,H,W)，其中第0帧差为0
                    offset_t0_full = ref0.expand(-1, noise_pred.size(1), -1, -1) - noise_pred
                    offset_t0_full = offset_t0_full.detach()


                if not mask_ready:
                    # 当前步的 "帧内 vs f=0" 差：对通道维做 L2
                    # noise_pred[:, 1:, :, :] - noise_pred[:, 0:1, :, :] -> (C, T-1, H, W)
                    cur_frame_diff = (noise_pred[:, 1:, :, :] - noise_pred[:, 0:1, :, :]) \
                            .pow(2).sum(dim=0).sqrt()  # (T-1, H, W)
                    frame_diffs_hist.append(cur_frame_diff.detach())

                    if len(frame_diffs_hist) == warmup_steps:
                        # 逐元素取 k=0..warmup_steps-1 的最大差 -> (T-1,H,W)
                        max_diff_over_warmup = torch.stack(frame_diffs_hist, dim=0).amax(dim=0)  # (T-1,H,W)
                        # 判定逐帧复用掩码：True 表示“该帧该点在 warmup 内始终接近 f=0”
                        offset_mag_t0 = offset_t0_full.pow(2).sum(dim=0).sqrt() 
                        second_order_gap = (max_diff_over_warmup - offset_mag_t0[1:]).abs() 
                        reuse_mask_frames = (second_order_gap <= float(epsilon)).to(noise_pred.device)

                        reuse_mask_frames[ref_frame, :, :] = False
                        mask_ready = True
                        # 可选：保存以便可视化
                        torch.save(reuse_mask_frames.cpu(), "temporal_reuse_mask_frames_ep08.pt")
                        torch.save(second_order_gap.cpu(), "second_order_gap.pt")

                if mask_ready and step_idx<40 and step_idx>10:
                        # 按你的公式：noise_x = noise_y + (noise_y0 - noise_x0)，y=ref_frame
                        ref_now = noise_pred[:, ref_frame:ref_frame+1, :, :].expand_as(offset_t0_full)  # (C,T,H,W)
                        reused_full = ref_now - offset_t0_full                                         # (C,T,H,W)

                        mask = reuse_mask_frames.unsqueeze(0)   # (1,T,H,W) -> 广播到C
                        noise_pred[:, 1:, :, :] = torch.where(
                                mask,
                                reused_full[:, 1:, :, :],    # 与 noise_pred 的切片同样是 20 帧
                                noise_pred[:, 1:, :, :]
                                )

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]

                latents = [temp_x0.squeeze(0)]
                print("latents.shape =", latents[0].shape)

            x0 = latents
            A_final=x0[0][10,:,ya,xa]
            B_final=x0[0][10,:,yb,xb]
            C_final=x0[0][10,:,yc,xc]
            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()
            if self.rank == 0:
                videos = self.vae.decode(x0)
        A_steps = np.stack(A_steps, axis=0)
        A0 = A0.cpu().numpy()
        B_steps = np.stack(B_steps, axis=0)
        B0 = B0.cpu().numpy()
        C_steps = np.stack(C_steps, axis=0)
        C0 = C0.cpu().numpy()
        B_final = B_final.cpu().numpy()
        A_final = A_final.cpu().numpy()
        C_final = C_final.cpu().numpy()
        np.savez("AB_points_noise_pred_56_channel10.npz",
         A_final=A_final,
         B_final=B_final,
         C_final=C_final,
         A_steps=A_steps,
         B_steps=B_steps,
         C_steps=C_steps,
         A0=A0,
         B0=B0,
         C0=C0)

        print("保存完成: AB_points_noise_pred_56.npz")
        print("A_steps.shape =", A_steps.shape, " B_steps.shape =", B_steps.shape)
        print("A0.shape =", A0.shape, " B0.shape =", B0.shape)    
        del noise, latents
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()
        print("video.shape =", videos[0].shape)
        return videos[0] if self.rank == 0 else None
