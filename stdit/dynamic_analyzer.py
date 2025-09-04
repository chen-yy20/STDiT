# dynamic_analyzer.py

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from pathlib import Path
from .global_envs import GlobalEnv

class DynamicAnalyzer:
    def __init__(self, T=21, H=60, W=104, C=16, block_size=(3,4,4), analysis_steps=5):
        """
        初始化动态分析器
        T, H, W, C: 潜空间维度
        block_size: (bt,bh,bw) 分块大小
        analysis_steps: 分析前n步
        """
        self.T = T
        self.H = H
        self.W = W
        self.C = C
        self.block_size = block_size
        self.analysis_steps = analysis_steps
        self.stored_features = []
        self.tag = GlobalEnv.get_envs('tag')
        
        print("\nInitialized Dynamic Analyzer:")
        print("-" * 40)
        print(f"Tag: {self.tag}")
        print(f"Feature dimensions: T={T}, H={H}, W={W}, C={C}")
        print(f"Block size: {block_size} (T,H,W)")
        print(f"Analysis steps: {analysis_steps}")
        print(f"Total blocks: {(T//block_size[0])*(H//block_size[1])*(W//block_size[2])}")
        print("-" * 40 + "\n")
        
        
    def step(self, latent, step):
        """存储潜变量，供后续分析"""
        if step < self.analysis_steps:
            self.stored_features.append(latent)
            print(f"Step {step} | Stored noise_pred (shape={latent.shape})", flush=True)
        elif step == self.analysis_steps:
            self.analyze()
        else:
            return
    
    def _block_3d(self, tensor):
        """3D数据分块"""
        step, C, T, H, W = tensor.shape
        bt, bh, bw = self.block_size
        
        assert T % bt == 0 and H % bh == 0 and W % bw == 0
        
        nt, nh, nw = T // bt, H // bh, W // bw
        total_blocks = nt * nh * nw
        
        tensor = tensor.permute(0, 2, 3, 4, 1)
        # 分块重排
        blocked = tensor.view(step, nt, bt, nh, bh, nw, bw, C)
        blocked = blocked.permute(0, 1, 3, 5, 2, 4, 6, 7)
        blocked = blocked.reshape(step, total_blocks, bt, bh*bw, C)
        
        return blocked
    
    def _compute_dynamics(self, blocked_latents):
        """计算时空动态性"""
        device = blocked_latents.device  # 获取输入数据的设备
        step, block_num, t, s, c = blocked_latents.shape
        
        temporal_dynamics = torch.zeros(block_num, device=device)  # 创建在同一设备上的张量
        spatial_dynamics = torch.zeros(block_num, device=device)
        
        for b in range(block_num):
            # 时间动态性
            for si in range(s):
                features = blocked_latents[:, b, :, si, :]
                diff_matrix = features.unsqueeze(0) - features.unsqueeze(1)
                diff_0 = diff_matrix[:, :, 0, :]
                diff_changes = diff_matrix - diff_0.unsqueeze(2)
                dynamics = torch.abs(diff_changes).mean(dim=(0,1,3))
                temporal_dynamics[b] += dynamics.mean()
            temporal_dynamics[b] /= s
            
            # 空间动态性
            for ti in range(t):
                features = blocked_latents[:, b, ti, :, :]
                diff_matrix = features.unsqueeze(0) - features.unsqueeze(1)
                diff_0 = diff_matrix[:, :, 0, :]
                diff_changes = diff_matrix - diff_0.unsqueeze(2)
                dynamics = torch.abs(diff_changes).mean(dim=(0,1,3))
                spatial_dynamics[b] += dynamics.mean()
            spatial_dynamics[b] /= t
        
        return temporal_dynamics, spatial_dynamics

    def analyze(self):
        """执行完整分析并保存结果"""
        if not self.stored_features:
            print("No features stored for analysis")
            return
            
        print(f"\nAnalyzing {self.tag}...", flush=True)
        
        # 准备数据
        latents_tensor = torch.stack(self.stored_features, dim=0)
        # 保持在GPU上
        latents_tensor = latents_tensor.cuda()
        blocked_latents = self._block_3d(latents_tensor)
        
        # 计算动态性（在GPU上）
        temporal_dynamics, spatial_dynamics = self._compute_dynamics(blocked_latents)
        
        # 可视化（需要将数据移到CPU）
        self._visualize_dynamics(temporal_dynamics.cpu(), 
                                spatial_dynamics.cpu(), 
                                save_name = self.tag)
        
        # 清除存储的数据
        self.stored_features = []
        
        print(f"Analysis completed for {self.tag}\n", flush=True)
        
        return temporal_dynamics, spatial_dynamics
    
    def _visualize_dynamics(self, temporal_dynamics, spatial_dynamics, save_name):
        """可视化时空动态性"""
        nt, nh, nw = self.T//self.block_size[0], self.H//self.block_size[1], self.W//self.block_size[2]
        
        temp_map = temporal_dynamics.reshape(nt, nh, nw).cpu().numpy()
        spat_map = spatial_dynamics.reshape(nt, nh, nw).cpu().numpy()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        sns.heatmap(temp_map.mean(axis=0), ax=ax1, cmap='viridis')
        ax1.set_title('Temporal Dynamics')
        ax1.set_xlabel('Width Blocks')
        ax1.set_ylabel('Height Blocks')
        
        sns.heatmap(spat_map.mean(axis=0), ax=ax2, cmap='viridis')
        ax2.set_title('Spatial Dynamics')
        ax2.set_xlabel('Width Blocks')
        ax2.set_ylabel('Height Blocks')
        
        plt.tight_layout()
        save_path = os.path.join(GlobalEnv.get_envs("save_dir"), f"Dynamic_{save_name}_{self.analysis_steps}.png")
        plt.savefig(save_path)
        plt.close()
        
        print(f"Dynamic result saved to {save_path}.")