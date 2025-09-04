import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

T = 21
H = 60
W = 104
C = 16

name = "cat"
load_step = 5

def block_3d(tensor, block_size):
    """
    对3D数据进行分块，合并空间维度s
    tensor: (step, T, H, W, C)
    block_size: (bt, bh, bw) - 时间、高度、宽度的分块大小
    返回: (step, block_num, t, s, C)，其中:
    - block_num = (T//bt * H//bh * W//bw)
    - s = bh * bw (合并的空间维度)s
    """
    step, T, H, W, C = tensor.shape
    bt, bh, bw = block_size
    
    # 确保可以整除
    assert T % bt == 0, f"Time dimension {T} not divisible by block size {bt}"
    assert H % bh == 0, f"Height {H} not divisible by block size {bh}"
    assert W % bw == 0, f"Width {W} not divisible by block size {bw}"
    
    # 计算每个维度的块数
    nt, nh, nw = T // bt, H // bh, W // bw
    total_blocks = nt * nh * nw
    
    # 重塑tensor
    blocked = tensor.view(step, nt, bt, nh, bh, nw, bw, C)
    # 调整维度顺序，使得块索引相邻
    blocked = blocked.permute(0, 1, 3, 5, 2, 4, 6, 7)
    # 合并块索引维度和空间维度
    blocked = blocked.reshape(step, total_blocks, bt, bh*bw, C)
    
    return blocked

def load_and_block_data(block_size=(3,4,4)):
    save_dir = f"./{name}"

    latents = []
    for i in range(load_step):
        latent = torch.load(f"{save_dir}/latent_{i:03d}.pt")
        latent = latent.permute(1,2,3,0)  # (T,H,W,C)
        latents.append(latent)
    
    # (step,T,H,W,C)
    latents_tensor = torch.stack(latents, dim=0)
    
    # 进行分块
    blocked_latents = block_3d(latents_tensor, block_size)
    
    T, H, W = latents_tensor.shape[1:4]
    bt, bh, bw = block_size
    
    print(f"Original data shape: {latents_tensor.shape}")
    print(f"Blocked data shape: {blocked_latents.shape}")
    print("\nDimensions explanation:")
    print(f"- Steps: {blocked_latents.shape[0]}")
    print(f"- Blocks: {blocked_latents.shape[1]} (= {T//bt}*{H//bh}*{W//bw})")
    print(f"- Time in block: {blocked_latents.shape[2]}")
    print(f"- Space in block: {blocked_latents.shape[3]} (= {bh}*{bw})")
    print(f"- Channels: {blocked_latents.shape[4]}")
    
    return blocked_latents

def compute_feature_dynamics(blocked_latents):
    """
    计算特征在去噪过程中的动态变化
    blocked_latents: (step, block_num, t, s, c)
    返回: temporal_dynamics, spatial_dynamics (block_num,)
    """
    step, block_num, t, s, c = blocked_latents.shape
    
    # 时间特征动态变化
    temporal_dynamics = torch.zeros(block_num)
    for b in range(block_num):
        # 对每个空间位置s
        for si in range(s):
            # 计算t之间的初始差异
            # features: (step, t, c)
            features = blocked_latents[:, b, :, si, :]
            
            # 计算每一步中时间特征之间的差异
            # (t, t, step, c)
            diff_matrix = features.unsqueeze(0) - features.unsqueeze(1)
            
            # 取初始状态(step=0)的差异作为基准
            # (t, t, c)
            diff_0 = diff_matrix[:, :, 0, :]
            
            # 计算每一步与初始状态的差异变化
            # (step, t, t, c)
            diff_changes = diff_matrix - diff_0.unsqueeze(2)
            
            # 在所有维度上取平均，得到这个空间位置的时间动态性
            # 先取绝对值，防止正负抵消
            dynamics = torch.abs(diff_changes).mean(dim=(0,1,3))  # (step,)
            temporal_dynamics[b] += dynamics.mean()  # 在step上取平均
        
        temporal_dynamics[b] /= s  # 对空间位置取平均
    
    # 空间特征动态变化
    spatial_dynamics = torch.zeros(block_num)
    for b in range(block_num):
        # 对每个时间点t
        for ti in range(t):
            # features: (step, s, c)
            features = blocked_latents[:, b, ti, :, :]
            
            # 计算每一步中空间特征之间的差异
            # (s, s, step, c)
            diff_matrix = features.unsqueeze(0) - features.unsqueeze(1)
            
            # 取初始状态的差异作为基准
            # (s, s, c)
            diff_0 = diff_matrix[:, :, 0, :]
            
            # 计算每一步与初始状态的差异变化
            # (step, s, s, c)
            diff_changes = diff_matrix - diff_0.unsqueeze(2)
            
            # 在所有维度上取平均
            dynamics = torch.abs(diff_changes).mean(dim=(0,1,3))  # (step,)
            spatial_dynamics[b] += dynamics.mean()  # 在step上取平均
        
        spatial_dynamics[b] /= t  # 对时间点取平均
    
    return temporal_dynamics, spatial_dynamics

def visualize_dynamics(temporal_dynamics, spatial_dynamics, block_size):
    """
    将时间和空间动态性可视化为热力图
    """
    nt, nh, nw = T//block_size[0], H//block_size[1], W//block_size[2]
    
    # 重塑为空间网格
    temp_map = temporal_dynamics.reshape(nt, nh, nw).cpu().numpy()
    spat_map = spatial_dynamics.reshape(nt, nh, nw).cpu().numpy()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 时间动态性热力图
    sns.heatmap(temp_map.mean(axis=0), ax=ax1, cmap='viridis')
    ax1.set_title('Temporal Dynamics')
    ax1.set_xlabel('Width Blocks')
    ax1.set_ylabel('Height Blocks')
    
    # 空间动态性热力图
    sns.heatmap(spat_map.mean(axis=0), ax=ax2, cmap='viridis')
    ax2.set_title('Spatial Dynamics')
    ax2.set_xlabel('Width Blocks')
    ax2.set_ylabel('Height Blocks')
    
    plt.tight_layout()
    plt.savefig(f"Dynamic_{name}_{load_step}.png")

if __name__ == "__main__":
    block_size = (3, 4, 4)
    blocked_latents = load_and_block_data(block_size)
    
    # 计算时间和空间动态性
    temporal_dynamics, spatial_dynamics = compute_feature_dynamics(blocked_latents)
    
    # 可视化结果
    visualize_dynamics(temporal_dynamics, spatial_dynamics, block_size)