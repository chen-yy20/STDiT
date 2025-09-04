import torch
import os 

def store_feature(feature: torch.Tensor, timestep: int, layer: int, name: str, rank:int = 0):
    
    dir_path = os.path.join("./stored_features/", f"r{rank}t{timestep}l{layer}")
    
    # 打印当前工作目录和目标目录
    print(f"Current working directory: {os.getcwd()}")
    print(f"Target directory: {dir_path}")
    
    # 检查目录是否存在
    if not os.path.exists(dir_path):
        print(f"Creating directory: {dir_path}")
        os.makedirs(dir_path, exist_ok=True)
    
    # 检查目录是否可写
    if not os.access(dir_path, os.W_OK):
        print(f"Directory {dir_path} is not writable!")
        return
        
    path = os.path.join(dir_path, name)
    print(f"Attempting to save to: {path}")
    
    try:
        torch.save(feature.detach().cpu(), path)  # 确保tensor在CPU上
        print(f"Successfully saved to {path}")
    except Exception as e:
        print(f"Error saving file: {str(e)}")
        raise
    
def split_latents_chunks(latents, temporal_chunks=2, height_chunks=2, width_chunks=2):
    """
    Split latents into chunks along temporal and spatial dimensions
    
    Args:
        latents: tensor of shape [C, T, H, W]
        temporal_chunks: number of temporal chunks
        height_chunks: number of splits for height dimension
        width_chunks: number of splits for width dimension
    
    Returns:
        chunks: list of tensor chunks
        chunk_info: dict containing splitting information
    """
    C, T, H, W = latents.shape
    
    # Verify divisions are valid
    assert T % temporal_chunks == 0, f"Temporal dimension {T} not divisible by {temporal_chunks}"
    assert H % height_chunks == 0, f"Height {H} not divisible by {height_chunks}"
    assert W % width_chunks == 0, f"Width {W} not divisible by {width_chunks}"
    
    # Calculate chunk sizes
    t_chunk_size = T // temporal_chunks
    h_chunk_size = H // height_chunks
    w_chunk_size = W // width_chunks
    
    chunks = []
    for t in range(temporal_chunks):
        for h in range(height_chunks):
            for w in range(width_chunks):
                # Extract chunk
                t_start = t * t_chunk_size
                t_end = (t + 1) * t_chunk_size
                h_start = h * h_chunk_size
                h_end = (h + 1) * h_chunk_size
                w_start = w * w_chunk_size
                w_end = (w + 1) * w_chunk_size
                
                chunk = latents[:, t_start:t_end, h_start:h_end, w_start:w_end]
                chunk = chunk.unsqueeze(0)
                chunks.append(chunk)
    
    chunk_info = {
        'original_shape': latents.shape,
        'temporal_chunks': temporal_chunks,
        'height_chunks': height_chunks,
        'width_chunks': width_chunks,
        't_chunk_size': t_chunk_size,
        'h_chunk_size': h_chunk_size,
        'w_chunk_size': w_chunk_size
    }
    
    return chunks, chunk_info

def merge_latents_chunks(chunks, chunk_info):
    """
    Merge chunks back to original tensor
    """
    C, T, H, W = chunk_info['original_shape']
    temporal_chunks = chunk_info['temporal_chunks']
    height_chunks = chunk_info['height_chunks']
    width_chunks = chunk_info['width_chunks']
    t_chunk_size = chunk_info['t_chunk_size']
    h_chunk_size = chunk_info['h_chunk_size']
    w_chunk_size = chunk_info['w_chunk_size']
    
    # 添加调试信息
    expected_chunks = temporal_chunks * height_chunks * width_chunks
    actual_chunks = len(chunks)
    print(f"Expected chunks: {expected_chunks}, Actual chunks: {actual_chunks}")
    print(f"Chunk info: {chunk_info}")
    
    # Initialize output tensor
    merged = torch.zeros(chunk_info['original_shape'], device=chunks[0].device)
    
    chunk_idx = 0
    for t in range(temporal_chunks):
        for h in range(height_chunks):
            for w in range(width_chunks):
                t_start = t * t_chunk_size
                t_end = (t + 1) * t_chunk_size
                h_start = h * h_chunk_size
                h_end = (h + 1) * h_chunk_size
                w_start = w * w_chunk_size
                w_end = (w + 1) * w_chunk_size
                
                current_chunk = chunks[chunk_idx].squeeze(0)
                merged[:, t_start:t_end, h_start:h_end, w_start:w_end] = current_chunk
                chunk_idx += 1
    
    return merged