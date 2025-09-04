#!/bin/bash
set -x

export PYTORCH_ENABLE_SDPA_FLASH_ATTENTION=1
export REQUESTED_GPUS=1
export TAG=$1
export MODEL=$2
export GPUS_PER_NODE=8
export MASTER_PORT=$(expr $RANDOM % 10000 + 10000)
export PARTITION='a01'

# 计算需要的节点数（每节点最多8张卡）
export NNODES=$(( ($REQUESTED_GPUS + $GPUS_PER_NODE - 1) / $GPUS_PER_NODE ))

# 确保请求的GPU数不超过最大限制
if [ $REQUESTED_GPUS -gt 32 ]; then
    echo "Warning: Maximum supported GPUs is 32. Setting to 32."
    export REQUESTED_GPUS=32
    export NNODES=4
fi

export WORLD_SIZE=$REQUESTED_GPUS
export PYTHONPATH=$PWD:$PYTHONPATH

# 添加MASTER_ADDR环境变量，在多节点环境中必需
export MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)

# 单节点情况
if [ $NNODES -eq 1 ]; then
    echo "Launching job with 1 node, $REQUESTED_GPUS total GPUs"
    
    srun \
        -p $PARTITION \
        -K \
        -N 1 \
        --job-name=ditango \
        --ntasks-per-node=$REQUESTED_GPUS \
        --gres=gpu:$REQUESTED_GPUS \
        --export=ALL \
        bash infer.sh
# 多节点情况
else
    # 计算最后一个节点使用的GPU数量
    export LAST_NODE_GPUS=$(( $REQUESTED_GPUS - ($NNODES - 1) * $GPUS_PER_NODE ))
    
    echo "Launching job with $NNODES nodes, $REQUESTED_GPUS total GPUs"
    echo "Using $GPUS_PER_NODE GPUs per full node, with $LAST_NODE_GPUS GPUs on the last node"
    
    # 如果最后一个节点使用的GPU数与前面节点相同，可以一次性启动所有节点
    if [ $LAST_NODE_GPUS -eq $GPUS_PER_NODE ]; then
        srun \
            -p $PARTITION \
            -K \
            -N $NNODES \
            --job-name=ditango \
            --ntasks-per-node=$GPUS_PER_NODE \
            --gres=gpu:$GPUS_PER_NODE \
            --export=ALL \
            bash infer.sh
    else
        # 先为前N-1个节点创建相同的配置（每节点8个GPU）
        if [ $(($NNODES - 1)) -gt 0 ]; then
            srun \
                -p $PARTITION \
                -K \
                -N $(($NNODES - 1)) \
                --job-name=ditango \
                --ntasks-per-node=$GPUS_PER_NODE \
                --gres=gpu:$GPUS_PER_NODE \
                --export=ALL \
                bash infer.sh &
        fi
        
        # 为最后一个节点创建特殊配置
        srun \
            -p $PARTITION \
            -K \
            -N 1 \
            --job-name=ditango \
            --ntasks-per-node=$LAST_NODE_GPUS \
            --gres=gpu:$LAST_NODE_GPUS \
            --export=ALL \
            bash infer.sh &
            
        # 等待所有任务完成
        wait
    fi
fi