#!/bin/bash
set -x

# Basic environment setup
export MASTER_ADDR=$(scontrol show jobid=$SLURM_JOB_ID | tr '=' ' ' | grep BatchHost | awk '{print $2}')
export NODE_RANK=$(expr $SLURM_PROCID / $NNODES)
export RANK=$SLURM_PROCID
export LOCAL_RANK=$(expr $SLURM_PROCID % $GPUS_PER_NODE)    

export CUDA_DEVICE_MAX_CONNECTIONS=1

# Mode selection
MODEL=${MODEL:-"1.3B"}
u=1
r=1
if [ "$MODEL" = '14B' ]; then
    size='1280*720'
else
    size='832*480'
fi


# Set script, args, and extra_args based on MODE
SCRIPT="generate.py"
ARGS="--task t2v-$MODEL \
    --size $size \
    --ckpt_dir ../models/Wan2.1-T2V-$MODEL \
    --ulysses_size 1 \
    --ring_size 1"
EXTRA_ARGS="--base_seed 42"

# Setup logging
Time=$(date +%Y%m%d_%H%M)

# 设置输出路径
export OUTPUT_DIR="./result/$TAG/$Time"
mkdir -p $OUTPUT_DIR

# Log filename includes modes
LOG_FILE="${OUTPUT_DIR}/Wan_${MODE}_rank${RANK}.log"

# Execute with logging based on rank
if [ "$RANK" -eq 0 ]; then
    # rank 0: redirect output to log file
    exec python $SCRIPT $ARGS $EXTRA_ARGS 2>&1 | tee $LOG_FILE
else
    # other ranks: execute without logging
    exec python $SCRIPT $ARGS $EXTRA_ARGS
fi