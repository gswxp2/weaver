#!/bin/bash
MODEL=${1:-"/mnt/afs/share/LLMCKPTs/huggyllama/llama-7b"}
GPU_UTILIZATION=${2:-0.35}
DATASET=${3:-"/mnt/afs/jfduan/datas/raw/sharegpt_v3/ShareGPT_V3_unfiltered_cleaned_split.json"}
LOGFILE=${3:-"log/spatial_7b.log"}

SM_UTIL=$(awk "BEGIN { printf \"%.0f\", $GPU_UTILIZATION * 100 }")
export CUDA_MPS_PIPE_DIRECTORY=/home/weaverae/weaverae/MuxServe/log/mps/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/home/weaverae/weaverae/MuxServe/log/mps/nvidia-log
export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=${SM_UTIL}


python benchmarks/benchmark_throughput.py \
    --dataset ${DATASET} \
    --model ${MODEL} \
    --gpu-memory-utilization ${GPU_UTILIZATION}
