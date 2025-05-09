
source ~/.bashrc
eval "$(/home/weaverae/miniforge3/condabin/conda shell.bash hook)"
conda activate weaver
export HF_ENDPOINT=https://hf-mirror.com
export HF_ENDPOINT=https://hf-mirror.com
killall python3 -9
killall python3 -9
killall python3 -9
output_prefix=$1
get_numa_affinity() {
    local gpu_id=$1
    nvidia-smi topo -m | awk -v gpu="GPU${gpu_id}" '$1 == gpu {print $(NF-1)}'
}
numa_sender=$(get_numa_affinity 0) 
numa_receiver=$(get_numa_affinity 1)
ROLE=SENDER CUDA_VISIBLE_DEVICES=0 GLOBAL_IDX=0 GLOBAL_WORLD=2  numactl -N $numa_sender -m $numa_sender python3 -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-8B --enforce-eager --num-scheduler-steps 8  > $output_prefix-sender.log 2>&1 &
ROLE=RECEIVER CUDA_VISIBLE_DEVICES=1 GLOBAL_IDX=1 GLOBAL_WORLD=2 numactl -N $numa_receiver -m $numa_receiver python3 ./llm_engine_example_besync.py -tp 1 --model meta-llama/Meta-Llama-3-8B  --num-scheduler-steps 8 --enforce-eager > $output_prefix-receiver.log 2>&1 &