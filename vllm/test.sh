
source /home/gsw/.bashrc
eval "$(conda shell.bash hook)"
conda activate vllm
export HF_ENDPOINT=https://hf-mirror.com
# ROLE=SENDER CUDA_VISIBLE_DEVICES=0 GLOBAL_IDX=0 GLOBAL_WORLD=2  numactl -N 1 -m 1 python3 /home/weaverae/weaverae/vllm/examples/llm_engine_example_sync.py -tp 1 --model meta-llama/Meta-Llama-3-8B --enforce-eager > /dev/null &
# pid=$!
# ROLE=RECEIVER CUDA_VISIBLE_DEVICES=1 GLOBAL_IDX=1 GLOBAL_WORLD=2 numactl -N 1 -m 1 python3 /home/weaverae/weaverae/vllm/examples/llm_engine_example_besync.py -tp 1 --model meta-llama/Meta-Llama-3-8B  --enforce-eager 2>&1 
# pid2=$!
# ROLE=SENDER CUDA_VISIBLE_DEVICES=0 GLOBAL_IDX=0 GLOBAL_WORLD=2 numactl -N 1 -m 1 python3 -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-8B --enforce-eager > ../serving.log &
# ROLE=RECEIVER CUDA_VISIBLE_DEVICES=1 GLOBAL_IDX=1 GLOBAL_WORLD=2  timeout 240 numactl -N 1 -m 1 python3 /home/weaverae/weaverae/vllm/examples/llm_engine_example_besync.py -tp 1 --model meta-llama/Meta-Llama-3-8B  --enforce-eager 2>&1 &
ROLE=SENDER CUDA_VISIBLE_DEVICES=0 GLOBAL_IDX=0 GLOBAL_WORLD=2  numactl -N 1 -m 1  python3 -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-8B --enforce-eager &
ROLE=RECEIVER CUDA_VISIBLE_DEVICES=1 GLOBAL_IDX=1 GLOBAL_WORLD=2 numactl -N 1 -m 1 python3 /home/weaverae/weaverae/vllm/examples/llm_engine_example_besync.py -tp 1 --model meta-llama/Meta-Llama-3-8B  --enforce-eager 2>&1 &
pid=$!
wait $pid

# sleep 10;
# wait $pid
# kill -9 $pid2
# # check if there are any above processes running
# while true; do
#     if [[ $(ps aux|grep llm_engine_example|grep -v grep|wc -l) -eq 0 ]]; then
#         break
#     fi
#     sleep 1;
# done

# ROLE=SENDER CUDA_VISIBLE_DEVICES=0 GLOBAL_IDX=0 GLOBAL_WORLD=1   python3 -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-8B --enforce-eager > ../serving.log
# python3 vllm/benchmarks/benchmark_serving.py     --backend vllm     --dataset-name sharegpt     --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json     --model meta-llama/Meta-Llama-3-8B     --num-prompts 1000     --endpoint /v1/completions     --tokenizer meta-llama/Meta-Llama-3-8B     --save-result   --request-rate 15  --ignore-eos

# python3 vllm/benchmarks/benchmark_serving.py     --backend vllm     --dataset-name random    --model meta-llama/Meta-Llama-3-8B     --num-prompts 1000     --endpoint /v1/completions     --tokenizer meta-llama/Meta-Llama-3-8B     --save-result   --request-rate 15  --ignore-eos