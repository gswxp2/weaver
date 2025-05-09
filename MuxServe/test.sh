
source /home/gsw/.bashrc
eval "$(conda shell.bash hook)"
conda activate base
export HF_ENDPOINT=https://hf-mirror.com
# python3 /home/weaverae/weaverae/vllm/examples/llm_engine_example_be.py -tp 2 --model meta-llama/Meta-Llama-3-8B --gpu-memory-utilization 0.4 --num_scheduler_steps 10 --enforce-eager &
sleep 10;
# python3 /home/weaverae/weaverae/vllm/examples/llm_engine_example_be.py -tp 2 --model meta-llama/Meta-Llama-3-8B --gpu-memory-utilization 0.4 --num_scheduler_steps 10 --enforce-eager &
# python3 /home/weaverae/weaverae/vllm/examples/llm_engine_example_be.py -tp 2 --model meta-llama/Meta-Llama-3-8B --gpu-memory-utilization 0.4
# python -m muxserve.launch examples/basic/model_config.yaml     --nnodes=1 --node-rank=0 --master-addr=127.0.0.1     --nproc_per_node=2     --server-port 4145 --flexstore-port 50025         --workload-file examples/basic/sharedgpt_n3_rate_12_5_3.json
pid=$!
sleep 60;
kill $pid
# # check if there are any above processes running
# while true; do
#     if [[ $(ps aux|grep llm_engine_example|grep -v grep|wc -l) -eq 0 ]]; then
#         break
#     fi
#     sleep 1;
# done
