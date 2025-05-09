
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate muxserve 
export HF_ENDPOINT=https://hf-mirror.com
killall python3 -9
killall python3 -9
killall python3 -9
killall python3 -9
# python3 /home/gsw/vllm-pipe/vllm/examples/llm_engine_example_be.py -tp 2 --model meta-llama/Meta-Llama-3-8B --gpu-memory-utilization 0.4 --num_scheduler_steps 10 --enforce-eager &
# sleep 10;
# python3 /home/gsw/vllm-pipe/vllm/examples/llm_engine_example_be.py -tp 2 --model meta-llama/Meta-Llama-3-8B --gpu-memory-utilization 0.4 --num_scheduler_steps 10 --enforce-eager &
# python3 /home/gsw/vllm-pipe/vllm/examples/llm_engine_example_be.py -tp 2 --model meta-llama/Meta-Llama-3-8B --gpu-memory-utilization 0.4
pushd ~/weaver/MuxServe 
python3 -m muxserve.launch /home/weaverae/weaver/ae_exps/figure-5a/model_config.yaml    --nnodes=1 --node-rank=0 --master-addr=127.0.0.1     --nproc_per_node=2     --server-port 4145 --flexstore-port 50025   --mps-dir ~/weaver/ae_exps/mps_log   --workload-file ~/weaver/ae_exps/figure-5a/workload.json
pid=$!

max_wait=300
elapsed=0

# 循环检查进程是否仍在运行
while kill -0 "$pid" 2>/dev/null; do
    # 如果已超过最大等待时间，则退出循环
    if [ "$elapsed" -ge "$max_wait" ]; then
        echo "Timeout reached. PID $pid is still running."
        break
    fi

    # 等待 1 秒然后继续检查
    sleep 1

    # 增加已等待的时间
    ((elapsed++))
done

# 检查进程是否已完成
if ! kill -0 "$pid" 2>/dev/null; then
    echo "PID $pid has finished."
else
    echo "PID $pid is still running after $max_wait seconds."
fi

killall python3 -9
# # check if there are any above processes running
# while true; do
#     if [[ $(ps aux|grep llm_engine_example|grep -v grep|wc -l) -eq 0 ]]; then
#         break
#     fi
#     sleep 1;
# done
