import yaml
import os

file = os.path.expanduser("~/weaver/MuxServe/examples/basic/models.yaml")
with open(file, 'r') as stream:
    try:
        models = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
config = "./model_config_template.yaml"
with open(config, 'r') as stream:
    try:
        configs = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
rates = []
# for i in range(1,10,2):
#     rates.append(i)
for i in [1,2,3.3,5]:
    rates.append(i)
for workloads in ["burst"]:
    for rate in [1,2,3.3,5]:
        models['models'][0]['rate'] = 10
        models['models'][1]['rate'] = rate
        yaml.dump(models, open('models.yaml', 'w'))
        
        # os.system("python /home/gsw/vllm-pipe/MuxServe/muxserve/muxsched/workload_utils.py \
        # --dataset-source /home/gsw/vllm-pipe/ShareGPT_V3_unfiltered_cleaned_split.json \
        # --workload_info_from_yaml True \ 
        # --output-file workload.json \
        # --model-yaml models.yaml ")
        workloads_dir = os.path.expanduser("~/weaver/burst.json")
        os.system(f"python ~/weaver/MuxServe/muxserve/muxsched/workload_utils.py \
        --dataset-source {workloads_dir} \
        --workload_info_from_yaml True \
        --runtime 200 \
        --output-file workload.json \
        --model-yaml models.yaml ")
        for config in ["mps","temporal"]:
            # we change the qps of llm-0 from 5 to 10
            mps = 50 if config == "mps" else 90
            configs['models'][0]['mps_percentage'] = [100, mps]
            configs['models'][1]['mps_percentage'] = [100, mps]
            yaml.dump(configs, open('model_config.yaml', 'w'))
            os.system(f"mkdir -p res_mux_{config}_{workloads}")
            # dump to local temp
            os.system(f"bash test_mux.sh > res_mux_{config}_{workloads}/{rate}.txt")