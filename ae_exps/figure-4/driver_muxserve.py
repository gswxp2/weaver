import yaml
import os
dataset_name = os.sys.argv[1]
file = os.path.expanduser("~/weaver/MuxServe/examples/basic/models.yaml")
with open(file, 'r') as stream:
    try:
        models = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
config = os.path.expanduser("~/weaver/ae_exps/figure-4/model_config_template.yaml")
with open(config, 'r') as stream:
    try:
        configs = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
rates = []
# for i in range(1,10,2):
#     rates.append(i)
if dataset_name == "azure":
    for i in range(1, 15, 3):
        rates.append(i)
    for i in range(15, 20, 1):
        rates.append(i)
elif dataset_name == "burst":
    for i in range(1, 11, 3):
        rates.append(i)
    for i in range(11, 16, 1):
        rates.append(i)
for rate in rates:
    models['models'][0]['rate'] = rate
    yaml.dump(models, open('models.yaml', 'w'))
    
    # os.system("python ~/weaver/MuxServe/muxserve/muxsched/workload_utils.py \
    # --dataset-source ~/weaver/ShareGPT_V3_unfiltered_cleaned_split.json \
    # --workload_info_from_yaml True \ 
    # --output-file workload.json \
    # --model-yaml models.yaml ")
    workloads_dir = "~/weaver/burst.json" if dataset_name == 'burst' else "~/weaver/azure.json"
    os.system(f"python ~/weaver/MuxServe/muxserve/muxsched/workload_utils.py \
    --dataset-source {workloads_dir} \
    --workload_info_from_yaml True \
    --runtime 200 \
    --output-file workload.json \
    --model-yaml models.yaml ")
    if dataset_name == "azure":
        os.environ["AVG_SEQ_LEN"] = "1024"
    else:
        os.environ["AVG_SEQ_LEN"] = "900"
    for config in ["mps", "temporal"]:
        mps = 50 if config == "mps" else 90
        configs['models'][0]['mps_percentage'] = [100, mps]
        configs['models'][1]['mps_percentage'] = [100, mps]
        yaml.dump(configs, open('model_config.yaml', 'w'))
        os.system(f"mkdir -p {dataset_name}/res_mux_{config}")
        # dump to local temp
        os.system(f"bash test_mux.sh > {dataset_name}/res_mux_{config}/{rate}.txt")