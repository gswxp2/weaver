# Figure 4 (a)(b)(e)(f) can be reproduced by following steps:
# Note that other subfigures in Figure 4 is conducted on the cloud L40s platform
# The results will be in ~/weaver/ae_exps/figure-4/[burst|azure]/[hot|cold].png

cd ~/weaver/ae_exps/figure-4
export HF_ENDPOINT=https://hf-mirror.com
conda activate weaver
# Not please don't use `python3`, otherwise it can be killed
python driver_weaver.py burst
python driver_weaver.py azure

conda activate muxserve
python driver_muxserve.py burst
python driver_muxserve.py azure

conda activate weaver
python digest_e2e.py burst
python digest_e2e.py azure


# Figure 5a can be reproduced by following steps:
# The results will be in ~/weaver/ae_exps/figure-5a/figure-5b.png

cd ~/weaver/ae_exps/figure-5a
export HF_ENDPOINT=https://hf-mirror.com
conda activate weaver
python driver_weaver.py

conda activate muxserve
python driver_muxserve.py

python digest_sensi.py 

# Figure 5b can be reproduced by following steps:
# The results will be in ~/weaver/ae_exps/figure-5b/result.png

cd ~/weaver/ae_exps/figure-5b
export HF_ENDPOINT=https://hf-mirror.com

conda activate weaver
python driver_weaver.py

python digest_sensi.py 

# Figure 6a can be reproduced by following steps:
# The results will be in ~/weaver/ae_exps/figure-6a/figure-6a.png
# The ablation study is done on the cloud L40s platform
# On A100 platform, the op splitting's has lesser benefits
cd ~/weaver/ae_exps/figure-6a
export HF_ENDPOINT=https://hf-mirror.com
conda activate weaver
python driver_weaver.py
python diges.py