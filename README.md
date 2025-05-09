# Efficient Multi-LLM Serving with Workload Weaving

Welcome to the artifact repository for the USENIX ATC'25 accepted paper: "Efficient Multi-LLM Serving with Workload Weaving"! This artifact provides the necessary code and instructions to reproduce the experiments presented in our paper.

## Environment Requirements

**To artifact reviewers:** Please skip this section and proceed to "How to Run". The required environment has already been set up on the provided platform.

1.  At least 2 GPUs on the same machine with P2P (Peer-to-Peer) access enabled.
2.  Miniconda or Miniforge for package management.
3.  NVIDIA driver.
4.  CUDA Toolkit (tested with version 12.1).
5.  Root permission is required to run the NVIDIA MPS service.

## Compile and Install Weaver and MuxServe

First, create the Conda environments:
```bash
yes |conda create -n muxserve python=3.10
yes |conda create -n weaver python=3.10
```

Next, set up essential environment variables. Ensure `CUDA_HOME` points to your CUDA installation:
```bash
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CUDA_CXX=$CUDA_HOME/bin/nvcc
```

### Building MuxServe
Activate the `muxserve` environment and install dependencies:
```bash
conda activate muxserve
cd ~/weaver/MuxServe; pip3 install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121;
cd ~/weaver/flash-attention; pip3 install --no-build-isolation -e . 
cd ~/weaver/MuxServe-vLLM; pip3 install -r requirements.txt; pip3 install --no-build-isolation -e . 
cd ~/weaver/MuxServe; pip3 install --no-build-isolation -e . 
```

### Building Weaver
Activate the `weaver` environment:
```bash
conda activate weaver
```

**Optional: Patch PyTorch for Faster Graph Replay**

This section is an optional step to patch PyTorch, potentially improving graph replay performance.

*   **Prerequisites:**
    *   Set up your PyTorch build environment by following the official PyTorch repository instructions.

*   **Patching and compiling PyTorch:**
    The following command line clones PyTorch, checks out version v2.5.1, and applies `~/weaver/pytorch.patch`.
    ```bash
    # git clone https://github.com/pytorch/pytorch.git
    cd pytorch
    git checkout v2.5.1
    git apply ~/weaver/pytorch.patch;
    export _GLIBCXX_USE_CXX11_ABI=1
    export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"/home/weaver/miniforge3//"}

    CC=`which gcc-9` CXX=`which g++-9` CXXFLAGS='-Wno-maybe-uninitialized -Wno-uninitialized -Wno-free-nonheap-object -Wno-nonnull -std=c++17' CFLAGS='-Wno-maybe-uninitialized -Wno-uninitialized -Wno-free-nonheap-object -Wno-nonnull' USE_ROCM=0 TORCH_CUDA_ARCH_LIST="8.0" REL_WITH_DEB_INFO=1 USE_CUDA=1 MAX_JOBS=32 python setup.py develop
    ```

**Build vLLM with Custom FlashAttention for Weaver**
```bash
mkdir ~/weaver/vllm/vllm/vllm_flash_attn/
cd ~/weaver/vllm; pip3 install -r requirements-cuda.txt; VLLM_FLASH_ATTN_SRC_DIR=~/weaver/flash-attention python3 setup.py develop
```

## How to Run

Follow these steps to run the experiments:

1.  **Enable NVIDIA MPS (Multi-Process Service)**
    This is required for running MuxServe.
    ```bash
    cd ~/weaver/ae_exps/scripts; 
    sudo bash start_mps.sh ~/weaver/ae_exps/mps_log
    ```

2.  **Fix GPU/CPU frequencies for stable performance**
    ```bash
    sudo nvidia-smi -pm 1
    # The following lock graphics clock (lgc) command is an example for A100 GPUs. 
    # Adjust frequencies if necessary for your specific hardware.
    sudo nvidia-smi -lgc 1410,1410 
    sudo cpupower frequency-set -g performance
    ```

3.  **Run the experiments**:

    **To artifact reviewers:** Before starting, please ensure no other reviewer or user is working on the machine (e.g., by running the `nvidia-smi` command). 

    This script will run for several hours.
    ```bash
    bash run_all.sh
    ```

After the script completes, you can find the log files in the `~/weaver/ae_exps/figure-x/` directories. The script will also generate figures (`*.png`) based on these log files, which will be located in the same `figure-x` directories.

## Cite Our Paper

If you find our work useful, please consider citing our paper:

Shiwei Gao, Qing Wang, Shaoxun Zeng, Youyou Lu and Jiwu Shu. Weaver: Efficient Multi-LLM Serving with Workload Weaving. To appear in the 2025 USENIX Annual Technical Conference (USENIX ATC'25), Boston MA USA, July 2025.

