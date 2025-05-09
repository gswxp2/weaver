#assume you have conda/mamba installed to manage the environment
yes |conda create -n muxserve python=3.10
yes |conda create -n weaver python=3.10

export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CUDA_CXX=$CUDA_HOME/bin/nvcc
conda activate muxserve

cd ~/weaver/MuxServe; pip3 install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121;
cd ~/weaver/flash-attention; pip3 install --no-build-isolation -e . 
cd ~/weaver/MuxServe-vLLM; pip3 install -r requirements.txt; pip3 install --no-build-isolation -e . 
cd ~/weaver/MuxServe; pip3 install --no-build-isolation -e . 

conda activate weaver

# Optional, clone pytorch and apply the custom patch, which brings a little speedup to the graph replay.
# git clone https://github.com/pytorch/pytorch.git; git checkout v2.5.1; git apply ~/weaver/pytorch.patch;
# setup pytorch build env following the instructions in the original repo
#
# compile pytorch with the custom patch
# export _GLIBCXX_USE_CXX11_ABI=1
# export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"/home/weaver/miniforge3//"}
# CC=`which gcc-9` CXX=`which g++-9` CXXFLAGS='-Wno-maybe-uninitialized -Wno-uninitialized -Wno-free-nonheap-object -Wno-nonnull -I/usr/local/cuda-12.1/include -std=c++17' CFLAGS='-Wno-maybe-uninitialized -Wno-uninitialized -Wno-free-nonheap-object -Wno-nonnull -I/usr/local/cuda-12.1/include' USE_ROCM=0 TORCH_CUDA_ARCH_LIST="8.0" REL_WITH_DEB_INFO=1 USE_CUDA=1 MAX_JOBS=32 python setup.py develop
mkdir ~/weaver/vllm/vllm/vllm_flash_attn/
cd ~/weaver/vllm; pip3 install -r requirements-cuda.txt; VLLM_FLASH_ATTN_SRC_DIR=~/weaver/flash-attention python3 setup.py develop


