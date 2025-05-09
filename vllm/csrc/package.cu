#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/all.h>

void fuck_test(torch::Tensor const& A, torch::Tensor const& perm){
    std::cout<<A.sizes()<<std::endl;
    return ;
}