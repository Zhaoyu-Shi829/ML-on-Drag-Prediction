#!/bin/bash

# Environmental varaibles
export PYTHONUSERBASE=/lscratch/zhaoyus/python_packages
echo $PYTHONUSERBASE

# NVIDIA config
export NVHPC=/lscratch/zhaoyus/Nvidia/hpc-sdk
export NVHPC_ROOT=/lscratch/zhaoyus/Nvidia/hpc-sdk/Linux_x86_64/21.9
export PATH=/lscratch/zhaoyus/Nvidia/hpc-sdk/Linux_x86_64/21.9/cuda/bin:$PATH
export PATH=/lscratch/zhaoyus/Nvidia/hpc-sdk/Linux_x86_64/21.9/comm_libs/mpi/bin:$PATH
export LD_LIBRARY_PATH=/lscratch/zhaoyus/Nvidia/hpc-sdk/Linux_x86_64/21.9/cuda/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/lscratch/zhaoyus/Nvidia/hpc-sdk/Linux_x86_64/21.9/cuda/11.4/extras/CUPT1/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/lscratch/zhaoyus/Nvidia/hpc-sdk/Linux_x86_64/21.9/comm_libs/mpi/lib:$LD_LIBRARY_PATH
export CPATH=/lscratch/zhaoyus/Nvidia/hpc-sdk/Linux_x86_64/21.9/comm_libs/mpi/include:$CPATH

export OPAL_PREFIX=/lscratch/zhaoyus/Nvidia/hpc-sdk/21.9/comm_libs/mpi
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/lscratch/zhaoyus/Nvidia/hpc-sdk/Linux_x86_64/21.9/cuda/11.4/

# Run python script
python3 CNN_BO_zy_gpu.py
