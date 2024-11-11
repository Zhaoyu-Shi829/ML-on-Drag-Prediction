#!/bin/bash
 
# Environmental varaibles
USER_BASE_DIR="/lscratch/username"               # Replace 'username' with your actual username
CUDA_VERSION="11.4"                              # Specify CUDA version
HPC_SDK_VERSION="21.9"      

export PYTHONUSERBASE="$USER_BASE_DIR/python_packages"
echo $PYTHONUSERBASE

# NVIDIA config
export NVHPC="$USER_BASE_DIR/Nvidia/hpc-sdk"
export NVHPC_ROOT="$NVHPC/Linux_x86_64/$HPC_SDK_VERSION"
export PATH="$NVHPC_ROOT/cuda/bin:$PATH"
export PATH="$NVHPC_ROOT/comm_libs/mpi/bin:$PATH"
export LD_LIBRARY_PATH="$NVHPC_ROOT/cuda/lib64:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$NVHPC_ROOT/cuda/$CUDA_VERSION/extras/CUPTI/lib64:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$NVHPC_ROOT/comm_libs/mpi/lib:$LD_LIBRARY_PATH"
export CPATH="$NVHPC_ROOT/comm_libs/mpi/include:$CPATH"

export OPAL_PREFIX="$NVHPC/$HPC_SDK_VERSION/comm_libs/mpi"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$NVHPC_ROOT/cuda/$CUDA_VERSION/"

# Run python script
python3 CNN_BO.py
