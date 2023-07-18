#!/bin/bash
# Helper script to build tensor adapter libraries for PyTorch
ml rocm/5.5.0 
ml rocmcc/5.5.0-magic
ml python/3.9.12
ml cmake/3.24.2
source /usr/workspace/st/envs/amd/rocm.env
set -e

mkdir -p build
mkdir -p $BINDIR/tensoradapter/pytorch
cd build

if [ $(uname) = 'Darwin' ]; then
	CPSOURCE=*.dylib
else
	CPSOURCE=*.so
fi
ROCM_BASE=/opt/rocm-5.5.0/lib
#USE_CUDA="ON"
#CUDA_TOOLKIT_ROOT_DIR=/opt/nvidia/hpc_sdk/Linux_x86_64/20.9/cuda/11.0/
export LD_LIBRARY_PATH=$ROCM_BASE:$LD_LIBRARY_PATH
TORCH_BASE=/usr/workspace/st/envs/amd/pytorch-2.0.0/lib/python3.9/site-packages/torch/lib
#CMAKE_FLAGS="-DCUDA_TOOLKIT_ROOT_DIR=$CUDA_TOOLKIT_ROOT_DIR -DTORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST -DUSE_CUDA=$USE_CUDA"
CMAKE_FLAGS="$CMAKE_FLAGS -DCMAKE_INSTALL_RPATH=$ROCM_BASE:$TORCH_BASE -DCMAKE_BUILD_RPATH=$ROCM_BASE:$TORCH_BASE" 
CMAKE_FLAGS="$CMAKE_FLAGS -DCMAKE_C_COMPILER=amdclang -DCMAKE_CXX_COMPILER=amdclang++ -DAMDGPU_TARGETS=gfx90a" 

echo $CMAKE_FLAGS

if [ $# -eq 0 ]; then
	$CMAKE_COMMAND $CMAKE_FLAGS ..
	make -j
	cp -v $CPSOURCE $BINDIR/tensoradapter/pytorch
else
	for PYTHON_INTERP in $@; do
		TORCH_VER=$($PYTHON_INTERP -c 'import torch; print(torch.__version__.split("+")[0])')
		mkdir -p $TORCH_VER
		cd $TORCH_VER
		$CMAKE_COMMAND --trace-expand $CMAKE_FLAGS -DCMAKE_EXE_LINKER_FLAGS="-L $ROCM_BASE" -DPYTHON_INTERP=$PYTHON_INTERP ../..
		make -j
		cp -v $CPSOURCE $BINDIR/tensoradapter/pytorch
		cd ..
	done
fi
