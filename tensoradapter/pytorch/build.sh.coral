#!/bin/bash
# Helper script to build tensor adapter libraries for PyTorch
set -e

mkdir -p build
mkdir -p $BINDIR/tensoradapter/pytorch
cd build

if [ $(uname) = 'Darwin' ]; then
	CPSOURCE=*.dylib
else
	CPSOURCE=*.so
fi

ENV_BASE=/usr/workspace/st/envs/coral/anaconda/envs/opence-1.8.0/

echo "CUDA ${CUDA_TOOLKIT_ROOT_DIR}"

export LD_LIBRARY_PATH=$ENV_BASE/lib:$LD_LIBRARY_PATH
export PKG_CONFIG_PATH=$ENV_BASE/lib/pkgconfig:$PKG_CONFIG_PATH

TORCH_BASE=$ENV_BASE/lib/python3.9/site-packages/torch/lib/
CMAKE_FLAGS="-DCMAKE_VERBOSE_MAKEFILE=ON -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_TOOLKIT_ROOT_DIR -DTORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST -DUSE_CUDA=$USE_CUDA -DCUDNN_LIBRARY_PATH=$ENV_BASE/lib -DCUDNN_INCLUDE_PATH=$ENV_BASE/include -DCMAKE_INSTALL_RPATH=$ENV_BASE/lib:$TORCH_BASE -DCMAKE_BUILD_RPATH=$ENV_BASE/lib:$TORCH_BASE "

if [ $# -eq 0 ]; then
	$CMAKE_COMMAND $CMAKE_FLAGS ..
	make -j
	cp -v $CPSOURCE $BINDIR/tensoradapter/pytorch
else
	for PYTHON_INTERP in $@; do
		TORCH_VER=$($PYTHON_INTERP -c 'import torch; print(torch.__version__.split("+")[0])')
		mkdir -p $TORCH_VER
		cd $TORCH_VER
		$CMAKE_COMMAND $CMAKE_FLAGS -DCMAKE_EXE_LINKER_FLAGS="-L $ENV_BASE/lib" -DPYTHON_INTERP=$PYTHON_INTERP ../..
		make -j
		cp -v $CPSOURCE $BINDIR/tensoradapter/pytorch
		cd ..
	done
fi
