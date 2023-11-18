#!/bin/bash
# Helper script to build tensor adapter libraries for PyTorch
# assumes we can inherit from top-level venv
COMP_HIPCC_VER=5.6.0
HIP_LIBRARIES_BASE=/usr/tce/packages/rocmcc/rocmcc-${COMP_HIPCC_VER}-magic/
mkdir -p build
mkdir -p $BINDIR/tensoradapter/pytorch
echo "BINDIR in tensoradapter build: $BINDIR"
echo `pwd` 
cd build

if [ $(uname) = 'Darwin' ]; then
	CPSOURCE=*.dylib
else
	CPSOURCE=*.so
fi

CUDA_TOOLKIT_ROOT_DIR=/opt/nvidia/hpc_sdk/Linux_x86_64/20.9/cuda/11.0/
#export LD_LIBRARY_PATH=$HIP_LIBRARIES_BASE/lib:$LD_LIBRARY_PATH

TORCH_BASE=`$BINDIR/../script/torch_path.py`
export LD_LIBRARY_PATH=$TORCH_BASE/lib:$LD_LIBRARY_PATH
echo "TORCH_BASE in tensoradapter build: $TORCH_BASE"
#CMAKE_FLAGS="-DCUDA_TOOLKIT_ROOT_DIR=$CUDA_TOOLKIT_ROOT_DIR -DTORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST -DUSE_CUDA=$USE_CUDA"
CMAKE_FLAGS="-DUSE_ROCM=ON"
#CMAKE_FLAGS="$CMAKE_FLAGS -DCMAKE_INSTALL_RPATH=$HIP_LIBRARIES_BASE/lib:$TORCH_BASE -DCMAKE_BUILD_RPATH=$HIP_LIBRARIES_BASE/lib:$TORCH_BASE" 
CMAKE_FLAGS="$CMAKE_FLAGS -DCMAKE_INSTALL_RPATH=$TORCH_BASE/lib:$TORCH_BASE:$HIP_LIBRARIES_BASE/lib: -DCMAKE_BUILD_RPATH=$TORCH_BASE/lib:$TORCH_BASE:$HIP_LIBRARIES_BASE/lib:" 
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
#		$CMAKE_COMMAND --trace-expand $CMAKE_FLAGS -DCMAKE_EXE_LINKER_FLAGS="-L $ROCM_BASE" -DPYTHON_INTERP=$PYTHON_INTERP ../..
#		$CMAKE_COMMAND $CMAKE_FLAGS -DCMAKE_EXE_LINKER_FLAGS="-L $HIP_LIBRARIES_BASE/lib " -DPYTHON_INTERP=$PYTHON_INTERP ../..
		$CMAKE_COMMAND $CMAKE_FLAGS -DCMAKE_EXE_LINKER_FLAGS="-L $TORCH_BASE/lib -L $HIP_LIBRARIES_BASE/lib" -DPYTHON_INTERP=$PYTHON_INTERP ../..
		make -j
		cp -v $CPSOURCE $BINDIR/tensoradapter/pytorch
		cd ..
	done
fi
