#!/bin/bash
ml rocm/5.5.0 
ml rocmcc/5.5.0-magic
ml cmake/3.24.2
COMP_HIPCC_VER=5.5.0
HIP_LIBRARIES_BASE=/usr/tce/packages/rocmcc/rocmcc-${COMP_HIPCC_VER}-magic/
TORCH_BASE=`./torch_path.py`
echo "torch: $TORCH_BASE"
BUILD_DIR=$1
echo "BUILD_DIR: $BUILD_DIR"
mkdir -p $BUILD_DIR
cd $BUILD_DIR
COMP_ARCH=gfx90a
cmake  \
    -DCMAKE_CXX_COMPILER=amdclang++ \
    -DCMAKE_C_COMPILER=amdclang \
    -DCMAKE_CXX_STANDARD="17" \
    -DENABLE_HIP=ON \
    -DCMAKE_HIP_ARCHITECTURES="${COMP_ARCH}" \
    -DBUILD_TORCH=ON \
    -DBUILD_SPARSE=OFF \
    -DBUILD_GRAPHBOLT=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_VERBOSE_MAKEFILE=OFF \
    -DCMAKE_INSTALL_PREFIX="../install_amd" \
    -DCMAKE_INSTALL_RPATH="$TORCH_BASE" \
    -DCMAKE_BUILD_RPATH="$TORCH_BASE" \
    -DHIPBLAS_DIR=${HIP_LIBRARIES_BASE}/hipblas/lib/cmake \
    -DHIPSPARSE_DIR=${HIP_LIBRARIES_BASE}/hipsparse/lib/cmake \
    ..
