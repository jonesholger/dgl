#!/bin/bash
if [ $# -eq 0 ]; then
    >&2 echo "No build directory provided: should be one level below dgl main dir"
    exit 1
fi
ml rocm/5.4.3 
ml rocmcc/5.4.3-magic
ml cmake/3.24.2
COMP_HIPCC_VER=5.4.3
HIP_LIBRARIES_BASE=/usr/tce/packages/rocmcc/rocmcc-${COMP_HIPCC_VER}-magic/
BUILD_DIR=$1
TOP_LEVEL=`pwd`
TORCH_BASE=`$TOP_LEVEL/script/torch_path.py`
echo "torch: $TORCH_BASE"

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
    -DCMAKE_INSTALL_RPATH="$HIP_LIBRARIES_BASE/lib" \
    -DCMAKE_BUILD_RPATH="$HIP_LIBRARIES_BASE/lib" \
    -DHIPBLAS_DIR=${HIP_LIBRARIES_BASE}/hipblas/lib/cmake \
    -DHIPSPARSE_DIR=${HIP_LIBRARIES_BASE}/hipsparse/lib/cmake \
    ..
