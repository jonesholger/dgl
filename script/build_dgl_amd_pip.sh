#!/bin/bash
if [ $# -eq 0 ]; then
    >&2 echo "No build directory provided: should be one level below dgl main dir"
    exit 1
fi
COMP_HIPCC_VER=5.6.0
ml rocm/$COMP_HIPCC_VER 
ml cmake/3.23.1

if [[ "$HOSTNAME" =~ .*"tioga".* || "$HOSTNAME" =~ .*"rzvernal".* ]]; then
ml rocmcc/$COMP_HIPCC_VER-magic
HIP_LIBRARIES_BASE=/usr/tce/packages/rocmcc/rocmcc-${COMP_HIPCC_VER}-magic/
COMP_ARCH=gfx90a
fi

if [[ "$HOSTNAME" =~ .*"corona".* ]]; then
HIP_LIBRARIES_BASE=/opt/rocm-${COMP_HIPCC_VER}/
COMP_ARCH=gfx906
fi

BUILD_DIR=$1
TOP_LEVEL=`pwd`
TORCH_BASE=`$TOP_LEVEL/script/torch_path.py`
echo "torch: $TORCH_BASE"
#export LD_LIBRARY_PATH=$TORCH_BASE/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$TORCH_BASE/lib:$LD_LIBRARY_PATH
#export ROCM_PATH=$TORCH_BASE
#    -DHIPBLAS_DIR=${HIP_LIBRARIES_BASE}/hipblas/lib/cmake \
#    -DHIPSPARSE_DIR=${HIP_LIBRARIES_BASE}/hipsparse/lib/cmake \
echo "BUILD_DIR: $BUILD_DIR"
mkdir -p $BUILD_DIR
cd $BUILD_DIR
cmake  \
    -DCMAKE_CXX_COMPILER=amdclang++ \
    -DCMAKE_C_COMPILER=amdclang \
    -DCMAKE_CXX_STANDARD="17" \
    -DENABLE_HIP=ON \
    -DCMAKE_HIP_ARCHITECTURES="${COMP_ARCH}" \
    -DBUILD_TORCH=Off \
    -DBUILD_SPARSE=OFF \
    -DBUILD_GRAPHBOLT=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_VERBOSE_MAKEFILE=OFF \
    -DCMAKE_INSTALL_PREFIX="../install_amd" \
    -DCMAKE_INSTALL_RPATH="$TORCH_BASE/lib" \
    -DCMAKE_BUILD_RPATH="$TORCH_BASE/lib" \
    -DHIPBLAS_DIR=${HIP_LIBRARIES_BASE}/hipblas/lib/cmake \
    -DHIPSPARSE_DIR=${HIP_LIBRARIES_BASE}/hipsparse/lib/cmake \
    ..
