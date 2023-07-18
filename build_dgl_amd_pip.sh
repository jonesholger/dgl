#!/bin/bash
ml rocm/5.5.0 
ml rocmcc/5.5.0-magic
ml python/3.9.12
ml cmake/3.24.2
TORCH_BASE=/usr/workspace/st/envs/amd/pytorch-2.0.0/lib/python3.9/site-packages/torch/lib
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
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_VERBOSE_MAKEFILE=ON \
    -DCMAKE_INSTALL_PREFIX="../install_amd" \
    -DCMAKE_INSTALL_RPATH="$TORCH_BASE" \
    -DCMAKE_BUILD_RPATH="$TORCH_BASE" \
    ..
