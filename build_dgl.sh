#!/bin/bash

ml cuda/11.4.1
ENV_BASE=/usr/workspace/st/envs/coral/anaconda/envs/opence-1.8.0/
TORCH_BASE=$ENV_BASE/lib/python3.9/site-packages/torch/lib/
cmake \
    -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_C_COMPILER=gcc \
    -DUSE_CUDA=ON \
    -DBUILD_TORCH=ON \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_VERBOSE_MAKEFILE=ON \
    -DCUDA_ARCH_NAME="Volta" \
    -DCMAKE_INSTALL_PREFIX="../install_lassen" \
    -DCMAKE_INSTALL_RPATH="$ENV_BASE/lib:$TORCH_BASE" \
    -DCMAKE_BUILD_RPATH="$ENV_BASE/lib:$TORCH_BASE" \
    ..
