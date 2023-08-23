#!/bin/bash

ml rocm/5.5.0 
ml rocmcc/5.5.0-magic
ml cmake/3.24.2
ml python/3.9.12

venv_path=$HOME/workspace/venv/dgl_test
python3 -m venv $venv_path
source $venv_path/bin/activate

export DGLBACKEND=pytorch
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=48

BASE_DIR=`pwd`
echo "BASE_DIR: $BASE_DIR"

#pip3 install --pre --no-cache-dir https://download.pytorch.org/whl/nightly/rocm5.5/torch-2.1.0.dev20230807%2Brocm5.5-cp39-cp39-linux_x86_64.whl
#pip3 install --pre --no-cache-dir https://download.pytorch.org/whl/nightly/rocm5.5/torchaudio-2.1.0.dev20230807%2Brocm5.5-cp39-cp39-linux_x86_64.whl
#pip3 install --pre --no-cache-dir https://download.pytorch.org/whl/nightly/rocm5.5/torchvision-0.16.0.dev20230807%2Brocm5.5-cp39-cp39-linux_x86_64.whl
#pip3 install --pre --no-cache-dir https://download.pytorch.org/whl/nightly/pytorch_triton_rocm-2.1.0-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
#pip3 install --pre --no-cache-dir https://download.pytorch.org/whl/nightly/torchdata-0.7.0.dev20230807-py3-none-any.whl

#pip3 install --no-cache-dir tensorflow-rocm==2.11.0.540

pip3 install -r $BASE_DIR/rocm_requirements.txt

pip3 install -r $BASE_DIR/ampl_test_requirements_min.txt
python3 $BASE_DIR/script/patch_caffe2_targets.py
rm -r build build_amd tensoradapter/pytorch/build
mkdir -p $BASE_DIR/build_amd
mkdir -p $BASE_DIR/build

$BASE_DIR/script/build_dgl_amd_pip.sh $BASE_DIR/build_amd
pushd $BASE_DIR/build_amd
make -j 24
./bin/runUnitTests
popd
pushd $BASE_DIR/build
cp $BASE_DIR/build_amd/lib/* .
cp -r $BASE_DIR/build_amd/tensoradapter .
popd
pushd $BASE_DIR/python
python3 setup.py clean
rm dist/dgl-1.2*.whl
python3 setup.py bdist_wheel
pip3 install --force-reinstall --no-deps dist/dgl-1.2*.whl
popd 
python3 $BASE_DIR/tutorials/blitz/2_dglgraph.py
pip3 install -r $BASE_DIR/requirements_after_dgl_install.txt









