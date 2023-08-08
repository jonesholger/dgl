#!/bin/bash

ml rocm/5.5.0 
ml rocmcc/5.5.0-magic
ml cmake/3.24.2
ml python/3.9.12

venv_path=$HOME/workspace/venv/dgl_test
python3 -m venv $venv_path
source $venv_path/bin/activate
export DGLBACKEND=pytorch
BASE_DIR=`pwd`
echo "BASE_DIR: $BASE_DIR"

pip3 install --pre --no-cache-dir https://download.pytorch.org/whl/nightly/rocm5.5/torch-2.1.0.dev20230807%2Brocm5.5-cp39-cp39-linux_x86_64.whl
pip3 install --pre --no-cache-dir https://download.pytorch.org/whl/nightly/rocm5.5/torchaudio-2.1.0.dev20230807%2Brocm5.5-cp39-cp39-linux_x86_64.whl
pip3 install --pre --no-cache-dir https://download.pytorch.org/whl/nightly/rocm5.5/torchvision-0.16.0.dev20230807%2Brocm5.5-cp39-cp39-linux_x86_64.whl
pip3 install --pre --no-cache-dir https://download.pytorch.org/whl/nightly/pytorch_triton_rocm-2.1.0-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
pip3 install --pre --no-cache-dir https://download.pytorch.org/whl/nightly/torchdata-0.7.0.dev20230807-py3-none-any.whl

# use the following if you've downloaded the above files to the whl directory
#pushd $BASE_DIR/whl
#pip3 install --pre --no-cache-dir torch-2.1.0.dev20230807+rocm5.5-cp39-cp39-linux_x86_64.whl --find-links .
#pip3 install --pre --no-cache-dir torchaudio-2.1.0.dev20230807+rocm5.5-cp39-cp39-linux_x86_64.whl --find-links .
#pip3 install --pre --no-cache-dir torchvision-0.16.0.dev20230807+rocm5.5-cp39-cp39-linux_x86_64.whl --find-links .
#pip3 install --pre --no-cache-dir pytorch_triton_rocm-2.1.0-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl --find-links .
#pip3 install --pre --no-cache-dir torchdata-0.7.0.dev20230807-py3-none-any.whl --find-links .
#popd
pip3 install -r $BASE_DIR/requirements.txt
# do we need the following for torch 5.5 
python3 $BASE_DIR/script/patch_caffe2_targets.py
mkdir -p $BASE_DIR/build_amd
mkdir -p $BASE_DIR/install_amd
mkdir -p $BASE_DIR/build

$BASE_DIR/script/build_dgl_amd_pip.sh $BASE_DIR/build_amd
pushd $BASE_DIR/build_amd
make -j 24
make install
./bin/runUnitTests
popd
pushd $BASE_DIR/build
cp $BASE_DIR/build_amd/lib/* .
popd
pushd $BASE_DIR/python
rm dgl-1.2*.whl
pip3 wheel .
pip3 install --force-reinstall dgl-1.2*.whl
popd 
python3 $BASE_DIR/tutorials/blitz/2_dglgraph.py









