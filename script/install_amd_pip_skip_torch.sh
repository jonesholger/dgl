#!/bin/bash

ml rocm/5.5.0 
ml rocmcc/5.5.0-magic
ml cmake/3.24.2

venv_path=$HOME/workspace/venv/dgl_test
source $venv_path/bin/activate
export DGLBACKEND=pytorch
BASE_DIR=`pwd`
echo "BASE_DIR: $BASE_DIR"
python3 $BASE_DIR/script/patch_caffe2_targets.py
rm -r build build_amd install_amd
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
python3 setup.py clean
rm dgl-1.2*.whl
pip3 wheel .
pip3 install --force-reinstall dgl-1.2*.whl
popd 
python3 $BASE_DIR/tutorials/blitz/2_dglgraph.py

