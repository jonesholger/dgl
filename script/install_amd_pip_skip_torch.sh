#!/bin/bash

ml rocm/5.4.3 
ml rocmcc/5.4.3-magic
ml cmake/3.24.2

venv_path=$HOME/workspace/venv/dgl_test
source $venv_path/bin/activate

export DGLBACKEND=pytorch
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=48

BASE_DIR=`pwd`
echo "BASE_DIR: $BASE_DIR"
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
python3 setup.py clean --all
rm dist/dgl-1.2*.whl
python3 setup.py bdist_wheel
pip3 install --force-reinstall --no-deps dist/dgl-1.2*.whl
popd 
python3 $BASE_DIR/tutorials/blitz/2_dglgraph.py
pip3 install -r $BASE_DIR/requirements_after_dgl_install.txt

