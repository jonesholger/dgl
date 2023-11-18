#!/bin/bash

ml rocm/5.6.0 
ml cmake/3.23.1
ml python/3.9.12

BASE_DIR=`pwd`
echo "BASE_DIR: $BASE_DIR"

if [[ "$HOSTNAME" =~ .*"tioga".* || "$HOSTNAME" =~ .*"rzvernal".* ]]; then
ml rocmcc/5.6.0-magic
venv_path=$HOME/workspace/venv/dgl_use_torch_rocm
source $venv_path/bin/activate
BUILD_DIR=$BASE_DIR/build_amd
fi

if [[ "$HOSTNAME" =~ .*"corona".* ]]; then
venv_path=$HOME/workspace/venv/dgl_use_torch_rocm_corona
source $venv_path/bin/activate
BUILD_DIR=$BASE_DIR/build_amd_corona
fi
echo "BUILD_DIR: $BUILD_DIR"
export DGLBACKEND=pytorch
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=48
export TF_FORCE_GPU_ALLOW_GROWTH=true
export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.9,max_split_size_mb:128

TOP_LEVEL=`pwd`
TORCH_BASE=`$TOP_LEVEL/script/torch_path.py`
echo "torch: $TORCH_BASE"
export LD_LIBRARY_PATH=$TORCH_BASE/lib:$LD_LIBRARY_PATH

rm -r build $BUILD_DIR tensoradapter/pytorch/build
mkdir -p $BUILD_DIR
mkdir -p $BASE_DIR/build

$BASE_DIR/script/build_dgl_amd_pip.sh $BUILD_DIR

pushd $BUILD_DIR
make -j 24
./bin/runUnitTests
popd

pushd $BASE_DIR/build
cp $BUILD_DIR/lib/* .
cp -r $BUILD_DIR/tensoradapter .
popd

pushd $BASE_DIR/python
python3 setup.py clean --all
rm dist/dgl-1.2*.whl
python3 setup.py bdist_wheel
pip3 install --force-reinstall --no-deps dist/dgl-1.2*.whl
popd 

python3 $BASE_DIR/tutorials/blitz/2_dglgraph.py

#pip3 install -r $BASE_DIR/requirements_after_dgl_install.txt

