#!/usr/bin/env python3
from pathlib import Path
import sys
import os
import shutil
import torch

p = Path(torch.__file__)
caffe2_cmake_path = str(p.parent) + '/share/cmake/Caffe2/'
caffe2_cmake_file = caffe2_cmake_path + 'Caffe2Targets.cmake'
caffe2_cmake_file_backup = caffe2_cmake_file + '.bak'
script_dir = os.path.dirname(os.path.realpath(__file__))
patch_file = script_dir + '/../patch/Caffe2Targets.cmake'
if os.path.isfile(caffe2_cmake_file):
    print('copy ' + caffe2_cmake_file + ' -> ' + caffe2_cmake_file_backup)
    shutil.copy(caffe2_cmake_file,caffe2_cmake_file_backup)
if os.path.isfile(patch_file):
    print('copy ' + patch_file + ' -> ' + caffe2_cmake_path)
    shutil.copy(patch_file,caffe2_cmake_path)





