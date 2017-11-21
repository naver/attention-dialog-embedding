#!/bin/bash

PROJECT_ROOT_PATH=`git rev-parse --show-toplevel`

autopep8 --in-place --jobs 32 --max-line-length 127 ${PROJECT_ROOT_PATH}/seq2b_attn/*.py
