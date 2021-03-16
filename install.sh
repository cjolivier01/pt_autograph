#!/bin/bash
set -x
wget https://storage.googleapis.com/tpu-pytorch/wheels/torch_xla-1.8-cp37-cp37m-linux_x86_64.whl .
python3.7 -m pip install torch_xla-1.8-cp37-cp37m-linux_x86_64.whl torch==1.8
python3.7 -m pip install -r requirements.txt
