#!/bin/bash/

export CUDA_VISIBLE_DEVICES=0,1,2,3
PORT=29501
NUM_GPU=4

CFG='configs/uformer_b_str.py'
RESUME='ckpt/uformer_b_tmim_str/latest.pth'

python -m torch.distributed.launch --master_port ${PORT} --nproc_per_node=${NUM_GPU} demo.py --cfg ${CFG} --resume ${RESUME} --test-dir path/to/image/folder --visualize-dir path/to/save/folder





