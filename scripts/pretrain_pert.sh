#!/bin/bash/

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
PORT=29501
NUM_GPU=8

CFG='configs/pert_tmim.py'
CKPT='pert_tmim'
python -m torch.distributed.launch --master_port ${PORT} --nproc_per_node=${NUM_GPU} train.py --cfg ${CFG} --ckpt-name ${CKPT} --save-log 

python test.py --cfg ${CFG} --ckpt-name ${CKPT}/latest.pth --save-log --visualize

