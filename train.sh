#!/bin/bash

set -e
set -x

data_dir="./data/imagenet/"
output_dir="./output/sslt_r50_100ep"

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port 12348  --nproc_per_node=1 \
  train_net.py \
  --data_dir ${data_dir} \
  --output_dir ${output_dir} \
  --batch_size 2 \
  --num_queries 2 \
  --epoch 1
