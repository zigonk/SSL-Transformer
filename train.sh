  GNU nano 4.8                                                                                        train.sh                                                                                                   #!/bin/bash

set -e
set -x

data_dir="./data/imagenet/"
output_dir="./output/sslt_r50_100ep"

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port 12348  --nproc_per_node=1 \
  train_net.py \
  --data_dir ${data_dir} \
  --output_dir ${output_dir} \
  --optimizer adam \
  --batch_size 64 \
  --num_queries 100 \
  --dec_layers 2 \
  --lr 0.01 \
  --epoch 1 \
  --dec_type cross-attn \
