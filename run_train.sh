#!/bin/bash
nohup python train.py --outdir=./training-runs \
    --cfg=diffit-256 \
    --data=./datasets/imagenet_9to4_1024x1024_256x256.zip \
    --gpus 1 \
    --batch-gpu 16 \
    --snap 1 \
    > train.log 2>&1 &

echo "PID: $!"
