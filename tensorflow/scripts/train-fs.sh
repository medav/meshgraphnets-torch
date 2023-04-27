#!/bin/bash

MODEL=cloth
ODIR=/data/checkpoints/cloth_flag_simple
DATA=/data/flag_simple

mkdir -p $ODIR
chmod -r 777 $ODIR

./scripts/tf1-docker.sh python train.py --model=$MODEL --checkpoint_dir=$ODIR --dataset_dir=$DATA
