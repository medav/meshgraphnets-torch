#!/bin/bash

MODEL=deforming_plate
ODIR=/data/checkpoints/hyperel_deforming_plate
DATA=/data/deforming_plate

mkdir -p $ODIR
chmod -r 777 $ODIR

./scripts/tf1-docker.sh python train.py --model=$MODEL --checkpoint_dir=$ODIR --dataset_dir=$DATA
