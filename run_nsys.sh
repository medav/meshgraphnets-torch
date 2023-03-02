#!/bin/bash

# Env vars:
# CASIO = /path/to/casio
# APP = name of application
# MODE = {ncu, nsys, prof, bench}
# PLAT = {a100, v100, p100}
# DEV = {cuda:0, cuda:1, ...}
# BS = batch size
# NW = number of warmup steps
# NI = number of benchmark iterations

set -x
set -e

RUN_NSYS=${RUN_NSYS:-yes}
[ "$RUN_NSYS" = "no" ] && exit 0

ODIR=nsys-output

mkdir -p $ODIR

NSYS=/opt/nvidia/nsight-systems/2022.1.3/bin/nsys
[ -x "$NSYS" ] || NSYS=/usr/local/cuda/bin/nsys
[ -x "$NSYS" ] || NSYS=nsys
echo "Using nsys: $NSYS"

MODE=nsys $NSYS profile \
    -t cuda,cudnn,cublas \
    -o $ODIR/nsys-rep \
    -f true \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    $*

