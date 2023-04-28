#!/bin/bash

[ -x "$MGN_DATA" ] || MGN_DATA=$(realpath ../data)

./scripts/tf1-docker.sh python np_infer.py $1 $2
