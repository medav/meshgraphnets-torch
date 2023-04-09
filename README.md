MeshGraphNets (Written in PyTorch)
==================================
This repository is a PyTorch rewrite of [DeepMind's MeshGrapNets](https://github.com/deepmind/deepmind-research/tree/master/meshgraphnets)
and is intended to be as faithful to the original as possible.

**Disclaimer: This repo is under active development. There may be some staleness
/ some small tweaks here or there may have broken some things. Please ping Mike (davies@cs.wisc.edu)
with any questions.**

## Repo Overview
The model is split between the common elements [graphnet.py](graphnet.py) and
dataset specific models. Currently this repository has code for
`deforming_plate` and `cfd` from the original MGN repo. Additionally, each
model file comes with a torch dataset for loading samples which are post
processed from tfrecord into compressed Numpy format.

## Files
|Filename|Description|
|--------|-------|
| graphnet.py | Core GNN model used by all applications of MGN |
| cfd.py | Code for CFD model and dataset |
| cloth.py | Code for Cloth model and dataset |
| deforming_plate.py | Code for Deforming Plate model and dataset |
| infer_dp.py | Runs inference for deforming plate model on pre-processed data and reports performance |
| train_*.py | Runs training for the models (**These are stale! May not work!**) |
| test/* | Collection of small scale test scripts (**Many are stale; use with caution**) |
| unsorted_segsum/* | Manual implementation of TensorFlow's UnsortedSegmentSum for CUDA |
| scripts/download_dataset.sh | Copied from original TF meshgraphnets to download deepmind data |
| scripts/convert_dataset.py | Used to convert TFRecords into .npz format |
| scripts/create_infer_data.py | Generates pre-processed inference data (for perf benchmarks) |
| scripts/infer_bench.py | Runs inference benchmark on a model with given input data |
| cudaprofile.py | Python hooks to call cudaProfilerStart/Stop |
| run_ncu/nsys.sh | Scripts to run nvidia tools on these models |

## Suggested First Steps
```bash
# 1. Download dataset
$ ./scripts/download_dataset.sh deforming_plate data

# 2. Preprocess dataset
$ python ./scripts/convert_dataset.py deforming_plate train

# 3. Create inference data
$ python ./scripts/create_infer_data.py <model> <split> <batch_size>

# 4. Run inference benchmark
$ python ./scripts/infer_bench.py <model> <input_file> <num_iters>
```
