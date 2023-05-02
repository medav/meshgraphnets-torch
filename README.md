MeshGraphNets (Written in PyTorch)
==================================
This repository is a PyTorch rewrite of [DeepMind's MeshGrapNets](https://github.com/deepmind/deepmind-research/tree/master/meshgraphnets)
and is intended to be as faithful to the original as possible.

**Disclaimer: This repo is under active development. There may be some staleness
/ some small tweaks here or there may have broken some things. Please ping Mike
(davies@cs.wisc.edu) with any questions.**

## Repo Overview
The model is split between the common elements [graphnet.py](graphnet.py) and
dataset specific models. Currently this repository has code for
`flag_simple`, `deforming_plate` and `cylinder_flow` from the original MGN repo. Additionally,
each model file comes with a torch dataset for loading samples which are post
processed from tfrecord into compressed Numpy format.

## Files
|Filename|Description|
|--------|-------|
| graphnet.py | Core GNN model used by all applications of MGN |
| incomprns.py | Code for Incompressible Navier-Stokes model |
| cloth.py | Code for Cloth model |
| hyper.py | Code for Hyper-Elasticity model |
| unsorted_segsum/* | Manual implementation of TensorFlow's UnsortedSegmentSum for CUDA |
| gather_concat/* | Implementation of a fused Gather+Concat for optimized forward pass |
| scripts/download_dataset.sh | Copied from original TF meshgraphnets to download deepmind data |
| scripts/convert_dataset | Used to convert TFRecords into .npz format |
| scripts/create_infer_data | Generates pre-processed inference data (for perf benchmarks) |
| scripts/infer_bench | Runs inference benchmark on a model with given input data |
| scripts/infer_bench | Runs torch profiler on a model (inference) with given input data |
| run_ncu.sh and nsys.sh | Scripts to run nvidia tools on these models |
| tensorflow/* | Original TensorFlow code modified to run identical input data for performance comparison |
| test/* | Collection of small scale test scripts (**Many are stale; use with caution**) |

## Suggested First Steps
```bash
# 1. Download dataset
$ ./scripts/download_dataset.sh deforming_plate data

# 2. Preprocess dataset
$ python ./scripts/convert_dataset deforming_plate train

# 3. Create inference data
$ python ./scripts/create_infer_data <dataset> <split> <batch_size>

# 4. Run inference benchmark
$ python ./scripts/infer_bench <dataset> <input_file> <num_iters>
```
