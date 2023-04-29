# Lint as: python3
# pylint: disable=g-bad-file-header
# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Utility functions for reading the datasets."""

import functools
import json
import os
import numpy as np

import tensorflow.compat.v1 as tf

from common import NodeType

name_map = {
    'target_world_pos': 'target|world_pos',
    'target_velocity': 'target|velocity',
    'prev_world_pos': 'prev|world_pos'
}

ignore_fields = {
    'node_offs',
    'cell_offs',
    'srcs',
    'dsts',
}

# TensorFlow MGN expects the following fields:
# {
# 'cells': <tf.Tensor 'inputs:0' shape=(?, 3) dtype=int32>,
# 'mesh_pos': <tf.Tensor 'inputs_1:0' shape=(?, 2) dtype=float32>,
# 'node_type': <tf.Tensor 'inputs_2:0' shape=(?, 1) dtype=int32>,
# 'velocity': <tf.Tensor 'inputs_5:0' shape=(?, 2) dtype=float32>,
# 'target|velocity': <tf.Tensor 'inputs_4:0' shape=(?, 2) dtype=float32>,
# 'pressure': <tf.Tensor 'inputs_3:0' shape=(?, 1) dtype=float32>
# }

def gen_from_npz(path):
    num_files = len([
        name for name in os.listdir(path)
        if os.path.isfile(os.path.join(path, name))
    ])

    for i in range(num_files):
        data = np.load(os.path.join(path, f'{i}.npz'))
        data = {
            name_map.get(k, k): v
            for k, v in data.items()
            if k not in ignore_fields
        }

        data['node_type'] = np.expand_dims(data['node_type'], 1)
        yield data


def convert_shape(shape):
    if len(shape) == 1: return [None]
    return [None] + list(shape[1:])

def load_dataset(path, split=None, float_type=tf.float32):
    data = next(gen_from_npz(path))
    fields = list(data.keys())

    dtypes = {
        field:float_type if data[field].dtype == np.float32 else tf.int32
        for field in fields
    }

    shapes = {
        field: tf.TensorShape(convert_shape(data[field].shape))
        for field in fields
    }

    return tf.data.Dataset.from_generator(
        lambda: gen_from_npz(path),
        output_types=dtypes,
        output_shapes=shapes)


def batch_dataset(ds, batch_size):
    """Batches input datasets."""
    shapes = ds.output_shapes
    types = ds.output_types

    def renumber(buffer, frame):
        nodes, cells = buffer
        new_nodes, new_cells = frame
        return nodes + new_nodes, tf.concat([cells, new_cells+nodes], axis=0)

    def batch_accumulate(ds_window):
        out = {}
        for key, ds_val in ds_window.items():
            initial = tf.zeros((0, shapes[key][1]), dtype=types[key])
            if key == 'cells':
                # renumber node indices in cells
                num_nodes = ds_window['node_type'].map(lambda x: tf.shape(x)[0])
                cells = tf.data.Dataset.zip((num_nodes, ds_val))
                initial = (tf.constant(0, tf.int32), initial)
                _, out[key] = cells.reduce(initial, renumber)
            else:
                def merge(prev, cur): return tf.concat([prev, cur], axis=0)
                out[key] = ds_val.reduce(initial, merge)
        return out

    if batch_size > 1:
        ds = ds.window(batch_size, drop_remainder=True)
        ds = ds.map(batch_accumulate, num_parallel_calls=8)
    return ds
