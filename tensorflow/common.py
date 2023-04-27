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
"""Commonly used data structures and functions."""

import enum
import tensorflow.compat.v1 as tf


class NodeType(enum.IntEnum):
  NORMAL = 0
  OBSTACLE = 1
  AIRFOIL = 2
  HANDLE = 3
  INFLOW = 4
  OUTFLOW = 5
  WALL_BOUNDARY = 6
  SIZE = 9


def triangles_to_edges(faces):
  """Computes mesh edges from triangles."""
  # collect edges from triangles
  edges = tf.concat([faces[:, 0:2],
                     faces[:, 1:3],
                     tf.stack([faces[:, 2], faces[:, 0]], axis=1)], axis=0)
  # those edges are sometimes duplicated (within the mesh) and sometimes
  # single (at the mesh boundary).
  # sort & pack edges as single tf.int64
  receivers = tf.reduce_min(edges, axis=1)
  senders = tf.reduce_max(edges, axis=1)
  packed_edges = tf.bitcast(tf.stack([senders, receivers], axis=1), tf.int64)
  # remove duplicates and unpack
  unique_edges = tf.bitcast(tf.unique(packed_edges)[0], tf.int32)
  senders, receivers = tf.unstack(unique_edges, axis=1)
  # create two-way connectivity
  return (tf.concat([senders, receivers], axis=0),
          tf.concat([receivers, senders], axis=0))


def construct_world_edges(world_pos, node_type):

  deformable_idx = tf.where(tf.not_equal(node_type[:, 0], NodeType.OBSTACLE))
  actuator_idx = tf.where(tf.equal(node_type[:, 0], NodeType.OBSTACLE))
  B = tf.squeeze(tf.gather(world_pos, deformable_idx))
  A = tf.squeeze(tf.gather(world_pos, actuator_idx))

  A = tf.cast(A, tf.float64)
  B = tf.cast(B, tf.float64)


  thresh = 0.03

  dists = squared_dist(A, B)
  rel_close_pair_idx = tf.where(tf.math.less(dists, thresh ** 2))


  close_pair_actuator = tf.gather(actuator_idx, rel_close_pair_idx[:,0])
  close_pair_def = tf.gather(deformable_idx, rel_close_pair_idx[:,1])
  close_pair_idx = tf.concat([close_pair_actuator, close_pair_def], 1)

  senders, receivers = tf.unstack(close_pair_idx, 2, axis=1)


  return (tf.concat([senders, receivers], axis=0),
          tf.concat([receivers, senders], axis=0))

def squared_dist(A, B):

  row_norms_A = tf.reduce_sum(tf.square(A), axis=1)
  row_norms_A = tf.reshape(row_norms_A, [-1, 1])  # Column vector.

  row_norms_B = tf.reduce_sum(tf.square(B), axis=1)
  row_norms_B = tf.reshape(row_norms_B, [1, -1])  # Row vector.



  return row_norms_A - 2 * tf.matmul(A, B, False, True) + row_norms_B
