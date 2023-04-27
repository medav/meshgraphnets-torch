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
"""Model for Deforming Plate."""
import os

import sonnet as snt
import tensorflow as tf

import common
import core_model
import normalization

import numpy as np
import os


class Model(snt.AbstractModule):
  """Model for static cloth simulation."""

  def __init__(self, learned_model, name='Model'):
    super(Model, self).__init__(name=name)
    with self._enter_variable_scope():
      self._learned_model = learned_model
      self._output_normalizer = normalization.Normalizer(
          size=3, name='output_normalizer')
      self._node_normalizer = normalization.Normalizer(
          size=3+common.NodeType.SIZE, name='node_normalizer')
      self._edge_normalizer = normalization.Normalizer(
          size=8, name='edge_normalizer')  # 2*(3D coord  + length) = 8
    self._world_edge_normalizer = normalization.Normalizer(
        size=4, name='world_edge_normalizer') # 3D coord + length = 4

  def _build_graph(self, inputs, is_training):
    """Builds input graph."""

    ##### MESH EDGE FEATURES #####
    senders, receivers = common.triangles_to_edges(inputs['cells'])
    relative_mesh_pos = (tf.gather(inputs['mesh_pos'], senders) -
                         tf.gather(inputs['mesh_pos'], receivers))

    relative_world_pos_mesh_edges = (tf.gather(inputs['world_pos'], senders) -
                         tf.gather(inputs['world_pos'], receivers))

    mesh_edge_features = tf.concat([
        relative_world_pos_mesh_edges,
        tf.norm(relative_world_pos_mesh_edges, axis=-1, keepdims=True),
        relative_mesh_pos,
        tf.norm(relative_mesh_pos, axis=-1, keepdims=True)], axis=-1)

    mesh_edges = core_model.EdgeSet(
        name='mesh_edges',
        features=self._edge_normalizer(mesh_edge_features, is_training),
        receivers=receivers,
        senders=senders)

    ##### WORLD EDGE FEATURES #####
    world_senders, world_receivers = common.construct_world_edges(inputs['world_pos'], inputs['node_type'])
    relative_world_pos = (tf.gather(inputs['world_pos'], world_senders) -
                          tf.gather(inputs['world_pos'], world_receivers))


    relative_world_norm = tf.norm(relative_world_pos, axis=-1, keepdims=True)
    world_edge_features = tf.concat([
        relative_world_pos,
        relative_world_norm], axis=-1)

    world_edges = core_model.EdgeSet(
        name='world_edges',
        features=self._world_edge_normalizer(world_edge_features, is_training),
        receivers=world_receivers,
        senders=world_senders)

    ##### NODE FEATURES #####
    actuator_mask = tf.tile(tf.equal(inputs['node_type'][:, :], common.NodeType.OBSTACLE), [1, 3])
    nonzero_velocity = inputs['target|world_pos'] - inputs['world_pos']
    zero_velocity = inputs['world_pos'] - inputs['world_pos']
    velocity = tf.where(actuator_mask, nonzero_velocity, zero_velocity)
    node_type = tf.one_hot(inputs['node_type'][:, 0], common.NodeType.SIZE)
    node_features = tf.concat([velocity, node_type], axis=-1)


    return core_model.MultiGraph(
        node_features=self._node_normalizer(node_features, is_training),
        edge_sets=[mesh_edges, world_edges])

  def _build(self, inputs):
    graph = self._build_graph(inputs, is_training=False)
    per_node_network_output = self._learned_model(graph)
    return self._update(inputs, per_node_network_output)

  @snt.reuse_variables
  def loss(self, inputs):
    """L2 loss on position."""
    graph = self._build_graph(inputs, is_training=True)
    network_output = self._learned_model(graph)

    target_position_change = inputs['target|world_pos'] - inputs['world_pos']
    target_normalized = self._output_normalizer(target_position_change)

    # build loss
    loss_mask = tf.equal(inputs['node_type'][:, 0], common.NodeType.NORMAL)
    error = tf.reduce_sum((target_normalized - network_output)**2, axis=1)
    loss = tf.reduce_mean(error[loss_mask])
    return loss

  def _update(self, inputs, per_node_network_output):
    """Integrate model outputs."""
    predicted_pos_change = self._output_normalizer.inverse(per_node_network_output)
    # integrate forward
    cur_position = inputs['world_pos']
    gt_next_position = inputs['target|world_pos']

    object_mask = tf.tile(tf.equal(inputs['node_type'][:, :], common.NodeType.NORMAL), [1, 3])

    next_position = tf.where(object_mask, cur_position + predicted_pos_change, gt_next_position)


    return next_position
