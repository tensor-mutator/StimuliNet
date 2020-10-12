"""
@author: Debajyoti Raychaudhuri

A TensorFlow implementation of FlowNetCS
"""

from __future__ import print_function, division, absolute_import
from tensorflow_addons.image import dense_image_warp
from typing import Tuple, Sequence
import tensorflow.compat.v1 as tf
import os
from ..network import Network
from ..FlowNetC import FlowNetC
from ..FlowNetS import FlowNetS
from ..mutator import Mutator

class FlowNetCS(Network):

      def __init__(self, image: Tuple[int, int], flow: Tuple[int, int], batch_norm: bool = True, trainable: bool = True) -> None:
          self._image = image
          self._flow = flow
          self._batch_norm = batch_norm
          self._trainable = trainable
          self._scope = 'FlowNetCS'

      def _build_graph_with_scope(self) -> tf.Graph:
          self.graph = tf.Graph()
          with self.graph.as_default():
               with tf.variable_scope(self._scope):
                    self._build_graph()
                    if self._trainable:
                       self.loss = type('loss', (object,), dict(input=self._flow_label, output=self._loss))
          return self.graph

      @property
      def graph_def(self) -> tf.GraphDef:
          return self.graph.as_graph_def()

      def _build_graph(self) -> None:
          Mutator.set_graph(self.graph)
          self._image_1 = tf.placeholder(shape=(None,) + self._image + (3,), dtype=tf.float32, name='image_1_cs')
          self._image_2 = tf.placeholder(shape=(None,) + self._image + (3,), dtype=tf.float32, name='image_2_cs')
          flownet_c = FlowNetC(self._image, self._flow, self._batch_norm, trainable=False)
          flownet_c_patch = tf.import_graph_def(flownet_c.graph_def,
                                                input_map={x.name: [self._image_1, self._image_2][i] for i, x in enumerate(flownet_c.inputs)},
                                                return_elements=list(map(lambda x: x.name, flownet_c.outputs)))
          flownet_s_input_tensor = self._compute_input_tensor_for_flownet_s(self._image_1, self._image_2, flownet_c_patch)
          flownet_s = FlowNetS(flownet_s_input_tensor.get_shape(), self._flow, self._batch_norm, trainable=self._trainable)
          if self._trainable:
             self._flow_label = tf.placeholder(dtype=tf.float32, shape=(None,) + self._image + (3,))
             self._flownet_cs_patch, self._loss = tf.import_graph_def(flownet_s.graph_def,
                                                          input_map={flownet_s.inputs[0].name: flownet_s_input_tensor,
                                                                     flownet_s.loss.input: self._flow_label},
                                                          return_elements=list(map(lambda x: x.name, flownet_s.outputs)) + [flownet_s.loss.output])
          else:
             self._flownet_cs_patch = tf.import_graph_def(flownet_s.graph_def,
                                                          input_map={flownet_s.inputs[0].name: flownet_s_input_tensor},
                                                          return_elements=list(map(lambda x: x.name, flownet_s.outputs)))

      def _compute_input_tensor_for_flownet_s(self, image_1: tf.Tensor, image_2: tf.Tensor, flow_out: tf.Tensor) -> tf.Tensor:
          warped = dense_image_warp(image_2, flow_out)
          brightness_error = Mutator.ChannelNorm()(image_1 - warped)
          return tf.concat([image_1, image_2, warped, flow_out * 0.05, brightness_error], axis=-1)

      @property
      def inputs(self) -> Sequence[tf.Tensor]:
          return [self._image_1, self._image_2]

      @property
      def outputs(self) -> Sequence[tf.Tensor]:
          return [self._flownet_cs_patch]

      def get_graph(self, dest: str = os.getcwd()) -> None:
          writer = tf.summary.FileWriter(dest, graph=self.graph)
          writer.close()
