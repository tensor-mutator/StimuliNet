"""
@author: Debajyoti Raychaudhuri

A TensorFlow implementation of FlowNetCSS
"""

from __future__ import print_function, division, absolute_import
from typing import Tuple, Sequence
import tensorflow.compat.v1 as tf
from tensorflow_addons.image import dense_image_warp
import os
from ..network import Network
from ..FlowNetCS import FlowNetCS
from ..FlowNetS import FlowNetS
from ..mutator import Mutator

class FlowNetCSS(Network):

      def __init__(self, image: Tuple[int, int], flow: Tuple[int, int], l2: float,
                   batch_norm: bool = True, trainable: bool = True) -> None:
          self._image = image
          self._flow = flow
          self._l2 = l2
          self._batch_norm = batch_norm
          self._trainable = trainable
          self._scope = 'FlowNetCSS'
          self._build_graph_with_scope()

      def _build_graph_with_scope(self) -> None:
          #self.graph = tf.Graph()
          #with self.graph.as_default():
          with tf.variable_scope(self._scope):
               self._build_graph()
               if self._trainable:
                  self.loss = type('loss', (object,), dict(input=self._flow_label, output=self._loss))

      def _build_graph(self) -> None:
          Mutator.scope(self._scope)
          #self._image_1 = tf.placeholder(shape=(None,) + self._image + (3,), dtype=tf.float32, name='image_1_css')
          #self._image_2 = tf.placeholder(shape=(None,) + self._image + (3,), dtype=tf.float32, name='image_2_css')
          flownet_cs = FlowNetCS(self._image, self._flow, self._l2, self._batch_norm, trainable=False)
          self._image_1, self._image_2 = flownet_cs.inputs[0], flownet_cs.inputs[1]
          flownet_s_input_tensor = self._compute_input_tensor_for_flownet_s(self._image_1, self._image_2, flownet_cs.outputs[0])
          #flownet_cs_patch = tf.import_graph_def(flownet_cs.graph_def,
          #                                       input_map={x.name: [self._image_1, self._image_2][i] for i, x in enumerate(flownet_cs.inputs)},
          #                                       return_elements=list(map(lambda x: x.name, flownet_cs.outputs)), name="FlowNetCS-Graph")
          #flownet_s_input_tensor = self._compute_input_tensor_for_flownet_s(self._image_1, self._image_2, flownet_cs_patch[0])
          Mutator.reset_scope(self._scope)
          flownet_s = FlowNetS(self._image, self._flow, self._l2, self._batch_norm, trainable=self._trainable)
          if self._trainable:
             #self._flow_label = tf.placeholder(dtype=tf.float32, shape=(None,) + self._flow + (2,))
             self._flow_label = flownet_s.loss.input
             flownet_s.inputs[0] = flownet_s_input_tensor
             self._flownet_css_patch, self._loss = flownet_s.outputs, flownet_s.loss.output
             #self._flownet_css_patch, self._loss = tf.import_graph_def(flownet_s.graph_def,
             #                                                         input_map={flownet_s.inputs[0].name: flownet_s_input_tensor,
             #                                                                    flownet_s.loss.input.name: self._flow_label},
             #                                                         return_elements=list(map(lambda x: x.name,
             #                                                                                  flownet_s.outputs)) + [flownet_s.loss.output.name],
             #                                                         name="FlowNetS-Graph")
          else:
             self._flownet_css_patch = flownet_s.outputs
             #self._flownet_css_patch = tf.import_graph_def(flownet_s.graph_def,
             #                                              input_map={flownet_s.inputs[0].name: flownet_s_input_tensor},
             #                                              return_elements=list(map(lambda x: x.name, flownet_s.outputs)),
             #                                              name="FlowNetS-Graph")

      def _compute_input_tensor_for_flownet_s(self, image_1: tf.Tensor, image_2: tf.Tensor, flow_out: tf.Tensor) -> tf.Tensor:
          warped = dense_image_warp(image_2, flow_out)
          brightness_error = Mutator.layers.ChannelNorm()(image_1 - warped)
          return tf.concat([image_1, image_2, warped, flow_out * 0.05, brightness_error], axis=-1)

      @property
      def inputs(self) -> Sequence[tf.Tensor]:
          return [self._image_1, self._image_2]

      @property
      def outputs(self) -> Sequence[tf.Tensor]:
          return [self._flownet_css_patch[0]]

      def get_graph(self, dest: str = os.getcwd()) -> None:
          writer = tf.summary.FileWriter(dest, graph=tf.get_default_graph())
          writer.close()
