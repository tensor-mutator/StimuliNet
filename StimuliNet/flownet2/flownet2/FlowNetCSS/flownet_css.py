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

      def __init__(self, image_src: tf.Tensor, image_dest: tf.Tensor, img_res: Tuple[int, int],
                   l2: float, flow: tf.Tensor = None, batch_norm: bool = True, trainable: bool = True) -> None:
          self._image_src = image_src
          self._image_dest = image_dest
          self._img_res = img_res
          self._flow = flow
          self._l2 = l2
          self._batch_norm = batch_norm
          self._trainable = trainable
          self._scope = 'FlowNetCSS'
          self._build_graph_with_scope()

      def _build_graph_with_scope(self) -> None:
          with tf.variable_scope(self._scope):
               self._build_graph()
               if self._trainable and self._flow is not None:
                  self.loss = type('loss', (object,), dict(input=self._flow, output=self._loss))

      def _build_graph(self) -> None:
          Mutator.scope(self._scope)
          flownet_cs = FlowNetCS(self._image_src, self._image_dest, self._img_res, self._l2, self._flow, self._batch_norm, trainable=False)
          flownet_s_input_tensor = self._compute_input_tensor_for_flownet_s(self._image_src, self._image_dest, flownet_cs.outputs[0])
          Mutator.reset_scope(self._scope)
          flownet_s = FlowNetS(flownet_s_input_tensor, self._img_res, self._l2, self._flow, self._batch_norm, trainable=self._trainable)
          self._flownet_css_patch = flownet_s.outputs
          if self._trainable and self._flow is not None:
             self._loss = flownet_s.loss.output

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
