"""
@author: Debajyoti Raychaudhuri

A TensorFlow implementation of FlowNet2.0
"""

from __future__ import print_function, division, absolute_import
from .network import Network
from .FlowNetCSS import FlowNetCSS
from .FlowNetSD import FlowNetSD
from .FlowNetFusion import FlowNetFusion
from .mutator import Mutator
from typing import Tuple, Sequence
import tensorflow.compat.v1 as tf
from tensorflow_addons.image import dense_image_warp
import os

class FlowNet2(Network):

      def __init__(self, image_src: tf.Tensor, image_dest: tf.Tensor, img_res: Tuple[int, int],
                   l2: float, flow: tf.Tensor = None, batch_norm: bool = True, trainable: bool = True) -> None:
          self._image_src = image_src
          self._image_dest = image_dest
          self._img_res = img_res
          self._flow = flow
          self._l2 = l2
          self._batch_norm = batch_norm
          self._trainable = trainable
          self._scope = "FlowNet2"
          self._build_graph_with_scope()

      def _build_graph_with_scope(self) -> None:
          with tf.variable_scope(self._scope):
               self._build_graph()
               if self._trainable and self._flow is not None:
                  loss_input_output = self._build_loss_ops(self._flow)
                  self.loss = type('loss', (object,), loss_input_output)

      def _build_graph(self) -> None:
          Mutator.scope(self._scope)
          flownet_css = FlowNetCSS(self._image_src, self._image_dest, self._img_res, self._l2, self._flow, self._batch_norm, trainable=False)
          Mutator.reset_scope(self._scope)
          flownet_sd = FlowNetSD(self._image_src, self._image_dest, self._img_res, self._l2, self._flow, self._batch_norm, trainable=False)
          flownet_fusion_input_tensor = self._compute_input_tensor_for_flownet_fusion(self._image_src, self._image_dest,
                                                                                      flownet_css.outputs[0], flownet_sd.outputs[0])
          Mutator.reset_scope(self._scope)
          self._flownet_fusion = FlowNetFusion(flownet_fusion_input_tensor, self._img_res, self._l2, self._flow, self._batch_norm,
                                               trainable=self._trainable)
          self._flownet_2_patch = self._flownet_fusion.outputs

      def _compute_input_tensor_for_flownet_fusion(self, image_1: tf.Tensor, image_2: tf.Tensor, 
                                                   flow_out_css: tf.Tensor, flow_out_sd: tf.Tensor) -> tf.Tensor:
          flow_out_css_norm = Mutator.layers.ChannelNorm()(flow_out_css)
          flow_out_sd_norm = Mutator.layers.ChannelNorm()(flow_out_sd)
          warped_sd = dense_image_warp(image_2, flow_out_sd)
          brightness_error_sd = Mutator.layers.ChannelNorm()(image_1 - warped_sd)
          warped_css = dense_image_warp(image_2, flow_out_css)
          brightness_error_css = Mutator.layers.ChannelNorm()(image_1 - warped_css)
          return tf.concat([image_1, flow_out_sd, flow_out_css, flow_out_sd_norm, flow_out_css_norm, brightness_error_sd,
                            brightness_error_css], axis=-1)

      @property
      def inputs(self) -> Sequence[tf.Tensor]:
          return [self._image_src, self._image_dest]

      @property
      def outputs(self) -> Sequence[tf.Tensor]:
          return [self._flownet_2_patch[0]]

      def _build_loss_ops(self, flow_in: tf.Tensor) -> tf.Tensor:
          flow0 = self._flownet_2_patch[0]
          flow0_labels = tf.image.resize(flow_in, [flow0.shape[1], flow0.shape[2]])
          loss = Mutator.average_endpoint_error(flow0_labels, flow0)
          tf.losses.add_loss(loss)
          return dict(input=flow_in, output=tf.losses.get_total_loss(add_regularization_losses=True))
