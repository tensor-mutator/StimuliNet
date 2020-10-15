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

      def __init__(self, image: Tuple[int, int], flow: Tuple[int, int], l2: float,
                   batch_norm: bool = True, trainable:  bool = True) -> None:
          self._image = image
          self._flow = flow
          self._l2 = l2
          self._batch_norm = batch_norm
          self._trainable = trainable
          self._scope = 'FlowNet2.0'
          self._build_graph_with_scope()

      def _build_graph_with_scope(self) -> None:
          with tf.variable_scope(self._scope):
               self._build_graph()
               if self._trainable:
                  loss_input_output = self._build_loss_ops(self._flow)
                  self.loss = type('loss', (object,), loss_input_output)

      def _build_graph(self) -> None:
          Mutator.trainable = self._trainable
          Mutator.scope(self._scope)
          #self._image_1 = tf.placeholder(shape=(None,) + self._image + (3,), dtype=tf.float32, name='image_1_2')
          #self._image_2 = tf.placeholder(shape=(None,) + self._image + (3,), dtype=tf.float32, name='image_2_2')
          flownet_css = FlowNetCSS(self._image, self._flow, self._l2, self._batch_norm, trainable=False)
          Mutator.reset_scope(self._scope)
          flownet_sd = FlowNetSD(self._image, self._flow, self._l2, self._batch_norm, trainable=False)
          flownet_css_patch = flownet_css.outputs
          flownet_sd_patch = flownet_sd.outputs
          flownet_css.inputs[0], flownet_css.inputs[1] = flownet_sd.inputs[0], flownet_sd.inputs[1]
          self._image_1, self._image_2 = flownet_css.inputs[0], flownet_css.inputs[1]
          #flownet_css_patch = tf.import_graph_def(flownet_css.graph_def,
          #                                        input_map={x.name: [self._image_1, self._image_2][i] for i, x in enumerate(flownet_css.inputs)},
          #                                        return_elements=list(map(lambda x: x.name, flownet_css.outputs)), name="FlowNetCSS-Graph")
          #flownet_sd_patch = tf.import_graph_def(flownet_sd.graph_def,
          #                                       input_map={x.name: [self._image_1, self._image_2][i] for i, x in enumerate(flownet_sd.inputs)},
          #                                       return_elements=list(map(lambda x: x.name, flownet_sd.outputs)), name="FlowNetSD-Graph")
          flownet_fusion_input_tensor = self._compute_input_tensor_for_flownet_fusion(self._image_1, self._image_2,
                                                                                      flownet_css_patch[0], flownet_sd_patch[0])
          Mutator.reset_scope(self._scope)
          self._flownet_fusion = FlowNetFusion(self._image, self._l2, self._batch_norm, trainable=self._trainable)
          self._flownet_fusion.inputs[0] = flownet_fusion_input_tensor
          self._flownet_2_patch = self._flownet_fusion.outputs
          #self._flownet_2_patch = tf.import_graph_def(self._flownet_fusion.graph_def,
          #                                            input_map={self._flownet_fusion.inputs[0].name: flownet_fusion_input_tensor},
          #                                            return_elements=list(map(lambda x: x.name, self._flownet_fusion.outputs)),
          #                                            name="FlowNetFusion-Graph")

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
          return [self._image_1, self._image_2]

      @property
      def outputs(self) -> Sequence[tf.Tensor]:
          return [self._flownet_2_patch[0]]

      def get_graph(self, dest: str = os.getcwd()) -> None:
          writer = tf.summary.FileWriter(dest, graph=tf.get_default_graph())
          writer.close()

      def _build_loss_ops(self, flow) -> tf.Tensor:
          flow = tf.placeholder(dtype=tf.float32, shape=(None,) + flow + (2,))
          flow0 = self._flownet_2_patch[0]
          flow0_labels = tf.image.resize(flow, [flow0.shape[1], flow0.shape[2]])
          loss = Mutator.average_endpoint_error(flow0_labels, flow0)
          tf.losses.add_loss(loss)
          return dict(input=flow, output=tf.losses.get_total_loss(add_regularization_losses=True))
