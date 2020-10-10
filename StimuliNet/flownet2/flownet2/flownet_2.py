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
from .flow_warp import FlowWarp
from .downsample import Downsample
from typing import Tuple, Sequence
import tensorflow.compat.v1 as tf
from tensorflow_addons.image import dense_image_warp
import os

class FlowNet2(Network):

      def __init__(self, image: Tuple[int, int, int], batch_norm: bool = True) -> None:
          self._image = image
          self._batch_norm = batch_norm
          self._scope = 'FlowNet2.0'

      def _build_graph_with_scope(self) -> tf.Graph:
          self.graph = tf.Graph()
          with self.graph.as_default():
               with tf.variable_scope(self._scope):
                    self._build_graph()
                    loss_input_output = self._build_loss_ops()
                    self.loss = type('loss', (object,), loss_input_output)
          return self.graph

      @property
      def graph_def(self) -> tf.GraphDef:
          return self.graph.as_graph_def()

      def _build_graph(self) -> None:
          Mutator.set_graph(self.graph)
          self._image_1 = tf.placeholder(shape=(None,) + image, dtype=tf.float32, name='image_1_2')
          self._image_2 = tf.placeholder(shape=(None,) + image, dtype=tf.float32, name='image_2_2')
          flownet_css = FlowNetCSS(self._image, self._batch_norm, trainable=False)
          flownet_sd = FlowNetSD(self._image, self._batch_norm, trainable=False)
          flownet_css_patch = tf.import_graph_def(flownet_css.graph_def,
                                                  input_map={x.name: [self._image_1, self._image_2][i] for i, x in enumerate(flownet_css.inputs)},
                                                  return_elements=list(map(lambda x: x.name, flownet_css.outputs)))
          flownet_sd_patch = tf.import_graph_def(flownet_sd.graph_def,
                                                 input_map={x.name: [self._image_1, self._image_2][i] for i, x in enumerate(flownet_sd.inputs)},
                                                 return_elements=list(map(lambda x: x.name, flownet_sd.outputs)))
          flownet_fusion_input_tensor = self._compute_input_tensor_for_flownet_fusion(self._image_1, self._image_2, flownet_css_patch, flownet_sd_patch)
          self._flownet_fusion = FlowNetFusion(flownet_fusion_input_tensor.get_shape(), self._batch_norm)
          self._flownet_2_patch = tf.import_graph_def(self._flownet_fusion.graph_def,
                              input_map={self._flownet_fusion.inputs[0].name: flownet_fusion_input_tensor},
                              return_elements=list(map(lambda x: x.name, self._flownet_fusion.outputs)))


      def _compute_input_tensor_for_flownet_fusion(self, image_1: tf.Tensor, image_2: tf.Tensor, 
                                                   flow_out_css: tf.Tensor, flow_out_sd: tf.Tensor) -> tf.Tensor:
          flow_out_css_norm = Mutator.ChannelNorm()(flow_out_css)
          flow_out_sd_norm = Mutator.ChannelNorm()(flow_out_sd)
          warped_sd = dense_image_warp(image_2, flow_out_sd)
          brightness_error_sd = Mutator.ChannelNorm()(image_1 - warped_sd)
          warped_css = dense_image_warp(image_2, flow_out_css)
          brightness_error_css = Mutator.ChannelNorm()(image_1 - warped_css)
          return tf.concat([image_1, flow_out_sd, flow_out_css, flow_out_sd_norm, flow_out_css_norm, brightness_error_sd,
                            brightness_error_css], axis=-1)

      @property
      def inputs(self) -> Sequence[tf.Tensor]:
          return [self._image_1, self._image_2]

      @property
      def outputs(self) -> Sequence[tf.Tensor]:
          return [self._flownet_2_patch]

      def get_graph(self, dest: str = os.getcwd()) -> None:
          writer = tf.summary.FileWriter(dest, graph=self.graph)
          writer.close()

      def _build_loss_ops(self) -> tf.Tensor:
          flow = tf.placeholder(dtype=tf.float32, shape=(None,) + self._image)
          flow0 = Mutator.get_operation(self._flownet_fusion._names.get('flow0'))
          flow0_labels = Downsample(flow, [flow0.shape[1], flow0.shape[2]])
          loss = Mutator.average_endpoint_error(flow0_labels, flow0)
          tf.losses.add_loss(loss)
          return dict(input=flow, output=tf.losses.get_total_loss())
