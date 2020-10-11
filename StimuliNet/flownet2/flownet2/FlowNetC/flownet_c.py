"""
@author: Debajyoti Raychaudhuri

A TensorFlow implementation of FlowNetCorr
"""

from __future__ import print_function, division, absolute_import
import tensorflow.compat.v1 as tf
import tensorflow.compat.v1.keras.layers as layers
from tensorflow_addons.layers import CorrelationCost
from typing import Tuple, Callable, Sequence
import numpy as np
import os
from ..mutator import Mutator
from ..network import Network

class FlowNetC(Network):

      def __init__(self, image: Tuple[int, int], flow: Tuple[int, int], batch_norm: bool = True, trainable: bool = True) -> None:
          self._image = image
          self._flow = flow
          self._batch_norm = batch_norm
          self._trainable = trainable
          self.flow_scale = 0.05
          self._scope = 'FlowNetC'
          self._build_graph_with_scope()

      def _fusion_stream(self, name: str) -> Callable:
          def create_fusion_stream(input: tf.Tensor) -> tf.Tensor:
              conv1 = Mutator.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), batch_norm=self._batch_norm,
                                            name=f'conv1{name}')(Mutator.pad(input, 3))
              conv2 = Mutator.layers.Conv2D(filters=128, kernel_size=(5, 5), strides=(2, 2), batch_norm=self._batch_norm,
                                            name=f'conv2{name}')(Mutator.pad(conv1, 2))
              conv3 = Mutator.layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(2, 2), batch_norm=self._batch_norm,
                                            name=f'conv3{name}')(Mutator.pad(conv2, 2))
              return conv3
          return create_fusion_stream

      def _build_graph_with_scope(self) -> tf.Graph:
          self.graph = tf.Graph()
          with self.graph.as_default():
               with tf.variable_scope(self._scope):
                    self._build_graph()
                    if self._trainable:
                       loss_input_output = self._build_loss_ops(self._flow)
                       self.loss = type('loss', (object,), loss_input_output)
          return self.graph

      @property
      def graph_def(self) -> tf.GraphDef:
          return self.graph.as_graph_def()

      def _build_graph(self) -> None:
          Mutator.set_graph(self.graph)
          Mutator.trainable = self._trainable
          self._image_1 = tf.placeholder(shape=(None,) + self._image + (3,), dtype=tf.float32, name='image_1_c')
          self._image_2 = tf.placeholder(shape=(None,) + self._image + (3,), dtype=tf.float32, name='image_2_c')
          self._downsampling()
          self._upsampling()

      def _downsampling(self) -> None:
          stream1_tensor_out = self._fusion_stream('a')(self._image_1)
          stream2_tensor_out = self._fusion_stream('b')(self._image_2)
          corr_out = CorrelationCost(pad=20, kernel_size=1, max_displacement=20, stride_1=1, stride_2=2,
                                     data_format="channels_last")([stream1_tensor_out, stream2_tensor_out])
          corr_out = layers.Activation(lambda x: tf.nn.leaky_relu(x, alpha=0.1), trainable=self._trainable, name='correlation')(corr_out)
          conv_redir_out = Mutator.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), batch_norm=self._batch_norm,
                                                 name='conv_redir')(stream1_tensor_out)
          fused = tf.concat([conv_redir_out, corr_out], axis=3, name='fuse')
          conv3_1 = Mutator.layers.Conv2D(filters=256, kernel_size=(3, 3), batch_norm=self._batch_norm, name='conv3_1')(Mutator.pad(fused))
          conv4 = Mutator.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(2, 2), batch_norm=self._batch_norm,
                                        name='conv4')(Mutator.pad(conv3_1))
          conv4_1 = Mutator.layers.Conv2D(filters=512, kernel_size=(3, 3), batch_norm=self._batch_norm, name='conv4_1')(Mutator.pad(conv4))
          conv5 = Mutator.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(2, 2), batch_norm=self._batch_norm,
                                        name='conv5')(Mutator.pad(conv4_1))
          conv5_1 = Mutator.layers.Conv2D(filters=512, kernel_size=(3, 3), batch_norm=self._batch_norm, name='conv5_1')(Mutator.pad(conv5))
          conv6 = Mutator.layers.Conv2D(filters=1024, kernel_size=(3, 3), strides=(2, 2), batch_norm=self._batch_norm,
                                        name='conv6')(Mutator.pad(conv5_1))
          conv6_1 = Mutator.layers.Conv2D(filters=1024, kernel_size=(3, 3), batch_norm=self._batch_norm, name='conv6_1')(Mutator.pad(conv6))

      def _upsampling(self) -> None:
          flow6 = Mutator.layers.Conv2DFlow(name='flow6')(Mutator.get_operation(self._names.get('conv6_1')))
          flow6_up = Mutator.layers.Upconv(name='flow6_up')(flow6)
          deconv5 = Mutator.layers.Deconv(filters=512, name='deconv5')(Mutator.get_operation(self._names.get('conv6_1')))
          fuse5 = tf.concat([Mutator.get_operation(self._names.get('conv5_1')), deconv5, flow6_up], axis=3, name='fuse5')
          flow5 = Mutator.layers.Conv2DFlow(name='flow5')(fuse5)
          flow5_up = Mutator.layers.Upconv(name='flow5_up')(flow5)
          deconv4 = Mutator.layers.Deconv(filters=256, name='deconv4')(fuse5)
          fuse4 = tf.concat([Mutator.get_operation(self._names.get('conv4_1')), deconv4, flow5_up], axis=3, name='fuse4')
          flow4 = Mutator.layers.Conv2DFlow(name='flow4')(fuse4)
          flow4_up = Mutator.layers.Upconv(name='flow4_up')(flow4)
          deconv3 = Mutator.layers.Deconv(filters=128, name='deconv3')(fuse4)
          fuse3 = tf.concat([Mutator.get_operation(self._names.get('conv3_1')), deconv3, flow4_up], axis=3, name='fuse3')
          flow3 = Mutator.layers.Conv2DFlow(name='flow3')(fuse3)
          flow3_up = Mutator.layers.Upconv(name='flow3_up')(flow3)
          deconv2 = Mutator.layers.Deconv(filters=64, name='deconv2')(fuse3)
          fuse2 = tf.concat([Mutator.get_operation(self._names.get('conv2a')), deconv2, flow3_up], axis=3, name='fuse2')
          flow2 = Mutator.layers.Conv2DFlow(name='flow2')(fuse2)

      @property
      def inputs(self) -> Sequence[tf.Tensor]:
          return [self._image_1, self._image_2]

      @property
      def outputs(self) -> Sequence[tf.Tensor]:
          return [Mutator.get_operation(self._names.get('flow2'))]

      def get_graph(self, dest: str = os.getcwd()) -> None:
          writer = tf.summary.FileWriter(dest, graph=self.graph)
          writer.close()

      def _build_loss_ops(self, flow) -> tf.Tensor:
          flow = tf.placeholder(dtype=tf.float32, shape=(None,) + flow + (2,))
          flow = flow * self.flow_scale
          losses = list()
          flow6 = Mutator.get_operation(self._names.get('flow6'))
          flow6_labels = tf.image.resize(flow, [flow6.shape[1], flow6.shape[2]])
          losses.append(Mutator.average_endpoint_error(flow6_labels, flow6))
          flow5 = Mutator.get_operation(self._names.get('flow5'))
          flow5_labels = tf.image.resize(flow, [flow5.shape[1], flow5.shape[2]])
          losses.append(Mutator.average_endpoint_error(flow5_labels, flow5))
          flow4 = Mutator.get_operation(self._names.get('flow4'))
          flow4_labels = tf.image.resize(flow, [flow4.shape[1], flow4.shape[2]])
          losses.append(Mutator.average_endpoint_error(flow4_labels, flow4))
          flow3 = Mutator.get_operation(self._names.get('flow3'))
          flow3_labels = tf.image.resize(flow, [flow3.shape[1], flow3.shape[2]])
          losses.append(Mutator.average_endpoint_error(flow3_labels, flow3))
          flow2 = Mutator.get_operation(self._names.get('flow2'))
          flow2_labels = tf.image.resize(flow, [flow2.shape[1], flow2.shape[2]])
          losses.append(Mutator.average_endpoint_error(flow2_labels, flow2))
          loss = tf.losses.compute_weighted_loss(losses, [0.32, 0.08, 0.02, 0.01, 0.005])
          return dict(input=flow, output=tf.losses.get_total_loss())
