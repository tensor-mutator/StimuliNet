"""
@author: Debajyoti Raychaudhuri

A TensorFlow implementation of FlowNetCorr
"""

from __future__ import print_function, division, absolute_import
import tensorflow.compat.v1 as tf
import tensorflow.compat.v1.keras.layers as layers
from typing import Tuple, Callable
from FlowNet2.correlation_package.correlation import Correlation
from FlowNet2.mutator import Mutator
from FlowNet.network import Network
import os

class FlowNetC(Network):

      def __init__(self, image: Tuple[int, int, int], batch_norm: bool = True, div_flow: int = 20) -> None:
          self._image_1 = tf.placeholder(shape=(None,) + image, dtype=tf.float32, name='image_1_c')
          self._image_2 = tf.placeholder(shape=(None,) + image, dtype=tf.float32, name='image_2_c')
          self._batch_norm = batch_norm
          self._div_flow = div_flow
          self._scope = 'FlowNetC'
          self._build_graph_with_scope()

      def _fusion_stream(self, name: str) -> Callable:
          def create_fusion_stream(input: tf.Tensor) -> tf.Tensor:
              conv1 = Mutator.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), batch_norm=self._batch_norm, name=f'conv1{name}')(input)
              conv2 = Mutator.Conv2D(filters=128, kernel_size=(5, 5), strides=(2, 2), batch_norm=self._batch_norm, name=f'conv2{name}')(conv1)
              conv3 = Mutator.Conv2D(filters=256, kernel_size=(5, 5), strides=(2, 2), batch_norm=self._batch_norm, name=f'conv3{name}')(conv2)
              return conv3
          return create_fusion_stream

      def _build_graph_with_scope(self) -> tf.Graph:
          self.graph = tf.Graph()
          with self.graph.as_default():
               with tf.variable_scope(self._scope):
                    self._build_graph()
          return self.graph

      def _build_graph(self) -> None:
          Mutator.set_graph(self.graph)
          self._downsampling()
          self._upsampling()

      def _downsampling(self) -> None:
          stream1_tensor_out = self._fusion_stream('a')(self._image_1)
          stream2_tensor_out = self._fusion_stream('b')(self._image_2)
          corr_out = Correlation(pad_size=20, kernel_size=1, max_displacement=20, stride1=1, stride2=2, corr_multiply=1)(stream1_tensor_out, stream2_tensor_out)
          corr_out = layers.Activation(lambda x: tf.nn.leaky_relu(x, alpha=0.1), name='correlation')(corr_out)
          conv_redir_out = Mutator.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), batch_norm=self._batch_norm, name='conv_redir')(stream1_tensor_out)
          fused = tf.concat([conv_redir_out, corr_out], axis=1, name='fuse')
          conv3_1 = Mutator.Conv2D(filters=256, kernel_size=(3, 3), batch_norm=self._batch_norm, name='conv3_1')(fused)
          conv4 = Mutator.Conv2D(filters=512, kernel_size=(3, 3), strides=(2, 2), batch_norm=self._batch_norm, name='conv4')(conv3_1)
          conv4_1 = Mutator.Conv2D(filters=512, kernel_size=(3, 3), batch_norm=self._batch_norm, name='conv4_1')(conv4)
          conv5 = Mutator.Conv2D(filters=512, kernel_size=(3, 3), strides=(2, 2), batch_norm=self._batch_norm, name='conv5')(conv4_1)
          conv5_1 = Mutator.Conv2D(filters=512, kernel_size=(3, 3), batch_norm=self._batch_norm, name='conv5_1')(conv5)
          conv6 = Mutator.Conv2D(filters=1024, kernel_size=(3, 3), strides=(2, 2), batch_norm=self._batch_norm, name='conv6')(conv5_1)
          conv6_1 = Mutator.Conv2D(filters=1024, kernel_size=(3, 3), batch_norm=self._batch_norm, name='conv6_1')(conv6)

      def _upsampling(self) -> None:
          flow6 = Mutator.PredictFlow(name='flow6')(Mutator.get_operation(self._names.get('conv6_1')))
          flow6_up = Mutator.Conv2DTranspose(filters=2, kernel_size=(4, 4), strides=(2, 2), padding=1, name='flow6_up')(flow6)
          deconv5 = Mutator.Deconv(filters=512, name='deconv5')(Mutator.get_operation(self._names.get('conv6_1')))
          fuse5 = tf.concat([Mutator.get_operation(self._names.get('conv5_1')), deconv5, flow6_up], axis=1, name='fuse5')
          flow5 = Mutator.PredictFlow(name='flow5')(fuse5)
          flow5_up = Mutator.Conv2DTranspose(filters=2, kernel_size=(4, 4), strides=(2, 2), padding=1, name='flow5_up')(flow5)
          deconv4 = Mutator.Deconv(filters=256, name='deconv4')(fuse5)
          fuse4 = tf.concat([Mutator.get_operation(self._names.get('conv4_1')), deconv4, flow5_up], axis=1, name='fuse4')
          flow4 = Mutator.PredictFlow(name='flow4')(fuse4)
          flow4_up = Mutator.Conv2DTranspose(filters=2, kernel_size=(4, 4), strides=(2, 2), padding=1, name='flow4_up')(flow4)
          deconv3 = Mutator.Deconv(filters=128, name='deconv3')(fuse4)
          fuse3 = tf.concat([Mutator.get_operation(self._names.get('conv3_1')), deconv3, flow4_up], axis=1, name='fuse3')
          flow3 = Mutator.PredictFlow(name='flow3')(fuse3)
          flow3_up = Mutator.Conv2DTranspose(filters=2, kernel_size=(4, 4), strides=(2, 2), padding=1, name='flow3_up')(flow3)
          deconv2 = Mutator.Deconv(filters=64, name='deconv2')(fuse3)
          fuse2 = tf.concat([Mutator.get_operation(self._names.get('conv2a')), deconv2, flow3_up], axis=1, name='fuse2')
          flow2 = Mutator.PredictFlow(name='flow2')(fuse2)

      def get_graph(self, dest: str = os.getcwd()) -> None:
          writer = tf.summary.FileWriter(dest, graph=self.graph)
          writer.close()
