"""
@author: Debajyoti Raychaudhuri

A TensorFlow implementation of FlowNetC
"""

from __future__ import print_function, division, absolute_import
import tensorflow.compat.v1 as tf
import tensorflow.compat.v1.keras.layers as layers
from typing import Tuple, Callable
from ..correlation_package.correlation import Correlation
from ..mutator import Mutator

class FlowNetC(object):

      def __init__(self, image_size: Tuple[int, int, int], batch_norm: bool = True, div_flow: int = 20) -> None:
          self._image_1 = self._image_2 = tf.placeholder(shape=(None,) + image_size, dtype=tf.float32)
          self._batch_norm = batch_norm
          self._div_flow = div_flow

      def _fusion_stream(self, name: str) -> Callable:
          def create_fusion_stream(input: tf.Tensor) -> tf.Tensor
              conv1 = Mutator.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), batch_norm=self._batch_norm, name=f'conv1{name}')(input)
              conv2 = Mutator.Conv2D(filters=128, kernel_size=(5, 5), strides=(2, 2), batch_norm=self._batch_norm, name=f'conv2{name}')(conv1)
              conv3 = Mutator.Conv2D(filters=256, kernel_size=(5, 5), strides=(2, 2), batch_norm=self._batch_norm, name=f'conv3{name}')(conv2)
              return conv3
          return create_fusion_stream

      def _build_model(self) -> tf.Tensor:
          stream1_tensor_out = self._fusion_stream()(self._image_1, 'a')
          stream2_tensor_out = self._fusion_stream()(self._image_2, 'b')
          corr_out = Correlation(pad_size=20, kernel_size=1, max_displacement=20, stride1=1, stride2=2, corr_multiply=1)(stream1_tensor_out, stream2_tensor_out)
          corr_out = layers.Activation(lambda x: tf.nn.leaky_relu(x, alpha=0.1))(corr_out)
          conv_redir_out = Mutator.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), batch_norm=self._batch_norm)(stream1_tensor_out)
          fused = tf.concat([conv_redir_out, corr_out], axis=1)
          conv3_1 = Mutator.Conv2D(filters=256, kernel_size=(3, 3), batch_norm=self._batch_norm, name='conv3_1')(fused)
          conv4 = Mutator.Conv2D(filters=512, kernel_size=(3, 3), strides=(2, 2), batch_norm=self._batch_norm, name='conv4')(conv3_1)
          conv4_1 = Mutator.Conv2D(filters=512, kernel_size=(3, 3), batch_norm=self._batch_norm, name='conv4_1')(conv4)
          conv5 = Mutator.Conv2D(filters=512, kernel_size=(3, 3), strides=(2, 2), batch_norm=self._batch_norm, name='conv5')(conv4_1)
          conv5_1 = Mutator.Conv2D(filters=512, kernel_size=(3, 3), batch_norm=self._batch_norm, name='conv5_1')(conv5)
          conv6 = Mutator.Conv2D(filters=1024, kernel_size=(3, 3), strides=(2, 2), batch_norm=self._batch_norm, name='conv6')(conv5_1)
          conv6_1 = Mutator.Conv2D(filters=1024, kernel_size=(3, 3), batch_norm=self._batch_norm, name='conv6_1')(conv6)
          flow6 = Mutator.PredictFlow(name='flow6')(conv6_1)
          flow6_up = Mutator.Conv2DTranspose(filters=2, kernel_size=(4, 4), strides=(2, 2), padding=1, name='flow6_up')(flow6)
          deconv5 = Mutator.Deconv(filters=512, name='deconv5')(flow6_up)
          fuse5 = tf.concat([conv5_1, deconv5, flow6_up], axis=1)
