from __future__ import print_function, division, absolute_import
import tensorflow.compat.v1 as tf
import tensorflow.compat.v1.keras.layers as layers
from typing import Tuple, Callable
from .correlation_package.correlation import Correlation
from .mutator import Mutator

class FlowNetC(object):

      def __init__(self, image_size: Tuple[int, int, int], batch_norm: bool = True, div_flow: int = 20) -> None:
	  self._image_1 = self._image_2 = tf.placeholder(shape=(None,) + image_size, dtype=tf.float32)
          self._batch_norm = batch_norm
          self._div_flow = div_flow

      def _fusion_stream(self) -> Callable:
	  def create_fusion_stream(input: tf.Tensor) -> tf.Tensor
	      conv1 = Mutator.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), batch_norm=self._batch_norm)(input)
	      conv2 = Mutator.Conv2D(filters=128, kernel_size=(5, 5), strides=(2, 2), batch_norm=self._batch_norm)(conv1)
	      conv3 = Mutator.Conv2D(filters=256, kernel_size=(5, 5), strides=(2, 2), batch_norm=self._batch_norm)(conv2)
	      return conv3
          return create_fusion_stream

      def _build_model(self) -> tf.Tensor:
	  stream1_tensor_out = self._fusion_stream()(self._image_1)
	  stream2_tensor_out = self._fusion_stream()(self._image_2)
          corr_out = Correlation(pad_size=20, kernel_size=1, max_displacement=20, stride1=1, stride2=2, corr_multiply=1)(stream1_tensor_out, stream2_tensor_out)
	  corr_out = layers.Activation(lambda x: tf.nn.leaky_relu(x, alpha=0.1))(corr_out)
          conv_redir_out = Mutator.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), batch_norm=self._batch_norm)(stream1_tensor_out)
	  merged = tf.concat([conv_redir_out, corr_out], axis=1)
          conv3_1 = Mutator.Conv2D(filters=256, kernel_size=(3, 3), batch_norm=self._batch_norm)(merged)
          conv3_2 = Mutator.Conv2D(filters=512, kernel_size=(3, 3), strides=(2, 2), batch_norm=self._batch_norm)(conv3_1)
          conv3_2_1 = Mutator.Conv2D(filters=512, kernel_size=(3, 3), batch_norm=self._batch_norm)(conv3_2)
          conv3_3 = Mutator.Conv2D(filters=512, kernel_size=(3, 3), strides=(2, 2), batch_norm=self._batch_norm)(conv3_2_1)
          conv3_3_1 = Mutator.Conv2D(filters=512, kernel_size=(3, 3), batch_norm=self._batch_norm)(conv3_3)
          conv3_4 = Mutator.Conv2D(filters=1024, kernel_size=(3, 3), strides=(2, 2), batch_norm=self._batch_norm)(conv3_3_1)
          conv3_4_1 = Mutator.Conv2D(filters=1024, kernel_size=(3, 3), batch_norm=self._batch_norm)(conv3_4)
	  flow1 = Mutator.PredictFlow()(conv3_4_1)
	  layers.ConvTranspose2D()
