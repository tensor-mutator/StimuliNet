"""
@author: Debajyoti Raychaudhuri

A TensorFlow implementation of FlowNetC
"""

typing import Callable, Tuple
import tensorflow.compat.v1 as tf
import tensorflow.compat.v1.keras.layers as layers

class Mutator(object):

      @staticmethod
      def BatchNorm() -> Callable:
	        def add_batch_norm(input_tensor: tf.Tensor) -> tf.Tensor:
              if self._batch_norm:
                 return layers.BatchNormalization(input_tensor)
              return input_tensor
          return add_batch_norm
      
      @staticmethod
      def Conv2D(filters: int, kernel_size: Tuple[int, int], strides: Tuple[int, int], batch_norm=True) -> Callable:
          def conv2d(input_tensor: tf.Tensor) -> tf.Tensor:
              tensor_out = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides)(input_tensor)
              if batch_norm:
                 tensor_out = layers.BatchNormalization()(tensor_out)
              return layers.Activation(lambda x: tf.nn.leaky_relu(x, alpha=0.1))(tensor_out)
          return conv2d

      @staticmethod
      def PredictFlow() -> Callable:
          def predict_flow(input_tensor: tf.Tensor) -> tf.Tensor:
	      tensor_out = layers.ZeroPadding1D()(input_tensor)
	      return layers.Conv2D(filters=2, kernel_size=(3, 3))(tensor_out)
	  return predict_flow
