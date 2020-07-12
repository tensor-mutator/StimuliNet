"""
@author: Debajyoti Raychaudhuri

A Mutator module for customized operations
"""

typing import Callable, Tuple
import tensorflow.compat.v1 as tf
import tensorflow.compat.v1.keras.layers as layers
from inspect import stack

class Mutator(object):
	
      graph = tf.get_default_graph()

      @staticmethod
      def BatchNorm() -> Callable:
          def add_batch_norm(input_tensor: tf.Tensor) -> tf.Tensor:
              if self._batch_norm:
                 return layers.BatchNormalization(input_tensor)
              return input_tensor
          return add_batch_norm

      @staticmethod
      def _set_name_to_instance(name: str, op_name: str) -> None:
	  stack_frames = stack()
	  frame_idx = 0
	  frame_obj = stack_frames[frame_idx][0]
	  while not frame_obj.f_locals.get('self', None):
	        frame_idx += 1
		frame_obj = stack_frames[frame_idx][0]
	  inst = frame_obj.f_locals.get('self')
	  if not getattr(inst, '_names', None):
	     setattr(inst, '_names', dict())
	  names = getattr(inst, '_names')
          scope = None
	  if getattr(inst, '_scope', None):
	     scope = getattr(inst, '_scope')
	  names[name] = f'{scope}/{op_name}' if scope else op_name
	  setattr(inst, '_names', names)
      
      @staticmethod
      def Conv2D(filters: int, kernel_size: Tuple[int, int], strides: Tuple[int, int], batch_norm: bool = True, name: str = None) -> Callable:
	  if name:
	     Mutator._set_name_to_instance(name, f'{name}/LeakyRelu')
          def conv2d(input_tensor: tf.Tensor) -> tf.Tensor:
              tensor_out = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides)(input_tensor)
              if batch_norm:
                 tensor_out = layers.BatchNormalization()(tensor_out)
              return layers.Activation(lambda x: tf.nn.leaky_relu(x, alpha=0.1), name=name)(tensor_out)
          return conv2d

      @staticmethod
      def Conv2DTranspose(filters: int, kernel_size: Tuple[int, int], strides: Tuple[int, int], padding: int = 1, name: str = None) -> Callable:
	  if name:
	     Mutator._set_name_to_instance(name, f'{name}/BiasAdd')
	  def conv_2d_transpose(input_tensor: tf.Tensor) -> tf.Tensor:
	      tensor_out = layers.ZeroPadding2D(padding)(input_tensor)
	      return layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, name=name)(tensor_out)
	  return conv_2d_transpose

      @staticmethod
      def Deconv(filters: int, name: str = None) -> Callable:
          if name:
	     Mutator._set_name_to_instance(name, f'{name}/LeakyRelu')
          def deconv(input_tensor: tf.Tensor) -> tf.Tensor:
	      tensor_out = layers.ZeroPadding2D(1)(input_tensor)
	      tensor_out = layers.Conv2DTranspose(filters=filters, kernel_size=(4, 4), strides=(2, 2))(tensor_out)
	      return layers.Activation(lambda x: tf.nn.leaky_relu(x, alpha=0.1))(tensor_out)
          return deconv

      @staticmethod
      def PredictFlow(name: str = None) -> Callable:
	  if name:
	     Mutator._set_name_to_instance(name, f'{name}/BiasAdd')
          def predict_flow(input_tensor: tf.Tensor) -> tf.Tensor:
	      tensor_out = layers.ZeroPadding2D(1)(input_tensor)
	      return layers.Conv2D(filters=2, kernel_size=(3, 3), name=name)(tensor_out)
	  return predict_flow
	
      @staticmethod
      def get_operation(name: str, scope: str = None) -> tf.Tensor:
          if scope:
	     return Mutator.graph.get_operation_by_name(f'{scope}/{name}').outputs[0]
          return Mutator.graph.get_operation_by_name(f'{name}').outputs[0]
