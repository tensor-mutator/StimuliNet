"""
@author: Debajyoti Raychaudhuri

A Mutator module for customized operations
"""

from __future__ import print_function, division, absolute_import
from typing import Callable, Tuple
import tensorflow.compat.v1 as tf
import tensorflow.compat.v1.keras.layers as layers
from inspect import stack

class Mutator(object):

      trainable: bool = True

      @classmethod
      def set_graph(cls, graph: tf.Graph) -> None:
          cls.graph = graph

      @staticmethod
      def BatchNorm() -> Callable:
          def add_batch_norm(input_tensor: tf.Tensor) -> tf.Tensor:
              if self._batch_norm:
                 return layers.BatchNormalization(trainable=Mutator.trainable)(input_tensor)
              return input_tensor
          return add_batch_norm

      @staticmethod
      def ChannelNorm() -> Callable:
          def add_channel_norm(input_tensor: tf.Tensor) -> tf.Tensor:
              return tf.sqrt(tf.reduce_sum(tf.square(input_tensor), axis=-1, keep_dims=True))
          return add_channel_norm    

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
      def Conv2D(filters: int, kernel_size: Tuple[int, int], strides: Tuple[int, int]=(1, 1), batch_norm: bool = True, name: str = None) -> Callable:
          if name:
             Mutator._set_name_to_instance(name, f'{name}/LeakyRelu')
          def conv2d(input_tensor: tf.Tensor) -> tf.Tensor:
              tensor_out = tf.pad(input_tensor, [[0, 0], [1, 1], [1, 1], [0, 0]])
              tensor_out = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, trainable=Mutator.trainable)(tensor_out)
              if batch_norm:
                 tensor_out = layers.BatchNormalization(trainable=Mutator.trainable)(tensor_out)
              return layers.Activation(lambda x: tf.nn.leaky_relu(x, alpha=0.1), trainable=Mutator.trainable, name=name)(tensor_out)
          return conv2d

      @staticmethod
      def Conv2DInter(filters: int, kernel_size: Tuple[int, int], strides: Tuple[int, int], batch_norm: bool = True, name: str = None) -> Callable:
          if name:
             if batch_norm:
                Mutator._set_name_to_instance(name, f'{name}/cond/Identity')
             else:
                Mutator._set_name_to_instance(name, f'{name}/BiasAdd')
          def conv2d(input_tensor: tf.Tensor) -> tf.Tensor:
              tensor_out = layers.ZeroPadding2D((kernel_size[0] - 1)//2, trainable=Mutator.trainable)(input_tensor)
              conv_op_name = name if not batch_norm else None
              tensor_out = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, trainable=Mutator.trainable, name=conv_op_name)(tensor_out)
              if batch_norm:
                 tensor_out = layers.BatchNormalization(trainable=Mutator.trainable, name=name)(tensor_out)
              return tensor_out
          return conv2d

      @staticmethod
      def Conv2DTranspose(filters: int, kernel_size: Tuple[int, int], strides: Tuple[int, int], padding: int = 1, name: str = None) -> Callable:
          if name:
             Mutator._set_name_to_instance(name, f'{name}/BiasAdd')
          def conv_2d_transpose(input_tensor: tf.Tensor) -> tf.Tensor:
              tensor_out = layers.ZeroPadding2D(padding, trainable=Mutator.trainable)(input_tensor)
              return layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, trainable=Mutator.trainable, name=name)(tensor_out)
          return conv_2d_transpose

      @staticmethod
      def Deconv(filters: int, name: str = None) -> Callable:
          if name:
             Mutator._set_name_to_instance(name, f'{name}/LeakyRelu')
          def deconv(input_tensor: tf.Tensor) -> tf.Tensor:
              tensor_out = layers.ZeroPadding2D(1, trainable=Mutator.trainable)(input_tensor)
              tensor_out = layers.Conv2DTranspose(filters=filters, kernel_size=(4, 4), strides=(2, 2), trainable=Mutator.trainable)(tensor_out)
              return layers.Activation(lambda x: tf.nn.leaky_relu(x, alpha=0.1), trainable=Mutator.trainable)(tensor_out)
          return deconv

      @staticmethod
      def PredictFlow(name: str = None) -> Callable:
          if name:
             Mutator._set_name_to_instance(name, f'{name}/BiasAdd')
          def predict_flow(input_tensor: tf.Tensor) -> tf.Tensor:
              tensor_out = layers.ZeroPadding2D(1, trainable=Mutator.trainable)(input_tensor)
              return layers.Conv2D(filters=2, kernel_size=(3, 3), trainable=Mutator.trainable, name=name)(tensor_out)
          return predict_flow

      @staticmethod
      def get_operation(name: str, scope: str = None) -> tf.Tensor:
          if scope:
             return Mutator.graph.get_operation_by_name(f'{scope}/{name}').outputs[0]
          return Mutator.graph.get_operation_by_name(f'{name}').outputs[0]

      @staticmethod
      def average_endpoint_error(labels: tf.Tensor, predictions: tf.Tensor) -> tf.Tensor:
          endpoint_error = tf.sqrt(tf.reduce_sum(tf.square(predictions - labels), axis=-1, keep_dims=True))
          return tf.reduce_mean(tf.reduce_sum(endpoint_error, axis=[1,2,3]))
