"""
@author: Debajyoti Raychaudhuri

A Mutator module for customized operations
"""

from __future__ import print_function, division, absolute_import
from typing import Callable, Tuple, Any
import tensorflow.compat.v1 as tf
import tensorflow.compat.v1.keras.layers as layers
from inspect import stack

class Mutator(object):

      trainable: bool = True
      graph = tf.get_default_graph()
      _scope = ""

      @classmethod
      def scope(cls, scope: str) -> None:
          cls._scope += "{}/".format(scope)

      @classmethod
      def reset_scope(cls, scope: str) -> None:
          parent_scope = cls._scope.split("{}/".format(scope))[0]
          cls._scope = "{}{}/".format(parent_scope, scope)

      @classmethod
      def set_graph(cls, graph: tf.Graph) -> None:
          cls.graph = graph

      @staticmethod
      def pad(tensor: tf.Tensor, padding: int = 1) -> tf.Tensor:
          return tf.pad(tensor, [[0, 0], [padding, padding], [padding, padding], [0, 0]])

      @staticmethod
      def antipad(tensor: tf.Tensor, channels: int, padding: int = 1, name: str = None) -> tf.Tensor:
          n, h, w = tf.shape(tensor)[0], tf.shape(tensor)[1], tf.shape(tensor)[2]
          return tf.slice(tensor, begin=[0, padding, padding, 0], size=[n, h - 2 * padding, w - 2 * padding, channels], name=name)

      @staticmethod
      def get_operation(name: str, scope: str = None) -> tf.Tensor:
          if scope:
             return Mutator.graph.get_operation_by_name(f'{scope}/{name}').outputs[0]
          return Mutator.graph.get_operation_by_name(f'{name}').outputs[0]

      @staticmethod
      def average_endpoint_error(labels: tf.Tensor, predictions: tf.Tensor) -> tf.Tensor:
          endpoint_error = tf.sqrt(tf.reduce_sum(tf.square(predictions - labels), axis=-1, keep_dims=True))
          return tf.reduce_mean(tf.reduce_sum(endpoint_error, axis=[1,2,3]))

      @classmethod
      def _set_name_to_instance(cls, name: str, op_name: str) -> None:
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
          names[name] = f'{cls._scope}{op_name}' if cls._scope != "" else op_name
          setattr(inst, '_names', names)

      class layers:

            @staticmethod
            def BatchNorm(axis: int = 0) -> Callable:
                def _op(input_tensor: tf.Tensor) -> tf.Tensor:
                    if self._batch_norm:
                       return layers.BatchNormalization(axis=axis, trainable=Mutator.trainable)(input_tensor)
                    return input_tensor
                return _op

            @staticmethod
            def ChannelNorm(axis: int = -1) -> Callable:
                def _op(input_tensor: tf.Tensor) -> tf.Tensor:
                    return tf.sqrt(tf.reduce_sum(tf.square(input_tensor), axis=axis, keep_dims=True))
                return _op

            @staticmethod
            def Conv2D(filters: int, kernel_size: Tuple[int, int], strides: Tuple[int, int] = (1, 1),
                       batch_norm: bool = True, activation: bool = True, name: str = None,
                       kernel_regularizer: tf.keras.regularizers.l2 = None) -> Callable:
                if name:
                   if activation:
                      Mutator._set_name_to_instance(name, f'{name}/LeakyRelu')
                   else:
                      if batch_norm:
                         Mutator._set_name_to_instance(name, f'{name}/cond/Identity')
                      else:
                         Mutator._set_name_to_instance(name, f'{name}/BiasAdd')
                def _op(input_tensor: tf.Tensor) -> tf.Tensor:
                    _name = name if not batch_norm and not activation else None
                    tensor_out = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, name=_name,
                                               trainable=Mutator.trainable, kernel_regularizer=kernel_regularizer)(input_tensor)
                    if batch_norm:
                       _name = name if not activation else None
                       tensor_out = layers.BatchNormalization(trainable=Mutator.trainable, name=_name)(tensor_out)
                    if activation:
                       return layers.Activation(lambda x: tf.nn.leaky_relu(x, alpha=0.1), trainable=Mutator.trainable, name=name)(tensor_out)
                    return tensor_out
                return _op

            @staticmethod
            def Conv2DTranspose(filters: int, kernel_size: Tuple[int, int], strides: Tuple[int, int] = (1, 1),
                                activation: bool = True, name: str = None) -> Callable:
                if name:
                   if activation:
                      Mutator._set_name_to_instance(name, f'{name}/LeakyRelu')
                   else:
                      Mutator._set_name_to_instance(name, f'{name}/BiasAdd')
                def _op(input_tensor: tf.Tensor) -> tf.Tensor:
                    _name = name if not activation else None
                    tensor_out = layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, name=_name,
                                                        trainable=Mutator.trainable)(input_tensor)
                    if activation:
                       return layers.Activation(lambda x: tf.nn.leaky_relu(x, alpha=0.1), trainable=Mutator.trainable, name=name)(tensor_out)
                    return tensor_out
                return _op

            @staticmethod
            def Conv2DFlow(name: str = None, scale: float = None, resize: Tuple[int, int] = None,
                           kernel_regularizer: tf.keras.regularizers.l2 = None) -> Callable:
                if name and (scale or resize):
                   Mutator._set_name_to_instance(name, name)
                def _op(input_tensor: tf.Tensor) -> tf.Tensor:
                    if scale or resize:
                       tensor_out = Mutator.layers.Conv2D(2, (3, 3), batch_norm=False, activation=False,
                                                          kernel_regularizer=kernel_regularizer)(Mutator.pad(input_tensor))
                       if scale:
                          _name = name if not resize else None
                          tensor_out = tf.multiply(tensor_out, scale, name=_name)
                       if resize:
                          tensor_out = tf.image.resize_bilinear(tensor_out, resize, align_corners=True, name=name)
                       return tensor_out
                    return Mutator.layers.Conv2D(2, (3, 3), batch_norm=False, activation=False, name=name,
                                                 kernel_regularizer=kernel_regularizer)(Mutator.pad(input_tensor))
                return _op

            @staticmethod
            def Deconv(filters: int, name: str = None) -> Callable:
                if name:
                   Mutator._set_name_to_instance(name, name)
                def _op(input_tensor: tf.Tensor) -> tf.Tensor:
                    tensor_out = Mutator.layers.Conv2DTranspose(filters, (4, 4), (2, 2))(input_tensor)
                    return Mutator.antipad(tensor_out, filters, name=name)
                return _op

            @staticmethod
            def Upconv(name: str = None) -> Callable:
                if name:
                   Mutator._set_name_to_instance(name, name)
                def _op(input_tensor: tf.Tensor) -> tf.Tensor:
                    tensor_out = Mutator.layers.Conv2DTranspose(2, (4, 4), (2, 2), activation=False)(input_tensor)
                    return Mutator.antipad(tensor_out, 2, name=name)
                return _op

            @staticmethod
            def Conv2DInter(filters: int, name: str = None) -> Callable:
                def _op(input_tensor: tf.Tensor) -> tf.Tensor:
                    return Mutator.layers.Conv2D(filters, (3, 3), batch_norm=False, activation=False, name=name)(Mutator.pad(input_tensor))
                return _op
