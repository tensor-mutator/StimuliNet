"""
@author: Debajyoti Raychaudhuri

A TensorFlow implementation of FlowNetSimple
"""

from __future__ import print_function, division, absolute_import
import tensorflow.compat.v1 as tf
import tensorflow.compat.v1.keras.layers as layers
from typing import Tuple, Sequence, Dict
import os
from ..mutator import Mutator
from ..network import Network
from ..exceptions import *

class FlowNetS(Network):

      def __init__(self, flownet_c_patch: tf.Tensor, img_res: Tuple[int, int], l2: float, flow: tf.Tensor = None,
                   batch_norm: bool = True, trainable: bool = True) -> None: 
          self._batch_norm = batch_norm
          self._flownet_c_patch = flownet_c_patch
          self._img_res = img_res
          self._flow = flow
          self._l2 = l2
          self._trainable = trainable
          self.flow_scale = 0.05
          self._scope = 'FlowNetS'
          self._build_graph_with_scope()

      def _build_graph_with_scope(self) -> None:
          with tf.variable_scope(self._scope):
               self._build_graph()
               if self._trainable and self._flow is not None:
                  loss_input_output = self._build_loss_ops(self._flow)
                  self.loss = type('loss', (object,), loss_input_output)

      def _build_graph(self) -> None:
          Mutator.trainable = self._trainable
          Mutator.scope(self._scope)
          self._downsampling()
          self._upsampling()

      def _downsampling(self) -> None:
          conv1 = Mutator.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), batch_norm=self._batch_norm,
                                 name='conv1')(Mutator.pad(self._flownet_c_patch, 3, channels=12, resolution=self._img_res))
          conv2 = Mutator.layers.Conv2D(filters=128, kernel_size=(5, 5), strides=(2, 2), batch_norm=self._batch_norm,
                                 name='conv2')(Mutator.pad(conv1, 2))
          conv3 = Mutator.layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(2, 2), batch_norm=self._batch_norm,
                                 name='conv3')(Mutator.pad(conv2, 2))
          conv3_1 = Mutator.layers.Conv2D(filters=256, kernel_size=(3, 3), batch_norm=self._batch_norm, name='conv3_1')(Mutator.pad(conv3))
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
          fuse2 = tf.concat([Mutator.get_operation(self._names.get('conv2')), deconv2, flow3_up], axis=3, name='fuse2')
          flow2 = Mutator.layers.Conv2DFlow(name='flow2', scale=20.0, resize=self._img_res,
                                            kernel_regularizer=tf.keras.regularizers.l2(self._l2))(fuse2)

      @property
      def inputs(self) -> Sequence[tf.Tensor]:
          return [self._flownet_c_patch]

      @property
      def outputs(self) -> Sequence[tf.Tensor]:
          return [Mutator.get_operation(self._names.get('flow2'))]
      
      def _build_loss_ops(self, flow_in: tf.Tensor) -> Dict:
          flow = flow_in * self.flow_scale
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
          return dict(input=flow_in, output=tf.losses.get_total_loss(add_regularization_losses=True))

      def model(self, *args, **kwargs) -> None:
          raise NotTrainableError(f"Model: {self.__class__.__name__} cannot be trained as a separate block")
