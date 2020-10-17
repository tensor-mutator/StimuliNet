"""
@author: Debajyoti Raychaudhuri

A TensorFlow implementation of FlowNetSD
"""

from __future__ import print_function, division, absolute_import
import tensorflow.compat.v1 as tf
import tensorflow.compat.v1.keras.layers as layers
from typing import Tuple, Sequence
import os
from ..mutator import Mutator
from ..network import Network

class FlowNetSD(Network):

      def __init__(self, image_src: tf.Tensor, image_dest: tf.Tensor, img_res: Tuple[int, int],
                   l2: float, flow: tf.Tensor = None, batch_norm: bool = True, trainable: bool = True) -> None:
          self._image_src = image_src
          self._image_dest = image_dest
          self._img_res = img_res
          self._flow = flow
          self._l2 = l2
          self._batch_norm = batch_norm
          self._trainable = trainable
          self.flow_scale = 20
          self._scope = 'FlowNetSD'
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
          self._input = tf.concat([self._image_src, self._image_dest], axis=3, name='input_sd')
          self._downsampling()
          self._upsampling()

      def _downsampling(self) -> None:
          conv0 = Mutator.layers.Conv2D(filters=64, kernel_size=(3, 3), batch_norm=self._batch_norm,
                                        name='conv0')(Mutator.pad(self._input))
          conv1 = Mutator.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), batch_norm=self._batch_norm,
                                        name='conv1')(Mutator.pad(conv0))
          conv1_1 = Mutator.layers.Conv2D(filters=128, kernel_size=(3, 3), batch_norm=self._batch_norm,
                                          name='conv1_1')(Mutator.pad(conv1))
          conv2 = Mutator.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), batch_norm=self._batch_norm,
                                        name='conv2')(Mutator.pad(conv1_1))
          conv2_1 = Mutator.layers.Conv2D(filters=128, kernel_size=(3, 3), batch_norm=self._batch_norm,
                                          name='conv2_1')(Mutator.pad(conv2))
          conv3 = Mutator.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), batch_norm=self._batch_norm,
                                        name='conv3')(Mutator.pad(conv2_1))
          conv3_1 = Mutator.layers.Conv2D(filters=256, kernel_size=(3, 3), batch_norm=self._batch_norm,
                                          name='conv3_1')(Mutator.pad(conv3))
          conv4 = Mutator.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(2, 2), batch_norm=self._batch_norm,
                                        name='conv4')(Mutator.pad(conv3_1))
          conv4_1 = Mutator.layers.Conv2D(filters=512, kernel_size=(3, 3), batch_norm=self._batch_norm,
                                          name='conv4_1')(Mutator.pad(conv4))
          conv5 = Mutator.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(2, 2), batch_norm=self._batch_norm,
                                        name='conv5')(Mutator.pad(conv4_1))
          conv5_1 = Mutator.layers.Conv2D(filters=512, kernel_size=(3, 3), batch_norm=self._batch_norm,
                                          name='conv5_1')(Mutator.pad(conv5))
          conv6 = Mutator.layers.Conv2D(filters=1024, kernel_size=(3, 3), strides=(2, 2), batch_norm=self._batch_norm,
                                        name='conv6')(Mutator.pad(conv5_1))
          conv6_1 = Mutator.layers.Conv2D(filters=1024, kernel_size=(3, 3), batch_norm=self._batch_norm,
                                          name='conv6_1')(Mutator.pad(conv6))

      def _upsampling(self) -> None:
          flow6 = Mutator.layers.Conv2DFlow(name='flow6')(Mutator.get_operation(self._names.get('conv6_1')))
          flow6_up = Mutator.layers.Upconv(name='flow6_up')(flow6)
          deconv5 = Mutator.layers.Deconv(filters=512, name='deconv5')(Mutator.get_operation(self._names.get('conv6_1')))
          fuse5 = tf.concat([Mutator.get_operation(self._names.get('conv5_1')), deconv5, flow6_up], axis=3, name='fuse5')
          interconv5 = Mutator.layers.Conv2DInter(filters=512, name='interconv5')(fuse5)
          flow5 = Mutator.layers.Conv2DFlow(name='flow5')(interconv5)
          flow5_up = Mutator.layers.Upconv(name='flow5_up')(flow5)
          deconv4 = Mutator.layers.Deconv(filters=256, name='deconv4')(fuse5)
          fuse4 = tf.concat([Mutator.get_operation(self._names.get('conv4_1')), deconv4, flow5_up], axis=3, name='fuse4')
          interconv4 = Mutator.layers.Conv2DInter(filters=256, name='interconv4')(fuse4)
          flow4 = Mutator.layers.Conv2DFlow(name='flow4')(interconv4)
          flow4_up = Mutator.layers.Upconv(name='flow4_up')(flow4)
          deconv3 = Mutator.layers.Deconv(filters=128, name='deconv3')(fuse4)
          fuse3 = tf.concat([Mutator.get_operation(self._names.get('conv3_1')), deconv3, flow4_up], axis=3, name='fuse3')
          interconv3 = Mutator.layers.Conv2DInter(filters=128, name='interconv4')(fuse3)
          flow3 = Mutator.layers.Conv2DFlow(name='flow3')(interconv3)
          flow3_up = Mutator.layers.Upconv(name='flow3_up')(flow3)
          deconv2 = Mutator.layers.Deconv(filters=64, name='deconv2')(fuse3)
          fuse2 = tf.concat([Mutator.get_operation(self._names.get('conv2_1')), deconv2, flow3_up], axis=3, name='fuse2')
          interconv2 = Mutator.layers.Conv2DInter(filters=64, name='interconv2')(fuse2)
          flow2 = Mutator.layers.Conv2DFlow(name='flow2', scale=0.05, resize=self._image, 
                                            kernel_regularizer=tf.keras.regularizers.l2(self._l2))(interconv2)

      @property
      def inputs(self) -> Sequence[tf.Tensor]:
          return [self._image_src, self._image_dest]

      @property
      def outputs(self) -> Sequence[tf.Tensor]:
          return [Mutator.get_operation(self._names.get('flow2'))]

      def get_graph(self, dest: str = os.getcwd()) -> None:
          writer = tf.summary.FileWriter(dest, graph=tf.get_default_graph())
          writer.close()

      def _build_loss_ops(self, flow_in: tf.Tensor) -> tf.Tensor:
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
