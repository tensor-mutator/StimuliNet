"""
@author: Debajyoti Raychaudhuri

A TensorFlow implementation of FlowNetFusion
"""

from __future__ import print_function, division, absolute_import
import tensorflow.compat.v1 as tf
import tensorflow.compat.v1.keras.layers as layers
from typing import Tuple, Callable, Sequence
import os
from ..mutator import Mutator
from ..network import Network
from ..exceptions import *

class FlowNetFusion(Network):

      def __init__(self, flownet_css_sd_patch: tf.Tensor, img_res: Tuple[int, int], l2: float, flow: tf.Tensor = None,
                   batch_norm: bool = True, trainable: bool = True) -> None:
          self._batch_norm = batch_norm
          self._flownet_css_sd_patch = flownet_css_sd_patch
          self._img_res = img_res
          self._flow = flow
          self._l2 = l2
          self._trainable = trainable
          self._scope = "FlowNetFusion"
          self._build_graph_with_scope()

      def _build_graph_with_scope(self) -> None:
          with tf.variable_scope(self._scope):
               self._build_graph()

      def _build_graph(self) -> None:
          Mutator.trainable = self._trainable
          Mutator.scope(self._scope)
          self._downsampling()
          self._upsampling()

      def _downsampling(self) -> None:
          conv0 = Mutator.layers.Conv2D(filters=64, kernel_size=(3, 3), batch_norm=self._batch_norm,
                                        name='conv0')(Mutator.pad(self._flownet_css_sd_patch, channels=11, resolution=self._img_res))
          conv1 = Mutator.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), batch_norm=self._batch_norm, name='conv1')(Mutator.pad(conv0))
          conv1_1 = Mutator.layers.Conv2D(filters=128, kernel_size=(3, 3), batch_norm=self._batch_norm, name='conv1_1')(Mutator.pad(conv1))
          conv2 = Mutator.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), batch_norm=self._batch_norm, name='conv2')(Mutator.pad(conv1_1))
          conv2_1 = Mutator.layers.Conv2D(filters=128, kernel_size=(3, 3), batch_norm=self._batch_norm, name='conv2_1')(Mutator.pad(conv2))

      def _upsampling(self) -> None:
          flow2 = Mutator.layers.Conv2DFlow(name='flow2')(Mutator.get_operation(self._names.get('conv2_1')))
          flow2_up = Mutator.layers.Upconv(name='flow2_up')(flow2)
          deconv1 = Mutator.layers.Deconv(filters=32, name='deconv1')(Mutator.get_operation(self._names.get('conv2_1')))
          fuse1 = tf.concat([Mutator.get_operation(self._names.get('conv1_1')), deconv1, flow2_up], axis=3, name='fuse1')
          interconv1 = Mutator.layers.Conv2DInter(filters=32, name='interconv1')(fuse1)
          flow1 = Mutator.layers.Conv2DFlow(name='flow1')(interconv1)
          flow1_up = Mutator.layers.Upconv(name='flow1_up')(flow1)
          deconv0 = Mutator.layers.Deconv(filters=16, name='deconv0')(fuse1)
          fuse0 = tf.concat([Mutator.get_operation(self._names.get('conv0')), deconv0, flow1_up], axis=3, name='fuse0')
          interconv0 = Mutator.layers.Conv2DInter(filters=16, name='interconv0')(fuse0)
          flow0 = Mutator.layers.Conv2DFlow(name='flow0', resize=self._img_res,
                                            kernel_regularizer=tf.keras.regularizers.l2(self._l2))(interconv0)

      @property
      def inputs(self) -> Sequence[tf.Tensor]:
          return [self._flownet_css_sd_patch]

      @property
      def outputs(self) -> Sequence[tf.Tensor]:
          return [Mutator.get_operation(self._names.get('flow0'))]
