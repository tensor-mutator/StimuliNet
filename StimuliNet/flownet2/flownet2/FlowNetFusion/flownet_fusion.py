"""
@author: Debajyoti Raychaudhuri

A TensorFlow implementation of FlowNetFusion
"""

from __future__ import print_function, division, absolute_import
import warnings
with warnings.catch_warnings():  
     warnings.filterwarnings("ignore", category=FutureWarning)
     import tensorflow.compat.v1 as tf
     import tensorflow.compat.v1.keras.layers as layers
from typing import Tuple, Callable, Sequence
from FlowNet2.mutator import Mutator
from FlowNet2.network import Network
import os

class FlowNetFusion(Network):

      def __init__(self, patch: tf.TensorShape, batch_norm: bool = True, trainable: bool = True) -> None:
          self._batch_norm = batch_norm
          self._patch = patch
          self._trainable = trainable
          self._scope = 'FlowNetFusion'
          self._build_graph_with_scope()

      def _build_graph_with_scope(self) -> tf.Graph:
          self.graph = tf.Graph()
          with self.graph.as_default():
               with tf.variable_scope(self._scope):
                    self._build_graph()
          return self.graph

      def _build_graph(self) -> None:
          Mutator.set_graph(self.graph)
          Mutator.trainable = self._trainable
          self._input = tf.placeholder(dtype=tf.float32, shape=self._patch, name='fusion_input')
          self._downsampling()
          self._upsampling()

      @property
      def graph_def(self) -> tf.GraphDef:
          return self.graph.as_graph_def()

      def _downsampling(self) -> None:
          conv0 = Mutator.Conv2D(filters=64, kernel_size=(3, 3), batch_norm=self._batch_norm, name='conv0')(self._input)
          conv1 = Mutator.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), batch_norm=self._batch_norm, name='conv1')(conv0)
          conv1_1 = Mutator.Conv2D(filters=128, kernel_size=(3, 3), batch_norm=self._batch_norm, name='conv1_1')(conv1)
          conv2 = Mutator.Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), batch_norm=self._batch_norm, name='conv2')(conv1_1)
          conv2_1 = Mutator.Conv2D(filters=128, kernel_size=(3, 3), batch_norm=self._batch_norm, name='conv2_1')(conv2)

      def _upsampling(self) -> None:
          flow2 = Mutator.PredictFlow(name='flow2')(Mutator.get_operation(self._names.get('conv2_1')))
          flow2_up = Mutator.Conv2DTranspose(filters=2, kernel_size=(4, 4), strides=(2, 2), padding=1, name='flow2_up')(flow2)
          deconv1 = Mutator.Deconv(filters=32, name='deconv1')(Mutator.get_operation(self._names.get('conv2_1')))
          fuse1 = tf.concat([Mutator.get_operation(self._names.get('conv1_1')), deconv1, flow2_up], axis=1, name='fuse1')
          interconv1 = Mutator.Conv2DInter(filters=32, kernel_size=(3, 3), batch_norm=self._batch_norm, name='interconv1')(fuse1)
          flow1 = Mutator.PredictFlow(name='flow1')(interconv1)
          flow1_up = Mutator.Conv2DTranspose(filters=2, kernel_size=(4, 4), strides=(2, 2), padding=1, name='flow1_up')(flow1)
          deconv0 = Mutator.Deconv(filters=16, name='deconv0')(fuse1)
          fuse0 = tf.concat([Mutator.get_operation(self._names.get('conv0')), deconv0, flow1_up], axis=1, name='fuse0')
          interconv0 = Mutator.Conv2DInter(filters=16, kernel_size=(3, 3), batch_norm=self._batch_norm, name='interconv0')(fuse0)
          flow0 = Mutator.PredictFlow(name='flow0')(interconv0)

      @property
      def inputs(self) -> Sequence[tf.Tensor]:
          return [self._input]

      @property
      def outputs(self) -> Sequence[tf.Tensor]:
          return [Mutator.get_operation(self._names.get('flow0'))]

      def get_graph(self, dest: str = os.getcwd()) -> None:
          writer = tf.summary.FileWriter(dest, graph=self.graph)
          writer.close()
