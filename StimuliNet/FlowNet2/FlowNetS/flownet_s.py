"""
@author: Debajyoti Raychaudhuri

A TensorFlow implementation of FlowNetS
"""

from __future__ import print_function, division, absolute_import
import tensorflow.compat.v1 as tf
import tensorflow.comapt.v1.keras.layers as layers
from typing import Tuple, Callable
from ..mutator import Mutator

class FlowNetS(object):

      def __init__(self, input_channels: int = 12, batch_norm: bool = True) -> None:
          self._input_channels = input_channels
          self._batch_norm = batch_norm
          self._scope = 'FlowNetS'
          self._build_graph_with_scope()

      def _build_graph_with_scope(self) -> tf.Graph:
          self.graph = tf.Graph()
          with self.graph.as_default():
               with tf.variable_scope(self._scope):
                    self._build_graph()
          return self.graph

      def _build_graph(self) -> None:
          
