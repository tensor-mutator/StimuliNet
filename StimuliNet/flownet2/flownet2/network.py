"""
@author: Debajyoti Raychaudhuri

An abstract template to implement
the network components
"""

from __future__ import print_function, division, absolute_import
from abc import ABCMeta, abstractmethod
import tensorflow.compat.v1 as tf
from typing import Sequence

class Network(metaclass=ABCMeta):

      @property
      @abstractmethod
      def graph_def(self) -> tf.GraphDef:
          ...
      
      @property
      @abstractmethod
      def inputs(self) -> Sequence[tf.Tensor]:
          ...

      @property
      @abstractmethod
      def outputs(self) -> Sequence[tf.Tensor]:
          ...

      @abstractmethod
      def get_graph(self, dest: str) -> None:
          ...

      def model(self, X: tf.tensor, y: tf.Tensor) -> tf.Tensor:
          return_ops = tf.import_graph_def(self.graph_def,
                                           input_map: {self.inputs[0].name: X[:, 0, :, :, :],
                                                       self.inputs[1].name: X[:, 1, :, :, :],
                                                       self.loss.input.name: y},
                                           return_elements: [self.loss.output])
          self.cost = return_ops[0]
          return self.cost
