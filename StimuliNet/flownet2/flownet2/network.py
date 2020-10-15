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

      def model(self, X: tf.Tensor, y: tf.Tensor = None) -> None:
          self.src_img, self.dest_img = self.inputs[0], self.inputs[1]
          self.src_img, self.dest_img = X[:, 0, :, :, :], X[:, 1, :, :, :]
          self.y = self.outputs[0]
          if y:
             self.loss.input = y
             self.cost = self.loss.output
