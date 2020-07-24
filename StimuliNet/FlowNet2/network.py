"""
@author: Debajyoti Raychaudhuri

An abstract template to implement
the network components
"""

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
