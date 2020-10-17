"""
@author: Debajyoti Raychaudhuri

An abstract template to implement
the network components
"""

from __future__ import print_function, division, absolute_import
from abc import ABCMeta, abstractmethod
import tensorflow.compat.v1 as tf
from typing import Sequence
import os

class Network(metaclass=ABCMeta):
      
      @property
      @abstractmethod
      def inputs(self) -> Sequence[tf.Tensor]:
          ...

      @property
      @abstractmethod
      def outputs(self) -> Sequence[tf.Tensor]:
          ...

      def get_graph(self, dest: str = os.getcwd()) -> None:
          writer = tf.summary.FileWriter(dest, graph=tf.get_default_graph())
          writer.close()
