from abc import ABCMeta, abstractmethod
import tensorflow.compat.v1 as tf
from typing import Sequence

class Network(metaclass=ABCMeta):

      @abstractmethod
      def _build_graph(self) -> None:
          ...

      @abstractmethod
      def inputs(self) -> Sequence[tf.Tensor]:
          ...

      @abstractmethod
      def outputs(self) -> Sequence[tf.Tensor]:
          ...

      @abstractmethod
      def get_graph(self, dest: str) -> None:
          ...
