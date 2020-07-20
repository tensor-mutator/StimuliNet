from abc import ABC, abstractmethod
import tensorflow.compat.v1 as tf

class Network(ABC):

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
