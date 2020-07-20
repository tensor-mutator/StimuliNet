"""
@author: Debajoti Raychaudhuri

A TensorFlow implementation of FlowNetCSS
"""

from FlowNet2.network import Network
from FlowNet2.FlowNetC import FlowNetCS
from FlowNet.FlowNetS import FlowNetS
from FlowNet2.mutator import Mutator
from FlowNet2.flow_warp import FlowWarp
from typing import Tuple, Sequence
import tensorflow.compat.v1 as tf

class FlowNetCSS(Network):

      def __init__(self, image: Tuple[int, int, int], batch_norm: bool = True, div_flow: int = 20) -> None:
          self._image = image
          self._batch_norm = batch_norm
          self._div_flow = div_flow
          self._scope = 'FlowNetCSS'

      def _build_graph_with_scope(self) -> tf.Graph:
          self.graph = tf.Graph()
          with self.graph.as_default():
               with tf.variable_scope(self._scope):
                    self._build_graph()
          return self.graph

      @property
      def graph_def(self):
          return self.graph.as_graph_def()

      def _build_graph(self) -> None:
          Mutator.set_graph(self.graph)
          self._image_1 = tf.placeholder(shape=(None,) + image, dtype=tf.float32, name='image_1_css')
          self._image_2 = tf.placeholder(shape=(None,) + image, dtype=tf.float32, name='image_2_css')
          flownet_cs = FlowNetCS(self._image, self._batch_norm, self._div_flow)
          flownet_cs_patch = tf.import_graph_def(flownet_cs.graph_def,
                                                input_map={x.name: [self._image_1, self._image_2][i] for i, x in enumerate(flownet_c.inputs)},
                                                return_elements=list(map(lambda x: x.name, flownet_c.outputs)))
          flownet_s_input_tensor = self._compute_input_tensor_flow_flownet_s(self._image_1, self._image_2, flownet_cs_patch)
          flownet_s = FlowNetS(flownet_s_input_tensor.get_shape(), self._batch_norm)
          self._flownet_css_patch = tf.import_graph_def(flownet_s.graph_def,
                              input_map={flownet_s.inputs[0].name: flownet_s_input_tensor},
                              return_elements=list(map(lambda x: x.name, flownet_s.outputs)))

      def _compute_input_tensor_for_flownet_s(self, image_1: tf.Tensor, image_2: tf.Tensor, flow_out: tf.Tensor) -> tf.Tensor:
          warped = FlowWarp(image_2, flow_out)
          brightness_error = tf.sqrt(tf.reduce_sum(tf.square(image_1 - warped), axis=-1, keep_dims=True))
          return brightness_error

      @property
      def inputs(self) -> Sequence[tf.Tensor]:
          return [self._image_1, self._image_2]

      @property
      def outputs(self) -> equence[tf.Tensor]:
          return [self._flownet_css_patch]

      def get_graph(self, dest: str = os.getcwd()) -> None:
          writer = tf.summary.FileWriter(dest, graph=self.graph)
          writer.close()
