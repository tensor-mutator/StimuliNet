import tensorflow.compat.v1 as tf
import tensorflow.compat.v1.keras.layers as layers

class FlowNetC(object):

      def __init__(self, image_size, batch_norm=True. div_flow=20):
          self._batch_norm = batch_norm
          self._div_flow = div_flow

      @staticmethod
      def BatchNorm():
	       def add_batch_norm(input_tensor):
              if self._batch_norm:
			      return layers.BatchNormalization(input_tensor)
              return input_tensor
          return add_batch_norm
			  
      def _build_model(self):
			 
			    