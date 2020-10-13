import tensorflow.compat.v1 as tf
import os
import json
from typing import List
from .network import Network
from .exceptions import *

class Pipeline:

      def __init__(self, network: Network, schedule: str, img_resolution: Tuple[int, int],
                   flow_resolution: Tuple[int, int]) -> None:
          self._network = network
          self._read_params(schedule)
          self._X_placeholder = tf.placeholder(shape=(None, 2,) + img_resolution + (3,), dtype=tf.float32)
          self._y_placeholder = tf.placeholder(shape=(None,) + flow_resolution + (2,), dtype=tf.float32)

      def _read_params(self, schedule: str) -> None:
          with open(os.path.join(os.path.split(__file__)[0], "flownet2.hyperparams"), "r") as f_obj:
               params = json.load(f_obj).get(schedule, None)
          if params is None:
             raise ScheduleNotFoundError(f"Schedule: {schedule} not found")
          boundaries = params["boundaries"]
          learning_rates = params["learning_rates"]
          beta1 = params.get("beta1", 0.9)
          beta2 = params.get("beta2", 0.999)
          epsilon = params.get("epsilon", 1e-08)
          lr_scheduler_callback = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, learning_rates)
          step = tf.Variable(0, trainable=False)
          self._optimizer = tf.train.AdamOptimizer(lr_scheduler_callback(step), beta1, beta2, epsilon)
          self._batch_size = params.get("batch_size", 64)
          self._epoch = params.get("epoch", 10000)

      def fit(self, X_train: tf.Tensor, X_test: tf.Tensor, y_train: tf.Tensor, y_test: tf.Tensor) -> None:
          

      def predict(self, X: tf.Tensor) -> tf.Tensor:
          
