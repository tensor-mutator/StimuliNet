""""
@author: Debajyoti Raychaudhuri

A training and testing pipeline to train and test
FlowNet 2.0 network
"""

import tensorflow.compat.v1 as tf
import os
import json
from contextlib import contextmanager
import numpy as np
from tqdm import tqdm
from glob import glob
from typing import Dict, List, Generator, Any
from .network import Network
from .exceptions import *
from .config import config

GREEN = "\033[32m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
DEFAULT = "\033[0m"
WIPE = "\033[2K"
UP = "\033[2A"

class Pipeline:

      def __init__(self, network: Network, schedule: str, img_resolution: Tuple[int, int],
                   flow_resolution: Tuple[int, int], frozen_config: List = None,
                   config: bin = config.DEFAULT) -> None:
          self._network = network
          self._read_params(schedule)
          self._img_res = img_resolution
          self._flow_res = flow_resolution
          self._config = config
          self._X_placeholder = tf.placeholder(shape=(None, 2,) + img_resolution + (3,), dtype=tf.float32)
          self._y_placeholder = tf.placeholder(shape=(None,) + flow_resolution + (2,), dtype=tf.float32)
          self._iterator = self._generate_iterator()
          self._local_model = self._generate_local_graph(network)
          self._predict_model = self._generate_target_graph(network)
          self._session = tf.Session(config=self._get_config())
          self._load_frozen_weights(frozen_config)
          self._model_name = network.__class__.__name__
          self._checkpoint_dir = self._generate_checkpoint_directory()

      def _read_params(self, schedule: str) -> None:
          with open(os.path.join(os.path.split(__file__)[0], "flownet2.hyperparams"), "r") as f_obj:
               params = json.load(f_obj).get(schedule, None)
          if params is None:
             raise ScheduleNotFoundError(f"Schedule: {schedule} not found")
          boundaries = params["boundaries"]
          learning_rates = params["learning_rates"]
          self._beta1 = params.get("beta1", 0.9)
          self._beta2 = params.get("beta2", 0.999)
          self._epsilon = params.get("epsilon", 1e-08)
          lr_scheduler_callback = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, learning_rates)
          step = tf.Variable(0, trainable=False)
          self._lr = lr_scheduler_callback(step)
          self._optimizer = tf.train.AdamOptimizer
          self._batch_size = params.get("batch_size", 64)
          self._n_epoch = params.get("epoch", 10000)
          self._batch_norm = params.get("batch_norm", False)

      @contextmanager
      def _fit_context(self) -> Generator:
          self._load_weights()
          train_writer, test_writer = self._generate_summary_writer()
          yield self._session, train_writer, test_writer
          self._save_weights()
          if train_writer and test_writer:
             train_writer.close()
             test_writer.close()

      def _load_weights(self) -> None:
          if self._config & config.LOAD_WEIGHTS:
             with self._session.graph.as_default():
                  self._saver = tf.train.Saver(max_to_keep=5)
                  if glob(os.path.join(self._checkpoint_dir, "{}.ckpt.*".format(self._model_name))):
                     ckpt = tf.train.get_checkpoint_state(self._checkpoint_dir)
                     self._saver.restore(self._session, ckpt.model_checkpoint_path)

      def _save_weights(self) -> None:
          if getattr(self, "_saver", None) is None:
             with self._session.graph.as_default():
                  self._saver = tf.train.Saver(max_to_keep=5)
          self._session.run(self._update_ops)
          self._saver.save(self._session, os.path.join(self._checkpoint_dir, "{}.ckpt".format(self._model_name)))

      def _generate_iterator(self) -> tf.data.Iterator:
          dataset = tf.data.Dataset.from_tensor_slices((self._X_placeholder, self._y_placeholder))
          dataset = dataset.shuffle(tf.cast(tf.shape(self._X_placeholder)[0], tf.int64)).batch(self._batch_size).prefetch(1)
          return dataset.make_initializable_iterator()

      def _generate_checkpoint_directory(self) -> str:
          dir = os.path.join(os.path.split(__file__)[0], "weights")
          if not os.path.exists(dir):
             os.mkdir(dir)
          return dir

      def _generate_local_graph(self, network: Network) -> Network:
          with tf.variable_scope("local"):
               network = network(self._img_res, self._flow_res, self._batch_norm)
               self._get_model(network)
               network.grad = self._optimizer(self._lr, self._beta1, self._beta2, self._epsilon).minimize(network.cost)
          return network

      def _load_frozen_weights(self, frozen_config: List) -> None:
          self._session.run(tf.global_variables_initializer())
          if frozen_config is None:
             return
          for conf in frozen_config:
              scope, path = conf.items()[0]
              ckpt_path = tf.train.get_checkpoint_state(path).model_checkpoint_path
              saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=f"local/{scope}-Graph/{scope}"))
              saver.restore(self._session, ckpt_path)
              saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=f"target/{scope}-Graph/{scope}"))
              saver.restore(self._session, ckpt_path)

      def _get_model(self, network: Network) -> None:
          X_, y_ = self._iterator.get_next()
          network.model(X_, y_)

       def _generate_target_graph(self, network: Ntwork) -> Network:
          with tf.variable_scope("target"):
               self._X_predict = tf.placeholder(shape=(None, 2,) + self._img_res + (3,), dtype=tf.float32, name="X")
               network = network(self._img_res, self._flow_res, self._batch_norm)
               network.model(self._X_predict)
          return network

      def _get_config(self) -> tf.ConfigProto:
          config = tf.ConfigProto()
          config.gpu_options.allow_growth = True
          return config

      def _generate_summary_writer(self) -> Any:
          if self._config & config.LOSS_EVENT:
             train_writer = tf.summary.FileWriter(os.path.join(self._checkpoint_dir, "{} TRAIN EVENTS".format(self._model_name)), self._session.graph)
             test_writer = tf.summary.FileWriter(os.path.join(self._checkpoint_dir, "{} TEST EVENTS".format(self._model_name)), self._session.graph)
             return train_writer, test_writer
          return None, None

      @property
      def _update_ops(self) -> tf.group:
          trainable_vars_local = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="local")
          trainable_vars_target = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="target")
          update_ops = list()
          for from_ ,to_ in zip(trainable_vars_local, trainable_vars_target):
              update_ops.append(to_.assign(from_))
          return tf.group(update_ops)

      def _save_summary(self, writer: tf.summary.FileWriter, epoch: int, loss: float, n_batches: int) -> None:
          summary = tf.Summary()
          if self._config & config.LOSS_EVENT:
             summary.value.add(tag="{} Performance/Epoch - Loss".format(self._model_name), simple_value=loss/n_batches)
          if writer:
             writer.add_summary(summary, epoch)

      def _fit(self, X_train: np.ndarray, X_test: np.ndarray,
               y_train: np.ndarray, y_test: np.ndarray, session: tf.Session,
               train_writer: tf.summary.FileWriter, test_writer: tf.summary.FileWriter) -> None:
          def run_(session, total_loss, train=True) -> List:
              if train:
                 _, loss = session.run([self._local_model.grad, self._local_model.cost])
              else:
                 loss = session.run(self._local_model.cost)
              total_loss += loss
              return total_loss
          n_batches_train = np.ceil(np.size(y_train, axis=0)/self._batch_size)
          n_batches_test = np.ceil(np.size(y_test, axis=0)/self._batch_size)
          with session.graph.as_default():
               for epoch in range(self._n_epoch):
                   train_loss = 0
                   test_loss = 0
                   session.run(self._iterator.initializer, feed_dict={self._X_placeholder: X_train,
                                                                      self._y_placeholder: y_train})
                   with tqdm(total=len(y_train)) as progress:
                        try:
                           while True:
                                 train_loss = run_(session, train_loss)
                                 progress.update(self._batch_size)
                        except tf.errors.OutOfRangeError:
                           ...
                   session.run(self._iterator.initializer, feed_dict={self._X_placeholder: X_test,
                                                                      self._y_placeholder: y_test})
                   with tqdm(total=len(y_test)) as progress:
                        try:
                           while True:
                                 test_loss = run_(session, test_loss, train=False)
                                 progress.update(self._batch_size)
                        except tf.errors.OutOfRangeError:
                           ...
                   self._print_summary(epoch+1, train_loss, n_batches_train, test_loss, n_batches_test)
                   self._save_summary(train_writer, epoch=epoch+1, loss=train_loss, n_batches=n_batches_train)
                   self._save_summary(test_writer, epoch=epoch+1, loss=test_loss, n_batches=n_batches_test)

      def _print_summary(self, epoch: int, train_loss: float, n_batches_train: int,
                         test_loss: float, n_batches_test: int) -> None:
          print(f"{UP}\r{WIPE}\n{WIPE}EPOCH: {CYAN}{epoch}{DEFAULT}")
          print(f"\n\tTraining set:")
          print(f"\n\t\tLoss: {GREEN}{train_loss/n_batches_train}{DEFAULT}")
          print(f"\n\tTest set:")
          print(f"\n\t\tLoss: {MAGENTA}{test_loss/n_batches_test}{DEFAULT}")

      def fit(self, X_train: tf.Tensor, X_test: tf.Tensor, y_train: tf.Tensor, y_test: tf.Tensor) -> None:
          with self._fit_context() as [session, train_writer, test_writer]:
               self._fit(X_train, X_test, y_train, y_test, session, train_writer, test_writer)

      def predict(self, X: tf.Tensor) -> tf.Tensor:
          self._load_weights()
          with self._session.graph.as_default():
               return self._session.run(self._predict_model.y, feed_dict={self._X_predict: X})

      def __del__(self) -> None:
          self._session.close()
