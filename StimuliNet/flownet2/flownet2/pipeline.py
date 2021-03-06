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
from typing import Dict, List, Generator, Any, Tuple
import cv2
import re
from .network import Network
from .exceptions import *
from .config import config
from .flowkit import write_flow, flow_to_image
from .mutator import Mutator
from .ansi import ANSI

class Pipeline:

      def __init__(self, network: Network, schedule: str, img_resolution: Tuple[int, int],
                   flow_resolution: Tuple[int, int], checkpoint_path: str, frozen_config: List = None,
                   config: bin = config.DEFAULT) -> None:
          self._network = network
          self._read_params(schedule)
          self._img_res = img_resolution
          self._flow_res = flow_resolution
          self._config = config
          self._X_src_placeholder = tf.placeholder(shape=(None,) + img_resolution + (3,), dtype=tf.float32)
          self._X_dest_placeholder = tf.placeholder(shape=(None,) + img_resolution + (3,), dtype=tf.float32)
          self._y_placeholder = tf.placeholder(shape=(None,) + flow_resolution + (2,), dtype=tf.float32)
          self._iterator = self._generate_iterator()
          self._local_model = self._generate_local_graph(network)
          self._predict_model = self._generate_target_graph(network)
          self._session = tf.Session(config=self._get_config())
          self._model_name = network.__name__
          self._load_frozen_weights(frozen_config)
          self._checkpoint_dir, self._flow_dir = self._generate_checkpoint_directory(checkpoint_path)

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
          self._l2 = params.get("l2", 0.0004)
          lr_scheduler_callback = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, learning_rates)
          step = tf.Variable(0, trainable=False)
          self._lr = lr_scheduler_callback(step)
          self._optimizer = tf.train.AdamOptimizer
          self._batch_size = params.get("batch_size", 64)
          self._n_epoch = params.get("epoch", 10000)
          self._batch_norm = params.get("batch_norm", False)

      @contextmanager
      def _fit_context(self) -> Generator:
          epoch = self._deserialize_weights()
          train_writer, test_writer = self._generate_summary_writer()
          yield self._session, train_writer, test_writer, epoch
          #self._serialize_weights()
          if train_writer and test_writer:
             train_writer.close()
             test_writer.close()

      @contextmanager
      def _epoch_context(self, epoch) -> Generator:
          yield
          self._serialize_weights(epoch)

      def _deserialize_weights(self) -> None:
          if self._config & config.LOAD_WEIGHTS:
             self._load_weights()
             return self._load_epoch()
          return 0

      def _load_epoch(self) -> int:
          if glob(os.path.join(self._checkpoint_dir, "{}.ckpt.*".format(self._model_name))):
             with open(os.path.join(self._checkpoint_dir, "{}.ckpt.epoch".format(self._model_name)), "r") as f:
                  return int(f.read().strip("/n"))
          return 0

      def _load_weights(self) -> None:
          with self._session.graph.as_default():
               var_list_local_to = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=f"local/{self._model_name}")
               var_list_target_to = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=f"target/{self._model_name}")
               var_list_from = list(map(lambda x: x.name.replace(":0", ""), var_list_local_to))
               if glob(os.path.join(self._checkpoint_dir, "{}.ckpt.*".format(self._model_name))):
                  ckpt = tf.train.get_checkpoint_state(self._checkpoint_dir).model_checkpoint_path
                  saver = tf.train.Saver(var_list=dict(zip(var_list_from, var_list_local_to)))
                  saver.restore(self._session, ckpt)
                  saver = tf.train.Saver(var_list=dict(zip(var_list_from, var_list_target_to)))
                  saver.restore(self._session, ckpt)

      def _save_weights(self) -> None:
          self._session.run(self._update_ops)
          with self._session.graph.as_default():
               saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=f"local/{self._model_name}"))
          saver.save(self._session, os.path.join(self._checkpoint_dir, "{}.ckpt".format(self._model_name)))

      def _save_epoch(self, epoch) -> None:
          with open(os.path.join(self._checkpoint_dir, "{}.ckpt.epoch".format(self._model_name)), "w") as f:
               return f.write(str(epoch))

      def _serialize_weights(self, epoch) -> None:
          self._save_weights()
          self._save_epoch(epoch)

      def _generate_iterator(self) -> tf.data.Iterator:
          dataset = tf.data.Dataset.from_tensor_slices((self._X_src_placeholder, self._X_dest_placeholder, self._y_placeholder))
          dataset = dataset.shuffle(tf.cast(tf.shape(self._y_placeholder)[0], tf.int64)).batch(self._batch_size).prefetch(1)
          return dataset.make_initializable_iterator()

      def _generate_checkpoint_directory(self, checkpoint_path) -> str:
          weight_dir = os.path.join(checkpoint_path, "weights")
          if not os.path.exists(weight_dir):
             os.mkdir(weight_dir)
          flow_dir = None
          if self._config & config.SAVE_FLOW:
             flow_dir = os.path.join(checkpoint_path, "flows")
             if not os.path.exists(flow_dir):
                os.mkdir(flow_dir)
          return weight_dir, flow_dir

      def _generate_local_graph(self, network: Network) -> Network:
          with tf.variable_scope("local"):
               Mutator.scope("local")
               network = self._get_model(network)
               network.grad = self._optimizer(self._lr, self._beta1, self._beta2, self._epsilon).minimize(network.loss.output)
          return network

      def _load_frozen_weights(self, frozen_config: List) -> None:
          def patch_ops(ckpt_op_names: List, patch: List) -> List:
              patched_ops = list()
              for op in ckpt_op_names:
                  for op_ in patch:
                      op_name = op_["op"]
                      decrement_val = op_["val"]
                      re_obj = re.search(op_name + r"[0-9_]{0,}/", op)
                      if re_obj:
                         sub = re_obj.group()
                         val = re.search(r"_[0-9]{1,}", sub).group().replace("_", "")
                         if int(val) - decrement_val == 0:
                            sub_new = sub.replace("_{}".format(val), "")
                         else:
                            sub_new = sub.replace(val, str(int(val) - decrement_val))
                         op = op.replace(sub, sub_new)
                  patched_ops.append(op)
              return patched_ops
          self._session.run(tf.global_variables_initializer())
          if frozen_config is None:
             return
          for conf in frozen_config:
              scope, path = conf["scope"], conf["path"]
              patch = conf.get("patch", None)
              ckpt = tf.train.get_checkpoint_state(path)
              if ckpt is None:
                 raise WeightsNotFoundError("weights not found for scope: {}".format(scope))
              ckpt_path = ckpt.model_checkpoint_path
              var_list_local_to = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=f"local/{self._model_name}/{scope}")
              var_list_target_to = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=f"target/{self._model_name}/{scope}")
              var_list_from = list(map(lambda x: x.name.replace(f"{self._model_name}/", "").replace(":0", ""), var_list_local_to))
              if patch:
                 var_list_from = patch_ops(var_list_from, patch)
              saver = tf.train.Saver(var_list=dict(zip(var_list_from, var_list_local_to)))
              saver.restore(self._session, ckpt_path)
              saver = tf.train.Saver(var_list=dict(zip(var_list_from, var_list_target_to)))
              saver.restore(self._session, ckpt_path)

      def _get_model(self, network: Network) -> Network:
          X_src_, X_dest_, y_ = self._iterator.get_next()
          return network(X_src_, X_dest_, self._img_res, self._l2, y_, self._batch_norm)

      def _generate_target_graph(self, network: Network) -> Network:
          with tf.variable_scope("target"):
               Mutator.reset_scope()
               Mutator.scope("target")
               self._X_src_predict = tf.placeholder(shape=(None,) + self._img_res + (3,), dtype=tf.float32, name="X_src")
               self._X_dest_predict = tf.placeholder(shape=(None,) + self._img_res + (3,), dtype=tf.float32, name="X_dest")
               network = network(self._X_src_predict, self._X_dest_predict, self._img_res, self._l2, batch_norm=self._batch_norm)
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

      def _fit(self, X_src_train: np.ndarray, X_src_test: np.ndarray, X_dest_train: np.ndarray, X_dest_test: np.ndarray,
               y_train: np.ndarray, y_test: np.ndarray, session: tf.Session,
               train_writer: tf.summary.FileWriter, test_writer: tf.summary.FileWriter, epoch_: int) -> None:
          def run_(session, total_loss, train=True) -> List:
              if train:
                 _, loss = session.run([self._local_model.grad, self._local_model.loss.output])
                 flow_payload = None
              else:
                 loss, flow, src_img, dest_img = session.run([self._local_model.loss.output, self._local_model.outputs[0], self._local_model.inputs[0],
                                                              self._local_model.inputs[1]])
                 flow_payload = [src_img, dest_img, flow]
              total_loss += loss
              return total_loss, flow_payload
          n_batches_train = np.ceil(np.size(y_train, axis=0)/self._batch_size)
          n_batches_test = np.ceil(np.size(y_test, axis=0)/self._batch_size)
          with session.graph.as_default():
               for epoch in range(epoch_, self._n_epoch):
                   with self._epoch_context(epoch+1):
                        self._count = 0
                        train_loss = 0
                        test_loss = 0
                        session.run(self._iterator.initializer, feed_dict={self._X_src_placeholder: X_src_train,
                                                                           self._X_dest_placeholder: X_dest_train,
                                                                           self._y_placeholder: y_train})
                        with tqdm(total=len(y_train)) as progress:
                             try:
                                while True:
                                      train_loss, _ = run_(session, train_loss)
                                      progress.update(self._batch_size)
                             except tf.errors.OutOfRangeError:
                                ...
                        session.run(self._iterator.initializer, feed_dict={self._X_src_placeholder: X_src_test,
                                                                           self._X_dest_placeholder: X_dest_test,
                                                                           self._y_placeholder: y_test})
                        with tqdm(total=len(y_test)) as progress:
                             try:
                                while True:
                                      test_loss, flow_payload = run_(session, test_loss, train=False)
                                      self._save_flow(epoch+1, *flow_payload)
                                      progress.update(self._batch_size)
                             except tf.errors.OutOfRangeError:
                                 ...
                        self._print_summary(epoch+1, train_loss, n_batches_train, test_loss, n_batches_test)
                        self._save_summary(train_writer, epoch=epoch+1, loss=train_loss, n_batches=n_batches_train)
                        self._save_summary(test_writer, epoch=epoch+1, loss=test_loss, n_batches=n_batches_test)

      def _save_flow(self, epoch: int, src_img: np.ndarray, dest_img: np.ndarray, flow: np.ndarray) -> None:
          if self._config & config.SAVE_FLOW:
             path = os.path.join(self._flow_dir, "EPOCH {}".format(str(epoch).zfill(10)))
             if not os.path.exists(path):
                os.mkdir(path)
             for src_img_, dest_img_, flow_ in zip(src_img, dest_img, flow):
                 self._count += 1
                 final_path = os.path.join(path, "flow.{}".format(str(self._count).zfill(10)))
                 os.mkdir(final_path)
                 cv2.imwrite(os.path.join(final_path, "src.png"), src_img_)
                 cv2.imwrite(os.path.join(final_path, "dest.png"), dest_img_)
                 cv2.imwrite(os.path.join(final_path, "flow.png"), flow_to_image(flow_))
                 write_flow(flow_, os.path.join(final_path, "flow.flo"))

      def _print_summary(self, epoch: int, train_loss: float, n_batches_train: int,
                         test_loss: float, n_batches_test: int) -> None:
          print(f"{ANSI.UP}\r{ANSI.WIPE}\n{ANSI.WIPE}EPOCH: {ANSI.CYAN}{epoch}{ANSI.DEFAULT}")
          print(f"\n\tTraining set:")
          print(f"\n\t\tLoss: {ANSI.GREEN}{train_loss/n_batches_train}{ANSI.DEFAULT}")
          print(f"\n\tTest set:")
          print(f"\n\t\tLoss: {ANSI.MAGENTA}{test_loss/n_batches_test}{ANSI.DEFAULT}")

      def fit(self, X_src_train: np.ndarray, X_src_test: np.ndarray, X_dest_train: np.ndarray, X_dest_test: np.ndarray,
              y_train: np.ndarray, y_test: np.ndarray) -> None:
          with self._fit_context() as [session, train_writer, test_writer, epoch]:
               self._fit(X_src_train, X_src_test, X_dest_train, X_dest_test, y_train, y_test, session, train_writer, test_writer, epoch)

      def predict(self, X_src: np.ndarray, X_dest: np.ndarray) -> tf.Tensor:
          self._load_weights()
          with self._session.graph.as_default():
               return self._session.run(self._predict_model.outputs[0], feed_dict={self._X_src_predict: X_src,
                                                                          self._X_dest_predict: X_dest})

      def save_graph(self, path: str) -> None:
          self._local_model.get_graph(path)

      def __del__(self) -> None:
          self._session.close()
