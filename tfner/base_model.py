__license__ = """
 Copyright (C) 2017 Guillaume Genthial
 Modifications copyright (C) 2020 CEA LIST

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import os

import tensorflow as tf

from tensorflow.python.tools import freeze_graph

import time

#from subprocess import run, PIPE

input_graph_name = "input_graph.pb"
output_graph_name = "output_graph.pb"

class BaseModel(object):
    """Generic class for general methods that are not specific to NER"""

    def __init__(self, config):
        """Defines self.config and self.logger

        Args:
            config: (Config instance) class with hyper parameters,
                vocab and embeddings

        """
        self.config = config
        self.logger = config.logger
        self.sess   = None
        self.saver  = None


    def reinitialize_weights(self, scope_name):
        """Reinitializes the weights of a given layer"""
        variables = tf.contrib.framework.get_variables(scope_name)
        init = tf.variables_initializer(variables)
        self.sess.run(init)


    def add_train_op(self, lr_method, lr, loss, clip=-1):
        """Defines self.train_op that performs an update on a batch

        Args:
            lr_method: (string) sgd method, for example "adam"
            lr: (tf.placeholder) tf.float32, learning rate
            loss: (tensor) tf.float32 loss to minimize
            clip: (python float) clipping of gradient. If < 0, no clipping

        """
        _lr_m = lr_method.lower() # lower to make sure

        with tf.variable_scope("train_step"):
            if _lr_m == 'adam': # sgd method
                optimizer = tf.train.AdamOptimizer(lr)
            elif _lr_m == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(lr)
            elif _lr_m == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(lr)
            elif _lr_m == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(lr)
            else:
                raise NotImplementedError("Unknown method {}".format(_lr_m))

            if clip > 0: # gradient clipping if clip is positive
                grads, vs     = zip(*optimizer.compute_gradients(loss))
                grads, gnorm  = tf.clip_by_global_norm(grads, clip)
                self.train_op = optimizer.apply_gradients(zip(grads, vs))
            else:
                self.train_op = optimizer.minimize(loss)


    def initialize_session(self):
        """Defines self.sess and initialize the variables"""
        self.logger.info("Initializing tf session")
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()


    def restore_session(self, dir_model):
        """Reload weights into session

        Args:
            sess: tf.Session()
            dir_model: dir with weights

        """
        self.logger.info("Reloading the latest trained model...")
        self.saver.restore(self.sess, dir_model)


    def save_session(self):
        """Saves session = weights"""
        if not os.path.exists(self.config.dir_model):
            os.makedirs(self.config.dir_model)
        self.saver.save(self.sess, self.config.dir_model)


    def close_session(self):
        """Closes the session"""
        self.sess.close()


    def add_summary(self):
        """Defines variables for Tensorboard

        Args:
            dir_output: (string) where the results are written

        """
        self.merged      = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.config.dir_output,
                self.sess.graph)


    def train(self, train, dev, TRAIN=True):
        """Performs training with early stopping and lr exponential decay

        Args:
            train: dataset that yields tuple of (sentences, tags)
            dev: dataset

        """
        self.add_summary() # tensorboard
        if TRAIN:
          best_score = 0
          nepoch_no_imprv = 0 # for early stopping
          for epoch in range(self.config.nepochs):
              self.logger.info("Epoch {:} out of {:}".format(epoch + 1,
                          self.config.nepochs))

              score = self.run_epoch(train, dev, epoch)
              self.config.lr *= self.config.lr_decay # decay learning rate

              # early stopping and saving best parameters
              if score >= best_score:
                  nepoch_no_imprv = 0
                  self.save_session()
                  best_score = score
                  self.logger.info("- new best score!")
              else:
                  nepoch_no_imprv += 1
                  if nepoch_no_imprv >= self.config.nepoch_no_imprv:
                      self.logger.info("- early stopping {} epochs without "\
                              "improvement".format(nepoch_no_imprv))

                      #self.close_session()
                      #self.build_freeze()
                      #self.restore_session(config.dir_model)
                      #self.freeze_my_graph()

                      #run(['python','freezeGraph.py','--lang',self.config.language])
                      break

    def evaluate(self, test, print_results=False):
        """Evaluate model on test set

        Args:
            test: instance of class Dataset

        """
        self.logger.info("Testing model over test set")
        metrics = self.run_evaluate(test,print_results)
        msg = " - ".join(["{} {:04.2f}".format(k, v)
                for k, v in metrics.items()])
        self.logger.info(msg)

    def evaluate_on_cplusplus_api(self, test, print_results=False):
        """Evaluate model on test set

        Args:
            test: instance of class Dataset

        """
        self.logger.info("Testing model over test set")
        start_time = time.perf_counter()
        metrics = self.run_evaluate_on_cplusplus_api(test,print_results)
        print("--- Execution time : %s seconds ---" % (time.perf_counter() - start_time))
        msg = " - ".join(["{} {:04.2f}".format(k, v)
                for k, v in metrics.items()])
        self.logger.info(msg)

    def freeze_my_graph(self):
        tf.train.write_graph(self.sess.graph.as_graph_def(),self.config.dir_output, input_graph_name)
        # We save out the graph to disk, and then call the const conversion
        # routine.
        input_graph_path = os.path.join(self.config.dir_output, input_graph_name)
        input_saver_def_path = ""
        input_binary = False
        output_node_names = "proj/output_node,transitions"
        restore_op_name = "save/restore_all"
        filename_tensor_name = "save/Const:0"
        output_graph_path = os.path.join(self.config.dir_output, output_graph_name)
        clear_devices = False
        freeze_graph.freeze_graph(input_graph_path,
                      input_saver_def_path,
                      input_binary,
                      self.config.dir_model,
                      output_node_names,
                      restore_op_name,
                      filename_tensor_name,
                      output_graph_path,
                      clear_devices,
                      "","")
        #freeze_graph.freeze_graph(self.config.input_graph,
                      #input_saver_def_path,
                      #input_binary,
                      #self.config.dir_model,
                      #output_node_names,
                      #restore_op_name,
                      #filename_tensor_name,
                      #self.config.output_graph,
                      #clear_devices,
                      #"","")

