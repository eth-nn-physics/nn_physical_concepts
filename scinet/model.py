#   Copyright 2018 SciNet (https://github.com/eth-nn-physics/nn_physical_concepts)
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import numpy as np
import tensorflow as tf
import cPickle as pickle
import io
from tqdm import tqdm_notebook


class Network(object):

    def __init__(self, input_size, latent_size, input2_size, output_size,
                 encoder_num_units=[100, 100], decoder_num_units=[100, 100], name='Unnamed',
                 tot_epochs=0, load_file=None):
        """
        Parameters:
        input_size: length of a single data vector.
        latent_size: number of latent neurons to be used.
        input2_size: number of neurons for 2nd input into decoder.
        output_size: length of a single label vector.
        encoder_num_units, decoder_num_units: Number of neurons in encoder and decoder hidden layers. Everything is fully connected.
        name: Used for tensorboard
        tot_epochs and  load_file are used internally for loading and saving, don't pass anything to them manually.
        """

        self.graph = tf.Graph()

        self.input_size = input_size
        self.latent_size = latent_size
        self.input2_size = input2_size
        self.output_size = output_size
        self.encoder_num_units = encoder_num_units
        self.decoder_num_units = decoder_num_units
        self.name = name
        self.tot_epochs = tot_epochs

        # Set up neural network
        self.graph_setup()
        self.session = tf.Session(graph=self.graph)
        with self.graph.as_default():
            initialize_uninitialized(self.session)

        # Load saved network
        self.load_file = load_file
        if self.load_file is not None:
            self.load(self.load_file)

    #########################################
    #           Public interface            #
    #########################################

    def train(self, epoch_num, batch_size, learning_rate, training_data, validation_data,
              beta_fun=lambda x: 0.001, test_step=None):
        """
        Trains the network.
        Parameters:
        epoch_num (int): number of training epochs
        batch_size (int), learning_rate (float): self-explanatory
        training_data, validation_data (list): format as in data_generator
        reg_constant (float, optional): constant for regularization
        beta_fun: gives the beta as a function of the epoch number
        test_step (int, optional): network is tested on validation data after this number of epochs and tensorboard summaries are written
        """
        with self.graph.as_default():
            initialize_uninitialized(self.session)

            for epoch_iter in tqdm_notebook(range(epoch_num)):
                self.tot_epochs += 1
                current_beta = beta_fun(self.tot_epochs)

                if test_step is not None and self.tot_epochs > 0 and self.tot_epochs % test_step == 0:
                    self.test(validation_data, beta=current_beta)

                for step, data_dict in enumerate(self.gen_batch(training_data, batch_size)):

                    parameter_dict = {self.learning_rate: learning_rate, self.beta: current_beta}
                    feed_dict = dict(data_dict, **parameter_dict)

                    self.session.run(self.training_op, feed_dict=feed_dict)

    def test(self, data, beta=0):
        """
        Test accuracy of neural network by comparing mean of output distribution to actual values.
        Parameters:
        data (list, same format as training data): Dataset used to determine accuracy
        """
        with self.graph.as_default():
            data_dict = self.gen_data_dict(data, random_epsilon=False)
            parameter_dict = {self.beta: beta}
            summary = self.session.run(self.all_summaries, feed_dict=dict(data_dict, **parameter_dict))
            self.summary_writer.add_summary(summary, global_step=self.tot_epochs)

    def run(self, data, layer, random_epsilon=False, additional_params={}):
        """
        Run the network and output return the result.
        Params:
        data: Data used for running the network. Same format as training data
        layer: Specifies the layer that is run. If none, then the latent means will be used.
        random_epsilon (bool): If True, the network will be run with noise injection, otherwise without
        """
        with self.graph.as_default():
            data_dict = self.gen_data_dict(data, random_epsilon)
            return self.session.run(layer, feed_dict=dict(data_dict, **additional_params))

    def save(self, file_name):
        """
        Saves state variables (weights, biases) of neural network
        Params:
        file_name (str): model is saved in folder tf_save as file_name.ckpt
        """
        with self.graph.as_default():
            saver = tf.train.Saver()
            saver.save(self.session, io.tf_save_path + file_name + '.ckpt')
            params = {'latent_size': self.latent_size,
                      'input_size': self.input_size,
                      'input2_size': self.input2_size,
                      'output_size': self.output_size,
                      'encoder_num_units': self.encoder_num_units,
                      'decoder_num_units': self.decoder_num_units,
                      'tot_epochs': self.tot_epochs,
                      'name': self.name}
            with open(io.tf_save_path + file_name + '.pkl', 'wb') as f:
                pickle.dump(params, f)
            print "Saved network to file " + file_name

    @classmethod
    def from_saved(cls, file_name, change_params={}):
        """
        Initializes a new network from saved data.
        file_name (str): model is loaded from tf_save/file_name.ckpt
        """
        with open(io.tf_save_path + file_name + '.pkl', 'rb') as f:
            params = pickle.load(f)
        params['load_file'] = file_name
        for p in change_params:
            params[p] = change_params[p]
        print params
        return cls(**params)

    #########################################
    #        Private helper functions       #
    #########################################

    def graph_setup(self):
        """
        Set up the computation graph for the neural network based on the parameters set at initialization
        """
        with self.graph.as_default():

            #######################
            # Define placeholders #
            #######################
            self.input = tf.placeholder(tf.float32, [None, self.input_size], name='input')
            self.epsilon = tf.placeholder(tf.float32, [None, self.latent_size], name='epsilon')
            self.learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')
            self.beta = tf.placeholder(tf.float32, shape=[], name='beta')
            self.input2 = tf.placeholder(tf.float32, shape=[None, self.input2_size], name='input2')
            self.labels = tf.placeholder(tf.float32, shape=[None, self.output_size], name='labels')

            ##########################################
            # Set up variables and computation graph #
            ##########################################
            with tf.variable_scope('encoder'):
                # input and output dimensions for each of the weight tensors
                enc_in_dims = [self.input_size] + self.encoder_num_units
                enc_out_dims = self.encoder_num_units + [2 * self.latent_size]
                temp_layer = self.input
                for k in range(len(enc_in_dims)):
                    with tf.variable_scope('{}th_enc_layer'.format(k)):
                        w = tf.get_variable('w', [enc_in_dims[k], enc_out_dims[k]],
                                            initializer=tf.initializers.random_normal(stddev=2. / np.sqrt(enc_in_dims[k] + enc_out_dims[k])))
                        b = tf.get_variable('b', [enc_out_dims[k]],
                                            initializer=tf.initializers.constant(0.))
                        squash = ((k + 1) != len(enc_in_dims))  # don't squash latent layer
                        temp_layer = forwardprop(temp_layer, w, b, name='enc_layer_{}'.format(k), squash=squash)

            with tf.name_scope('latent_layer'):
                self.log_sigma = temp_layer[:, :self.latent_size]
                self.mu = temp_layer[:, self.latent_size:]
                self.mu_sample = tf.add(self.mu, tf.exp(self.log_sigma) * self.epsilon, name='add_noise')
                self.mu_with_input2 = tf.concat([self.mu_sample, self.input2], axis=1)

            with tf.name_scope('kl_loss'):
                self.kl_loss = kl_divergence(self.mu, self.log_sigma, dim=self.latent_size)

            with tf.variable_scope('decoder'):
                temp_layer = self.mu_with_input2

                dec_in_dims = [self.latent_size + self.input2_size] + self.decoder_num_units
                dec_out_dims = self.decoder_num_units + [self.output_size]
                for k in range(len(dec_in_dims)):
                    with tf.variable_scope('{}th_dec_layer'.format(k)):
                        w = tf.get_variable('w', [dec_in_dims[k], dec_out_dims[k]],
                                            initializer=tf.initializers.random_normal(stddev=2. / np.sqrt(dec_in_dims[k] + dec_out_dims[k])))
                        b = tf.get_variable('b', [dec_out_dims[k]],
                                            initializer=tf.initializers.constant(0.))
                        squash = ((k + 1) != len(dec_in_dims))  # don't squash latent layer
                        temp_layer = forwardprop(temp_layer, w, b, name='dec_layer_{}'.format(k), squash=squash)

                self.output = temp_layer

            with tf.name_scope('recon_loss'):
                self.recon_loss = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(self.labels, self.output), axis=1))

            #####################
            # Cost and training #
            #####################
            with tf.name_scope('cost'):
                self.cost = self.recon_loss + self.beta * self.kl_loss
            with tf.name_scope('optimizer'):
                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                gvs = optimizer.compute_gradients(self.cost)
                capped_gvs = [(tf.clip_by_value(grad, -10., 10.), var) for grad, var in gvs]
                self.training_op = optimizer.apply_gradients(capped_gvs)

            #########################
            # Tensorboard summaries #
            #########################
            tf.summary.histogram('latent_means', self.mu)
            tf.summary.histogram('latent_log_sigma', self.log_sigma)
            tf.summary.histogram('ouput_means', self.output)
            tf.summary.scalar('recon_loss', self.recon_loss)
            tf.summary.scalar('kl_loss', self.kl_loss)
            tf.summary.scalar('cost', self.cost)
            tf.summary.scalar('beta', self.beta)

            self.summary_writer = tf.summary.FileWriter(io.tf_log_path + self.name + '/', graph=self.graph)
            self.summary_writer.flush()  # write out graph
            self.all_summaries = tf.summary.merge_all()

    def gen_batch(self, data, batch_size, shuffle=True, random_epsilon=True):
        """
        Generate batches for training the network.
        Params:
        data: same format as training data (see Data_loader)
        batch_size (int)
        shuffle (bool): if true, data is shuffled before batches are created
        random_epsilon (bool): if true, epsilon is drawn from a normal distribution; otherwise, epsilon=0
        """
        epoch_size = len(data[0]) / batch_size
        if shuffle:
            p = np.random.permutation(len(data[0]))
            data = [data[i][p] for i in [0, 1, 2]]
        for i in range(epoch_size):
            batch_slice = slice(i * batch_size, (i + 1) * batch_size)
            batch = [data[j][batch_slice] for j in [0, 1, 2]]
            yield self.gen_data_dict(batch, random_epsilon=random_epsilon)

    def gen_data_dict(self, data, random_epsilon=True):
        """
        Params:
        data: same format as training data (see data_loader)
        random_epsilon (bool): if true, epsilon is drawn from a normal distribution; otherwise, epsilon=0
        """
        if random_epsilon is True:
            eps = np.random.normal(size=[len(data[0]), self.latent_size])
        else:
            eps = np.zeros([len(data[0]), self.latent_size])
        return {self.input: data[0],
                self.input2: data[1],
                self.labels: data[2],
                self.epsilon: eps}

    def load(self, file_name):
        """ 
        Loads network, params as in save 
        """
        with self.graph.as_default():
            saver = tf.train.Saver()
            saver.restore(self.session, io.tf_save_path + file_name + '.ckpt')
            print "Loaded network from file " + file_name


###########
# Helpers #
###########


def forwardprop(x, w, b, squash=True, act_fun=tf.nn.elu, name=''):
    """
    Forward-propagation.
    """
    if name != '':
        name = '_' + name
    pre_act = tf.add(tf.matmul(x, w, name=('w_mul' + name)), b, name=('b_add' + name))
    if name != '':
        tf.summary.histogram('pre-act' + name, pre_act)
    if squash:
        return act_fun(pre_act, name=('act_fun' + name))
    else:
        return pre_act


def initialize_uninitialized(sess):
    global_vars = tf.global_variables()
    is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))


def kl_divergence(means, log_sigma, dim, target_sigma=0.1):
    # KL divergence between given distribution and unit Gaussian
    target_sigma = tf.constant(target_sigma, shape=[dim])
    return 1 / 2. * tf.reduce_mean(tf.reduce_sum(1 / target_sigma**2 * means**2 +
                                                 tf.exp(2 * log_sigma) / target_sigma**2 - 2 * log_sigma + 2 * tf.log(target_sigma), axis=1) - dim)
