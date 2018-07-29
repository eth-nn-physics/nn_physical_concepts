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

    def __init__(self, latent_size, input_size, time_series_length, output_size,
                 encoder_num_units=[100, 100], decoder_num_units=[100, 100], euler_num_units=[], name='Unnamed',
                 tot_epochs=0, load_file=None):
        """
        Parameters:
        input_size: number of time steps used for initial input into the network.
        latent_size: number of latent neurons to be used.
        time_series_length: number of time steps (although each time step can contain mutliple values).
        output_size: number of values in each time step (e.g. 2 if each time step is a vector in R^2).
        encoder_num_units, decoder_num_units: Number of neurons in encoder and decoder hidden layers. Everything is fully connected.
        name: Used for tensorboard
        tot_epochs and  load_file are used internally for loading and saving, don't pass anything to them manually.
        """

        self.graph = tf.Graph()

        self.input_size = input_size
        self.latent_size = latent_size
        self.encoder_num_units = encoder_num_units
        self.decoder_num_units = decoder_num_units
        self.name = name
        self.tot_epochs = tot_epochs
        self.euler_num_units = euler_num_units
        self.output_size = output_size
        self.time_series_length = time_series_length
        self.rnn_depth = time_series_length - input_size

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
              beta_fun=lambda x: 0.001, euler_l2_coeff=1.e-5, test_step=None):
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
                    parameter_dict = {self.learning_rate: learning_rate, self.beta: current_beta, self.euler_l2_coeff: euler_l2_coeff}
                    feed_dict = dict(data_dict, **parameter_dict)
                    self.session.run(self.training_op, feed_dict=feed_dict)

    def test(self, data, beta=0, l2_coeff=0):
        """
        Test accuracy of neural network by comparing mean of output distribution to actual values.
        Parameters:
        data (list, same format as training data): Dataset used to determine accuracy
        """
        with self.graph.as_default():
            data_dict = self.gen_data_dict(data, random_epsilon=False)
            parameter_dict = {self.beta: beta, self.euler_l2_coeff: l2_coeff}
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
                      'encoder_num_units': self.encoder_num_units,
                      'decoder_num_units': self.decoder_num_units,
                      'tot_epochs': self.tot_epochs,
                      'name': self.name,
                      'time_series_length': self.time_series_length,
                      'euler_num_units': self.euler_num_units,
                      'output_size': self.output_size}
            with open(io.tf_save_path + file_name + '.pkl', 'wb') as f:
                pickle.dump(params, f)
            print "Saved network to file " + file_name

    #########################################
    #        Public helper functions        #
    #########################################

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

    def recon_loss_fun(self, prediction, euler_index):
        # the full time series goes in strides of output_size (each observation contains output_size data points)
        ind = self.output_size * self.input_size + self.output_size * (euler_index - 1)
        observation = self.full_time_series[:, ind: ind + self.output_size]
        return tf.squared_difference(prediction, observation)

    def graph_setup(self):
        """
        Set up the computation graph for the neural network based on the parameters set at initialization
        """
        with self.graph.as_default():

            #######################
            # Define placeholders #
            #######################
            self.full_time_series = tf.placeholder(tf.float32, [None, self.output_size * self.time_series_length], name='full_time_series')
            self.epsilon = tf.placeholder(tf.float32, [None, self.latent_size], name='epsilon')
            self.learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')
            self.beta = tf.placeholder(tf.float32, shape=[], name='beta')
            self.euler_l2_coeff = tf.placeholder(tf.float32, shape=[], name='euler_l2_coeff')

            ##################
            # Set up encoder #
            ##################
            with tf.name_scope('prepare_in1'):
                self.in1 = self.full_time_series[:, :self.output_size * self.input_size]
            # input and output dimensions for each of the weight tensors
            enc_in_num = [self.output_size * self.input_size] + self.encoder_num_units
            enc_out_num = self.encoder_num_units + [2 * self.latent_size]

            encoder_input = self.in1
            with tf.variable_scope('dynamic_encoder'):
                previous_enc_layer = encoder_input
                for k in range(len(enc_out_num)):
                    with tf.variable_scope('{}th_enc_layer'.format(k)):
                        w = tf.get_variable('w_enc{}'.format(k),
                                            [enc_in_num[k], enc_out_num[k]],
                                            initializer=tf.glorot_normal_initializer())
                        b = tf.get_variable('b_enc{}'.format(k),
                                            [enc_out_num[k]],
                                            initializer=tf.random_normal_initializer())
                        # create next layer
                        squash = (k != (len(enc_out_num) - 1))
                        previous_enc_layer = forwardprop(previous_enc_layer, w, b, squash=squash, name='{}th_enc_layer'.format(k))

            with tf.name_scope('dynamic_state'):
                pre_state = previous_enc_layer
                self.state_means = tf.nn.tanh(pre_state[:, :self.latent_size])
                self.state_log_sigma = tf.clip_by_value(pre_state[:, self.latent_size:], -5., 0.5)
                self.state_log_sigma = pre_state[:, self.latent_size:]
            with tf.name_scope('state_sample'):
                self.state_sample = tf.add(self.state_means, tf.exp(self.state_log_sigma) * self.epsilon, name='add_noise')
            with tf.name_scope('kl_loss'):
                self.kl_loss = kl_divergence(self.state_means, self.state_log_sigma, self.latent_size)

            ###################################
            # Set up variables for Euler step #
            ###################################
            in_euler = [self.latent_size] + self.euler_num_units
            out_euler = self.euler_num_units + [self.latent_size]
            with tf.variable_scope('RNN'):

                ###################
                # Prepare decoder #
                ###################
                dec_in_num = [self.latent_size] + self.decoder_num_units
                dec_out_num = self.decoder_num_units + [self.output_size]
                with tf.variable_scope('decoder_vars'):
                    self.dec_weights = []
                    self.dec_biases = []
                    self.decoder_l2_loss = tf.constant(0.)
                    for k in range(len(dec_out_num)):
                        self.dec_weights.append(tf.get_variable('w_dec{}'.format(k),
                                                                [dec_in_num[k], dec_out_num[k]],
                                                                initializer=tf.glorot_normal_initializer()))

                        self.dec_biases.append(tf.get_variable('b_dec{}'.format(k),
                                                               [dec_out_num[k]],
                                                               initializer=tf.random_normal_initializer()))
                        self.decoder_l2_loss = self.decoder_l2_loss + tf.nn.l2_loss(self.dec_weights[-1]) + tf.nn.l2_loss(self.dec_biases[-1])

                    def decoder_net(latent_state):
                        temp_state = latent_state
                        for k, (w, b) in enumerate(zip(self.dec_weights, self.dec_biases)):
                            squash = ((k + 1) != len(self.dec_weights))  # don't squash last layer
                            temp_state = forwardprop(temp_state, w, b, name='{}th_dec_layer'.format(k), squash=squash)
                        return temp_state

                with tf.variable_scope('euler_vars'):
                    self.euler_weights = [
                        tf.get_variable('w_euler{}'.format(k),
                                        [in_euler[k], out_euler[k]],
                                        initializer=tf.glorot_normal_initializer())
                        for k in range(len(out_euler))
                    ]
                    self.euler_biases = [
                        tf.get_variable('b_euler{}'.format(k),
                                        [out_euler[k]],
                                        initializer=tf.random_normal_initializer())
                        for k in range(len(out_euler))
                    ]

                with tf.name_scope('euler_l2_loss'):
                    self.euler_l2_loss = tf.add_n([tf.nn.l2_loss(self.euler_weights[i]) for i in range(len(out_euler))])

                ###########################################
                # Define computation graph for Euler step #
                ###########################################
                self.latent_vector_list = [self.state_sample]
                with tf.name_scope('initial_euler_loss'):
                    self.decoded_list = [decoder_net(self.state_sample)]
                    recon_losses_list = [self.recon_loss_fun(self.decoded_list[-1], 0)]

                for s in range(self.rnn_depth):
                    with tf.name_scope('{}th_euler_step'.format(s + 1)):
                        temp_state = self.latent_vector_list[-1]
                        for j, (w, b) in enumerate(zip(self.euler_weights, self.euler_biases)):
                            # To use the Euler weights, replace this line by
                            # temp_state = my_activation_function(tf.matmul(temp_state, w) + b)
                            temp_state = temp_state + b
                        self.latent_vector_list.append(temp_state)
                    with tf.name_scope('decode_{}th_euler_step'.format(s + 1)):
                        self.decoded_list.append(decoder_net(temp_state))
                        recon_losses_list.append(self.recon_loss_fun(self.decoded_list[-1], s + 1))

                with tf.name_scope('gather_recon_losses'):
                    self.recon_loss = tf.reduce_mean(tf.stack(recon_losses_list))

            #####################
            # Cost and training #
            #####################
            with tf.name_scope('cost'):
                self.cost = tf.add_n([self.recon_loss,
                                      self.beta * self.kl_loss,
                                      self.euler_l2_coeff * self.euler_l2_loss], name='add_costs')
            with tf.name_scope('optimizer'):
                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                gvs = optimizer.compute_gradients(self.cost)
                capped_gvs = [(tf.clip_by_value(grad, -10., 10.), var) for grad, var in gvs]
                self.training_op = optimizer.apply_gradients(capped_gvs)

            #########################
            # Tensorboard summaries #
            #########################
            tf.summary.histogram('state_means', self.state_means)
            tf.summary.histogram('state_log_sigma', self.state_log_sigma)
            for i, (w, b) in enumerate(zip(self.euler_weights, self.euler_biases)):
                tf.summary.histogram('euler_weight_{}'.format(i), w)
                tf.summary.histogram('euler_bias_{}'.format(i), b)
            tf.summary.scalar('cost', self.cost)
            tf.summary.scalar('reconstruction_cost', self.recon_loss)
            tf.summary.scalar('kl_cost', self.kl_loss)
            tf.summary.scalar('euler_l2_loss', self.euler_l2_loss)
            tf.summary.scalar('beta', self.beta)
            tf.summary.scalar('L2_coeff', self.euler_l2_coeff)
            self.summary_writer = tf.summary.FileWriter(io.tf_log_path + self.name + '/', graph=self.graph)
            self.summary_writer.flush()
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
        epoch_size = len(data) / batch_size
        if shuffle:
            p = np.random.permutation(len(data))
            data = data[p]
        for i in range(epoch_size):
            batch_slice = slice(i * batch_size, (i + 1) * batch_size)
            batch = data[batch_slice]
            yield self.gen_data_dict(batch, random_epsilon=random_epsilon)

    def gen_data_dict(self, data, random_epsilon=True):
        """
        Params:
        data: same format as training data (see data_loader)
        random_epsilon (bool): if true, epsilon is drawn from a normal distribution; otherwise, epsilon=0
        """
        if random_epsilon is True:
            eps = np.random.normal(size=[len(data), self.latent_size])
        else:
            eps = np.zeros([len(data), self.latent_size])
        return {self.full_time_series: data,
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
