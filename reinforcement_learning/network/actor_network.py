import math
import numpy as np
import tensorflow as tf

# Hyper Parameters
from neural_network.network_types import NetworkOptimizer

LAYER1_SIZE = 400
LAYER2_SIZE = 300
LEARNING_RATE = 1e-4
TAU = 0.001
BATCH_SIZE = 64


class ActorNetwork:
    """
    Reference: https://github.com/maxkferg/DDPG/blob/master/ddpg/actor_network.py
    """

    def __init__(self, sess, state_dim, action_dim, layer_sizes=[], optimizer_args={}, tau=TAU):
        tf.compat.v1.disable_eager_execution()
        tf.compat.v1.enable_resource_variables()
        tf.compat.v1.enable_v2_tensorshape()
        tf.compat.v1.enable_control_flow_v2()
        self.sess = sess
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.tau = tau
        # create actor network
        self.state_input, self.action_output, self.weights = self.create_network(state_dim, action_dim, layer_sizes)

        # create target actor network
        self.target_state_input, self.target_action_output, self.target_update, self.target_weights = self.create_target_network(
            state_dim, self.weights)

        self.net = []
        for weight in self.weights:
            self.net.append(weight['w'])
            self.net.append(weight['b'])

        # define training rules
        self.q_gradient_input = tf.compat.v1.placeholder("float", [None, self.action_dim])
        self.parameters_gradients = tf.gradients(self.action_output, self.net, -self.q_gradient_input)
        learning_rate = optimizer_args['learning_rate'] if 'learning_rate' in optimizer_args else 0.001
        beta_m = optimizer_args['beta_m'] if 'beta_m' in optimizer_args else 0.9
        beta_v = optimizer_args['beta_v'] if 'beta_v' in optimizer_args else 0.999
        epsilon = optimizer_args['epsilon'] if 'epsilon' in optimizer_args else 1e-07

        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta_m, beta2=beta_v, epsilon=epsilon).apply_gradients(
            zip(self.parameters_gradients, self.net))

        self.sess.run(tf.compat.v1.global_variables_initializer())

        self.update_target()

    def create_network(self, state_dim, action_dim, layer_sizes):
        state_input = tf.compat.v1.placeholder("float", [None, state_dim])

        for i, layer_size in enumerate(layer_sizes):
            if layer_size == 'auto':
                layer_sizes[i] = int(math.sqrt(state_dim * action_dim))

        weights = [dict() for _ in range(len(layer_sizes) + 1)]
        for i, weight in enumerate(weights):
            if i == 0:
                weight.update({'w': self.variable([state_dim, layer_sizes[0]], state_dim),
                               'b': self.variable([layer_sizes[0]], state_dim)})
            elif i == len(weights)-1:
                weight.update(
                    {'w': tf.Variable(tf.compat.v1.random_uniform([layer_sizes[len(layer_sizes) - 1], action_dim], -3e-3, 3e-3)),
                     'b': tf.Variable(tf.compat.v1.random_uniform([action_dim], -3e-3, 3e-3))})
            else:
                weight.update({'w': self.variable([layer_sizes[i - 1], layer_sizes[i]], layer_sizes[i - 1]),
                               'b': self.variable([layer_sizes[i]], layer_sizes[i - 1])})

        layer = None
        action_output = None
        for i, weight in enumerate(weights):
            if i == 0:
                layer = tf.nn.relu(tf.matmul(state_input, weight['w']) + weight['b'])
            elif i == len(weights)-1:
                action_output = tf.tanh(tf.matmul(layer, weight['w']) + weight['b'])
            else:
                layer = tf.nn.relu(tf.matmul(layer, weight['w']) + weight['b'])

        return state_input, action_output, weights

    def create_target_network(self, state_dim, weights):
        state_input = tf.compat.v1.placeholder("float", [None, state_dim])
        ema = tf.train.ExponentialMovingAverage(decay=1-self.tau)

        net = []
        for weight in weights:
            net.append(weight['w'])
            net.append(weight['b'])
        target_update = ema.apply(net)
        target_weights = [dict() for _ in range(len(weights))]
        for i, weight in enumerate(target_weights):
            weight.update({'w': ema.average(weights[i]['w']),
                           'b': ema.average(weights[i]['b'])})

        layer = None
        action_output = None
        for i, weight in enumerate(target_weights):
            if i == 0:
                layer = tf.nn.relu(tf.matmul(state_input, weight['w']) + weight['b'])
            elif i == len(weights)-1:
                action_output = tf.tanh(tf.matmul(layer, weight['w']) + weight['b'])
            else:
                layer = tf.nn.relu(tf.matmul(layer, weight['w']) + weight['b'])

        return state_input, action_output, target_update, target_weights

    def update_target(self):
        self.sess.run(self.target_update)

    def train(self, q_gradient_batch, state_batch):
        self.sess.run(self.optimizer, feed_dict={
            self.q_gradient_input: q_gradient_batch,
            self.state_input: state_batch
        })

    def actions(self, state_batch):
        return self.sess.run(self.action_output, feed_dict={
            self.state_input: state_batch
        })

    def actions_for_state(self, state):
        return self.sess.run(self.action_output, feed_dict={
            self.state_input: [state if type(state) == np.ndarray else np.array([state])]
        })

    def action(self, state):
        return self.actions_for_state(state)[0]

    def target_actions(self, state_batch):
        return self.sess.run(self.target_action_output, feed_dict={
            self.target_state_input: state_batch
        })

    def target_actions_for_state(self, state):
        return self.sess.run(self.target_action_output, feed_dict={
            self.target_state_input: [state if type(state) == np.ndarray else np.array([state])]
        })

    def target_action(self, state):
        return self.target_actions_for_state(state)[0]

    # f fan-in size
    def variable(self, shape, f):
        return tf.Variable(tf.compat.v1.random_uniform(shape, -1 / math.sqrt(f), 1 / math.sqrt(f)))
