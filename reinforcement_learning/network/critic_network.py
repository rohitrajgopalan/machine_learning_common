import math

import tensorflow as tf

LAYER1_SIZE = 400
LAYER2_SIZE = 300
LEARNING_RATE = 1e-3
TAU = 0.001
L2 = 0.01


class CriticNetwork:
    """
    Reference: https://github.com/maxkferg/DDPG/blob/master/ddpg/critic_network.py
    """

    def __init__(self, sess, state_dim, action_dim, l2_value=L2, layer_sizes=[], optimizer_args={}, tau=TAU):
        tf.compat.v1.disable_eager_execution()
        tf.compat.v1.enable_resource_variables()
        tf.compat.v1.enable_v2_tensorshape()
        tf.compat.v1.enable_control_flow_v2()
        self.time_step = 0
        self.sess = sess
        self.tau = tau
        # create q network
        self.state_input, self.action_input, self.q_value_output, self.weights = self.create_q_network(state_dim,
                                                                                                       action_dim, layer_sizes)

        # create target q network (the same structure with q network)
        self.target_state_input, self.target_action_input, self.target_q_value_output, self.target_update = self.create_target_q_network(
            state_dim, action_dim, self.weights)

        self.net = []
        for weight in self.weights:
            self.net.append(weight['w'])
            self.net.append(weight['b'])

        self.y_input = tf.compat.v1.placeholder("float", [None, 1])
        weight_decay = tf.add_n([l2_value * tf.nn.l2_loss(var) for var in self.net])
        self.cost = tf.reduce_mean(tf.square(self.y_input - self.q_value_output)) + weight_decay
        learning_rate = optimizer_args['learning_rate'] if 'learning_rate' in optimizer_args else 0.001
        beta_m = optimizer_args['beta_m'] if 'beta_m' in optimizer_args else 0.9
        beta_v = optimizer_args['beta_v'] if 'beta_v' in optimizer_args else 0.999
        epsilon = optimizer_args['epsilon'] if 'epsilon' in optimizer_args else 1e-07

        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta_m, beta2=beta_v, epsilon=epsilon).minimize(self.cost)
        self.action_gradients = tf.gradients(self.q_value_output, self.action_input)

        # initialization
        self.sess.run(tf.compat.v1.global_variables_initializer())

        self.update_target()

    def create_q_network(self, state_dim, action_dim, layer_sizes):
        state_input = tf.compat.v1.placeholder("float", [None, state_dim])
        action_input = tf.compat.v1.placeholder("float", [None, action_dim])

        for i, layer_size in enumerate(layer_sizes):
            if layer_size == 'auto':
                layer_sizes[i] = int(math.sqrt(state_dim + action_dim))

        weights = [dict() for _ in range(len(layer_sizes) + 1)]
        for i, weight in enumerate(weights):
            if i == 0:
                weight.update({'w': self.variable([state_dim, layer_sizes[0]], state_dim),
                               'b': self.variable([layer_sizes[0]], state_dim)})
            elif i == len(weights) - 1:
                weight.update(
                    {'w': tf.Variable(tf.compat.v1.random_uniform([layer_sizes[len(layer_sizes) - 1], 1], -3e-3, 3e-3)),
                     'b': tf.Variable(tf.compat.v1.random_uniform([1], -3e-3, 3e-3))})
            elif i == len(layer_sizes) - 1:
                weight.update({'w': self.variable([layer_sizes[len(layer_sizes) - 2] if len(layer_sizes) >= 2 else state_dim, layer_sizes[len(layer_sizes) - 1]],
                                                  (layer_sizes[len(layer_sizes) - 2] if len(layer_sizes) >= 2 else state_dim) + action_dim),
                               'w_action': self.variable([action_dim, layer_sizes[len(layer_sizes) - 1]], (
                                   layer_sizes[len(layer_sizes) - 2] if len(layer_sizes) >= 2 else state_dim) + action_dim),
                               'b': self.variable([layer_sizes[len(layer_sizes) - 1]],
                                                  (layer_sizes[len(layer_sizes) - 2] if len(layer_sizes) >= 2 else state_dim) + action_dim)})
            else:
                weight.update({'w': self.variable([layer_sizes[i - 1], layer_sizes[i]], layer_sizes[i - 1]),
                               'b': self.variable([layer_sizes[i]], layer_sizes[i - 1])})

        layer = None
        q_value_output = None
        for i, weight in enumerate(weights):
            if i == 0:
                layer = tf.nn.relu(tf.matmul(state_input, weight['w']) + weight['b'])
            elif i == len(weights) - 1:
                q_value_output = tf.identity(tf.matmul(layer, weight['w']) + weight['b'])
            elif i == len(layer_sizes) - 1:
                layer = tf.nn.relu(
                    tf.matmul(layer, weight['w']) + tf.matmul(action_input, weight['w_action']) + weight['b'])
            else:
                layer = tf.nn.relu(tf.matmul(layer, weight['w']) + weight['b'])

        return state_input, action_input, q_value_output, weights

    def create_target_q_network(self, state_dim, action_dim, weights):
        state_input = tf.compat.v1.placeholder("float", [None, state_dim])
        action_input = tf.compat.v1.placeholder("float", [None, action_dim])

        ema = tf.train.ExponentialMovingAverage(decay=1 - self.tau)

        net = []
        for weight in weights:
            net.append(weight['w'])
            if 'w_action' in weight:
                net.append(weight['w_action'])
            net.append(weight['b'])

        target_update = ema.apply(net)
        target_weights = [dict() for _ in range(len(weights))]
        for i, weight in enumerate(target_weights):
            weight.update({'w': ema.average(weights[i]['w']),
                           'b': ema.average(weights[i]['b'])})
            if 'w_action' in weights[i]:
                weight.update({'w_action': ema.average(weights[i]['w_action'])})

        layer = None
        q_value_output = None
        for i, weight in enumerate(target_weights):
            if i == 0:
                layer = tf.nn.relu(tf.matmul(state_input, weight['w']) + weight['b'])
            elif i == len(target_weights)-1:
                q_value_output = tf.identity(tf.matmul(layer, weight['w']) + weight['b'])
            elif 'w_action' in weight:
                layer = tf.nn.relu(tf.matmul(layer, weight['w']) + tf.matmul(action_input, weight['w_action']) + weight['b'])
            else:
                layer = tf.nn.relu(tf.matmul(layer, weight['w']) + weight['b'])

        return state_input, action_input, q_value_output, target_update

    def update_target(self):
        self.sess.run(self.target_update)

    def train(self, y_batch, state_batch, action_batch):
        self.time_step += 1
        self.sess.run(self.optimizer, feed_dict={
            self.y_input: y_batch,
            self.state_input: state_batch,
            self.action_input: action_batch
        })

    def gradients(self, state_batch, action_batch):
        return self.sess.run(self.action_gradients, feed_dict={
            self.state_input: state_batch,
            self.action_input: action_batch
        })[0]

    def target_qs(self, state_batch, action_batch):
        return self.sess.run(self.target_q_value_output, feed_dict={
            self.target_state_input: state_batch,
            self.target_action_input: action_batch
        })

    def target_q(self, state, action):
        return self.sess.run(self.target_q_value_output, feed_dict={
            self.target_state_input: [state],
            self.target_action_input: [action]
        })[0, 0]

    def q_values(self, state_batch, action_batch):
        return self.sess.run(self.q_value_output, feed_dict={
            self.state_input: state_batch,
            self.action_input: action_batch})

    def q_value(self, state, action):
        return self.sess.run(self.q_value_output, feed_dict={
            self.state_input: [state],
            self.action_input: [action]
        })[0, 0]

    # f fan-in size
    def variable(self, shape, f):
        return tf.Variable(tf.compat.v1.random_uniform(shape, -1 / math.sqrt(f), 1 / math.sqrt(f)))
