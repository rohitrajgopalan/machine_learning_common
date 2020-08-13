from reinforcement_learning.agent.agent import Agent
from reinforcement_learning.noise.ou_noise import OUNoise
import tensorflow as tf
from reinforcement_learning.network.ddpg_actor_network import DDPGActorNetwork
from reinforcement_learning.network.ddpg_critic_network import DDPGCriticNetwork
import numpy as np


class DDPGAgent(Agent):
    actor_network = None
    critic_network = None
    exploration_noise = None
    enable_noise = False
    action_min = None
    action_max = None

    def exploration_noise_init(self, exploration_noise_args):
        if not self.enable_noise:
            pass
        mu = exploration_noise_args['mu'] if 'mu' in exploration_noise_args else 0
        theta = exploration_noise_args['theta'] if 'theta' in exploration_noise_args else 0.15
        sigma = exploration_noise_args['sigma'] if 'sigma' in exploration_noise_args else 0.2
        self.exploration_noise = OUNoise(self.action_dim, mu, theta, sigma)

    def actor_network_init(self, network_args):
        layer_sizes = network_args['layer_sizes']
        optimizer_type = network_args['optimizer_type']
        optimizer_args = network_args['optimizer_args']
        tau = network_args['tau']
        conv_layer_info_list = network_args['conv_layer_info_list'] if 'conv_layer_info_list' in network_args else []
        add_pooling = network_args['add_pooling'] if 'add_pooling' in network_args else False
        convert_to_grayscale = network_args['convert_to_grayscale'] if 'convert_to_grayscale' in network_args else False
        self.actor_network = DDPGActorNetwork(self.state_dim, self.action_dim, layer_sizes, optimizer_type, optimizer_args, tau, conv_layer_info_list, add_pooling, convert_to_grayscale)

    def critic_network_init(self, network_args):
        state_layer_sizes = network_args['state_layer_sizes']
        action_layer_size = network_args['action_layer_size']
        network_layer_sizes = network_args['network_layer_sizes']
        optimizer_type = network_args['optimizer_type']
        optimizer_args = network_args['optimizer_args']
        tau = network_args['tau']
        conv_layer_info_list = network_args['conv_layer_info_list'] if 'conv_layer_info_list' in network_args else []
        add_pooling = network_args['add_pooling'] if 'add_pooling' in network_args else False
        convert_to_grayscale = network_args['convert_to_grayscale'] if 'convert_to_grayscale' in network_args else False
        self.critic_network = DDPGCriticNetwork(self.state_dim, state_layer_sizes, self.action_dim, action_layer_size, network_layer_sizes, optimizer_type, optimizer_args, tau, conv_layer_info_list, add_pooling, convert_to_grayscale)

    def step(self, r1, r2, should_action_be_blocked=False):
        super().step(r1, r2, should_action_be_blocked)
        if not self.active:
            self.exploration_noise.reset()

    def optimize_network(self, experiences):
        if self.flatten_state:
            state_shape = [len(experiences)]
            for s in self.state_dim:
                state_shape.append(s)
            state_shape = tuple(state_shape)
        else:
            state_shape = (len(experiences), self.state_dim)
        states = np.zeros(state_shape)
        next_states = np.zeros(state_shape)
        actions = np.zeros((len(experiences), self.action_dim))
        rewards = np.zeros((len(experiences)))
        terminals = np.zeros((len(experiences)))
        for batch_idx, experience in enumerate(experiences):
            s, a, s_, r, terminal = experience
            states[batch_idx] = s if self.state_type == np.ndarray else np.array([s])
            action = self.actions[a]
            actions[batch_idx] = action if self.action_type == np.ndarray else np.array([action])
            next_states[batch_idx] = s_ if self.state_type == np.ndarray else np.array([s_])
            rewards[batch_idx] = r
            terminals[batch_idx] = terminals

        states = tf.convert_to_tensor(states)
        actions = tf.convert_to_tensor(actions)
        rewards = tf.convert_to_tensor(rewards)
        rewards = tf.cast(rewards, dtype=tf.float32)
        terminals = tf.convert_to_tensor(terminals)
        terminals = tf.cast(terminals, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states)

        with tf.GradientTape() as tape:
            target_actions = self.actor_network.target_actions(next_states)
            y = rewards + (self.discount_factor * self.critic_network.target_qs(next_states, target_actions) * terminals)
            critic_value = self.critic_network.q_values(states, actions)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_gradients = tape.gradient(critic_loss, self.critic_network.model.trainable_variables)
        self.critic_network.apply_gradients(critic_gradients)

        with tf.GradientTape() as tape:
            actions = self.actor_network.actions(states)
            critic_value = self.critic_network.q_values(states, actions)
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_gradients = tape.gradient(actor_loss, self.actor_network.model.trainable_variables)
        self.actor_network.apply_gradients(actor_gradients)
        self.actor_network.update_target()
        self.critic_network.update_target()

    def calculate_target_value(self, s_, r, active):
        a_ = self.actor_network.target_action(s_)
        target_q = self.critic_network.target_q(s_, a_)
        return r + (self.discount_factor * target_q * active)

    def get_target_error(self, s, a, s_, r, active):
        return self.calculate_target_value(s_, r, active) - self.critic_network.q_value(s, self.actions[a])

    def assign_initial_action(self):
        action = self.actor_network.action(self.current_state)
        if self.enable_noise:
            action += self.exploration_noise.noise()
        action = np.clip(action, self.action_min, self.action_max)
        self.initial_action = self.add_action(action)

    def get_results(self):
        self.actions = []
        action_counter = 0
        total_reward = 0
        final_policy = {}
        for state in self.experienced_states:
            action = self.actor_network.action(state)
            action_counter += 1
            total_reward += self.critic_network.q_value(state, action)
            self.actions.append(action)
            if self.flatten_state:
                state = state.flatten()
            if type(state) == np.ndarray:
                state = tuple(state)
            final_policy[state] = [action_counter-1]
        return total_reward, final_policy
