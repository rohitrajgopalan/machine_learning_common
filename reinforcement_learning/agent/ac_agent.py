from reinforcement_learning.agent.agent import Agent
from reinforcement_learning.noise.ou_noise import OUNoise
import tensorflow as tf
from reinforcement_learning.network.actor_network import ActorNetwork
from reinforcement_learning.network.critic_network import CriticNetwork
import numpy as np


class ACAgent(Agent):
    actor_network = None
    critic_network = None
    exploration_noise = None
    enable_noise = False
    sess = tf.compat.v1.InteractiveSession()

    def __init__(self, args={}):
        super().__init__(args)
        self.reset()

    def exploration_noise_init(self, exploration_noise_args):
        if not self.enable_noise:
            pass
        mu = exploration_noise_args['mu'] if 'mu' in exploration_noise_args else 0
        theta = exploration_noise_args['theta'] if 'theta' in exploration_noise_args else 0.15
        sigma = exploration_noise_args['sigma'] if 'sigma' in exploration_noise_args else 0.2
        self.exploration_noise = OUNoise(self.action_dim, mu, theta, sigma)

    def actor_network_init(self, actor_network_args):
        layer_sizes = actor_network_args['layer_sizes']
        optimizer_args = actor_network_args['optimizer_args']
        tau = actor_network_args['tau']
        self.actor_network = ActorNetwork(self.sess, self.state_dim, self.action_dim, layer_sizes, optimizer_args, tau)

    def critic_network_init(self, critic_network_args):
        l2_loss_val = critic_network_args['l2_loss'] if 'l2_loss' in critic_network_args else 0.01
        layer_sizes = critic_network_args['layer_sizes']
        optimizer_args = critic_network_args['optimizer_args']
        tau = critic_network_args['tau']
        self.critic_network = CriticNetwork(self.sess, self.state_dim, self.action_dim, l2_loss_val, layer_sizes, optimizer_args, tau)

    def step(self, r1, r2, should_action_be_blocked=False):
        super().step(r1, r2, should_action_be_blocked)
        if not self.active:
            self.exploration_noise.reset()

    def add_to_supervised_learning(self, r, should_action_be_blocked):
        blocked_boolean = 1 if should_action_be_blocked else 0
        if self.enable_action_blocking:
            self.action_blocker.add(self.current_state, self.initial_action, blocked_boolean)
        new_data = {}
        if self.state_dim == 1:
            new_data.update({'STATE': self.current_state})
        else:
            for i in range(self.state_dim):
                state_val = self.current_state[i]
                if type(state_val) == bool:
                    state_val = int(state_val)
                new_data.update({'STATE_VAR{0}'.format(i + 1): state_val})
        action = self.actions[self.initial_action]
        action = np.array([action])
        if self.action_dim == 1:
            new_data.update({'INITIAL_ACTION': self.initial_action if type(action) == str else action[0, 0]})
        else:
            for i in range(self.action_dim):
                new_data.update({'INITIAL_ACTION_VAR{0}'.format(i + 1): action[0, i]})

        new_data.update({'REWARD': r,
                         'GAMMA': self.discount_factor,
                         'TARGET_VALUE': self.calculate_target_value(self.next_state, r,
                                                                     int(self.active)),
                         'BLOCKED?': blocked_boolean})
        self.historical_data = self.historical_data.append(new_data, ignore_index=True)

    def optimize_network(self, experiences):
        states = []
        actions = []
        outputs = []
        for experience in experiences:
            s, a, s_, r, terminal = experience
            states.append(s)
            action = self.actions[a]
            actions.append(action)
            target_value = self.calculate_target_value(s_, r, 1 - terminal)
            outputs.append(target_value)

        states = np.asarray(states)
        actions = np.asarray(actions)
        outputs = np.resize(outputs, [self.replay_buffer.minibatch_size, 1])
        self.critic_network.train(outputs, states, actions)
        action_batch_for_gradients = self.actor_network.actions(states)
        q_gradient_batch = self.critic_network.gradients(states, action_batch_for_gradients)
        self.actor_network.train(q_gradient_batch, states)

        self.actor_network.update_target()
        self.critic_network.update_target()

    def calculate_target_value(self, s_, r, active):
        a_ = self.actor_network.target_action(s_)
        target_q = self.critic_network.target_q(s_, a_)
        return r + (self.discount_factor * target_q * active)

    def assign_initial_action(self):
        action = self.actor_network.action(self.current_state)
        if self.enable_noise:
            action += self.exploration_noise.noise()
        self.initial_action = self.add_action(action)

    def get_results(self):
        self.actions = []
        action_counter = 0
        total_reward = 0
        final_policy = {}
        for state in self.experienced_states:
            actions_ = self.actor_network.actions_for_state(state)
            chosen_actions = []
            for action_ in actions_:
                action_counter += 1
                chosen_actions.append(action_counter-1)
                total_reward += self.critic_network.q_value(state, action_)
                self.actions.append(action_)
            if len(chosen_actions) > 0:
                final_policy[tuple(state)] = chosen_actions
        return total_reward, final_policy
