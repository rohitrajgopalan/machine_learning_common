from copy import deepcopy

import pandas as pd
import numpy as np
import enum

from reinforcement_learning.network.actionvaluenetwork import ActionValueNetwork
from reinforcement_learning.optimizer.adam_optimizer import Adam
from reinforcement_learning.replay.replay_buffer import ReplayBuffer


class LearningType(enum.Enum):
    Online = 1
    Replay = 2


class Agent:
    agent_id = 0
    n_update_steps = 0
    learning_type = None
    replay_buffer = None
    num_replay = 1

    algorithm = None
    lambda_val = 0.0

    active = True

    actions = []

    initial_state = None
    current_state = None
    next_state = None

    initial_action = -1
    actual_action = -1

    optimizer = None
    network = None

    state_space = []

    historical_data = None
    historical_data_columns = []

    def __init__(self, args=None):
        if args is None:
            args = {}
        start_loc = None
        state_dim = 0
        for key in args:
            if 'random_seed' in key:
                setattr(self, key[:key.index('random_seed')] + 'rand_generator', np.random.RandomState(args[key]))
            elif key == 'state_dim':
                state_dim = args['state_dim']
            elif key == 'start_loc':
                start_loc = args['start_loc']
            else:
                setattr(self, key, args[key])

        if state_dim == 1:
            self.historical_data_columns.append('STATE')
        else:
            for i in range(1, state_dim + 1):
                self.historical_data_columns.append('STATE_VAR{0}'.format(i))
        self.historical_data_columns.append('INITIAL_ACTION')
        self.historical_data_columns.append('REWARD')

        self.initialize_state(state_dim, start_loc)
        self.reset()

    def network_init(self, network_type, num_hidden_units, random_seed):
        self.network = ActionValueNetwork(network_type, len(list(self.current_state)), num_hidden_units,
                                          len(self.actions),
                                          random_seed)

    def buffer_init(self, replay_buffer_size, minibatch_size, random_seed):
        if not self.learning_type == LearningType.Replay:
            pass
        self.replay_buffer = ReplayBuffer(replay_buffer_size, minibatch_size, random_seed)

    def add_to_state_space(self, s):
        if s is None:
            pass
        if s not in self.state_space:
            self.state_space.append(s)

    def initialize_state(self, state_dim, start_loc):
        start_loc_list = list(start_loc)
        if state_dim > len(start_loc_list):
            for _ in range(len(start_loc_list), state_dim):
                start_loc_list.append(0)
            self.initial_state = tuple(start_loc_list)
        else:
            self.initial_state = tuple(start_loc)
        self.current_state = self.initial_state
        self.state_space.append(self.current_state)

    def reset(self):
        self.current_state = self.initial_state
        self.historical_data = pd.DataFrame(columns=self.historical_data_columns)
        self.active = True
        self.n_update_steps = 0

    def optimizer_init(self, learning_rate, beta_m, beta_v, epsilon):
        self.optimizer = Adam(self.network.layer_sizes, learning_rate, beta_m, beta_v, epsilon)

    def step(self, r, terminal=False):
        self.n_update_steps += 1
        if not self.initial_action == self.actual_action:
            r *= -1
        self.algorithm.policy.update(r, self.initial_action)
        self.active = not terminal

        self.add_to_state_space(self.next_state)

        current_q = deepcopy(self.network)
        if self.learning_type == LearningType.Replay:
            self.replay_buffer.append(self.current_state, self.initial_action, r, int(terminal), self.next_state)
            if self.replay_buffer.size() > self.replay_buffer.minibatch_size:
                for _ in range(self.num_replay):
                    experiences = self.replay_buffer.sample()
                    self.optimize_network_bulk(experiences, current_q)
        else:
            self.optimize_network(self.current_state, self.initial_action, self.next_state, r, int(terminal), current_q)

        # TODO: Add data to supervised learning
        self.add_to_supervised_learning(r)

        if self.active:
            self.current_state = self.next_state

    def add_to_supervised_learning(self, r):
        new_data = {}
        l = list(self.current_state)
        if len(l) == 1:
            new_data.update({'STATE': self.current_state})
        else:
            for i, value in enumerate(l):
                new_data.update({'STATE_VAR{0}'.format(i + 1): value})

        new_data.update({'INITIAL_ACTION': self.initial_action, 'REWARD': r})
        self.historical_data = self.historical_data.append(new_data, ignore_index=True)

    def choose_next_action(self):
        action_values = self.network.get_action_values(self.current_state)
        self.initial_action = self.algorithm.policy.choose_action(action_values)
        # TODO: add supervised learning to block potential negative actions
        self.actual_action = self.initial_action

    def optimize_network_bulk(self, experiences, current_q):
        coin_side = self.network.determine_coin_side()
        unique_idx_states_experienced = []
        delta_mat = np.zeros((0, self.network.num_actions))
        for experience in experiences:
            s, a, r, terminal, s_ = experience
            delta_vec = self.get_delta_vec(s, a, s_, r, terminal, current_q)
            state_index = self.state_space.index(s)
            if state_index in unique_idx_states_experienced:
                delta_mat[unique_idx_states_experienced.index(state_index), :] += delta_vec
            else:
                unique_idx_states_experienced.append(state_index)
                delta_mat = np.append(delta_mat, delta_vec)

        states = np.zeros((len(unique_idx_states_experienced), len(list(self.current_state))))
        for i in range(len(unique_idx_states_experienced)):
            idx = unique_idx_states_experienced[i]
            states[0] = np.array([self.state_space[idx]])
        self.update_network_with_delta_mat(states, delta_mat, coin_side)

    def optimize_network(self, s, a, s_, r, terminal, current_q):
        delta_vec, coin_side = self.get_delta_vec(s, a, s_, r, terminal, current_q)
        delta_mat = np.zeros((1, self.network.num_actions))
        delta_mat[0, :] = delta_vec

        states = np.zeros((1, len(list(s))))
        states[0] = np.array([s])

        self.update_network_with_delta_mat(states, delta_mat, coin_side)

    def update_network_with_delta_mat(self, states, delta_mat, coin_side):
        target_update = self.network.get_target_update(states, delta_mat, coin_side)
        weights = self.network.get_weights()

        for i in range(len(weights)):
            weights[i] = self.optimizer.update_weights(weights[i], target_update)

        self.network.set_weights(weights)

    def get_delta_vec(self, s, a, s_, r, terminal, current_q):
        target_error, coin_side = self.algorithm.get_target_error(s, a, s_, r, terminal, self.network, current_q)
        delta_vec = np.zeros(len(self.actions))
        delta_vec[a] = target_error
        return delta_vec, coin_side

    def get_total_reward(self):
        total_reward = 0

        for state in self.state_space:
            action_values = self.network.get_action_values(state)
            total_reward += np.sum(action_values)

        return total_reward

    def get_all_positive_actions(self):
        positive_actions = {}

        for state in self.state_space:
            action_values = self.network.get_action_values(state)
            pa_list = list((np.where(action_values > 0))[0])
            positive_actions[state] = [action for action in self.actions if self.actions.index(action) in pa_list]
        return positive_actions