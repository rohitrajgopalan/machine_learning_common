import enum
from copy import deepcopy

import numpy as np
import pandas as pd

from reinforcement_learning.environment.environment import ActionType
from reinforcement_learning.network.actionvaluenetwork import ActionValueNetwork
from reinforcement_learning.optimizer.adam_optimizer import Adam
from reinforcement_learning.policy.epsilon_greedy import EpsilonGreedy
from reinforcement_learning.policy.softmax import Softmax
from reinforcement_learning.policy.ucb import UCB
from reinforcement_learning.replay.replay_buffer import ReplayBuffer
from reinforcement_learning.supervised.negative_action_blocker import NegativeActionBlocker


class LearningType(enum.Enum):
    Online = 1
    Replay = 2


class Agent:
    agent_id = 0
    learning_type = None
    replay_buffer = None
    num_replay = 1

    algorithm = None

    active = True

    actions = []

    initial_state = None
    current_state = None
    next_state = None

    initial_action = -1
    actual_action = -1

    optimizer = None
    network = None

    state_dim = 0

    state_space = []

    historical_data = None
    historical_data_columns = []

    enable_action_blocking = False
    action_blocker = None

    def __init__(self, args=None):
        # In an multi-agent setting, data from other agents might come and populate here which will
        # break the code and give unexpected results.
        # So a pre-clean-up is necessary
        if len(self.state_space) > 0:
            self.state_space = []
        if len(self.historical_data_columns) > 0:
            self.historical_data_columns = []

        self.did_block_action = False
        actions_csv_file = ''

        if args is None:
            args = {}
        for key in args:
            if 'random_seed' in key:
                setattr(self, key[:key.index('random_seed')] + 'rand_generator', np.random.RandomState(args[key]))
            elif 'actions_csv_file' in key:
                actions_csv_file = args[key]
            else:
                setattr(self, key, args[key])

        self.n_update_steps = 0

        if self.state_dim == 1:
            if self.initial_state is None:
                self.initial_state = 0
        else:
            start_loc_list = list(self.initial_state)
            if self.state_dim > len(start_loc_list):
                for _ in range(len(start_loc_list), self.state_dim):
                    start_loc_list.append(0)
            self.initial_state = tuple(start_loc_list)
        self.current_state = self.initial_state
        self.add_to_state_space(self.current_state)

        if self.state_dim == 1:
            self.historical_data_columns.append('STATE')
        else:
            for i in range(1, self.state_dim + 1):
                self.historical_data_columns.append('STATE_VAR{0}'.format(i))
        self.historical_data_columns.append('INITIAL_ACTION')
        self.historical_data_columns.append('REWARD')
        self.historical_data_columns.append('ALGORITHM')
        self.historical_data_columns.append('GAMMA')
        self.historical_data_columns.append('POLICY')
        self.historical_data_columns.append('NETWORK_TYPE')
        self.historical_data_columns.append('HYPERPARAMETER')
        self.historical_data_columns.append('TARGET_VALUE')
        self.historical_data_columns.append('BLOCKED?')

        if len(actions_csv_file) > 0:
            self.load_actions_from_csv(actions_csv_file)

        # We want to ensure that all components have the same number of actions
        self.algorithm.num_actions = len(self.actions)

        self.reset()

    def network_init(self, network_type, initializer_type, activation_function, alpha, random_seed):
        self.network = ActionValueNetwork(network_type, initializer_type, activation_function, alpha, self.state_dim,
                                          len(self.actions), random_seed)

    def network_init(self, network_type, initializer_type, activation_function, alpha, num_hidden_units, random_seed):
        self.network = ActionValueNetwork(network_type, initializer_type, activation_function, alpha, self.state_dim,
                                          num_hidden_units,
                                          len(self.actions),
                                          random_seed)

    def buffer_init(self, num_replay, replay_buffer_size, minibatch_size, random_seed):
        if not self.learning_type == LearningType.Replay:
            pass
        self.num_replay = num_replay
        self.replay_buffer = ReplayBuffer(replay_buffer_size, minibatch_size, random_seed)

    def blocker_init(self, csv_dir):
        if not self.enable_action_blocking:
            pass
        self.action_blocker = NegativeActionBlocker(csv_dir, self.state_dim)

    def add_to_state_space(self, s):
        if s is None:
            pass
        if not type(s) == int:
            if type(s) == np.ndarray:
                s = tuple(s.reshape(1, -1)[0])
            elif type(s) == list:
                s = tuple(s)
        if s not in self.state_space:
            self.state_space.append(s)

    def reset(self):
        self.current_state = self.initial_state
        self.historical_data = pd.DataFrame(columns=self.historical_data_columns)
        self.active = True

    def optimizer_init(self, learning_rate, beta_m, beta_v, epsilon):
        self.optimizer = Adam(self.network.layer_sizes, learning_rate, beta_m, beta_v, epsilon)

    def step(self, r1, r2):
        self.n_update_steps += 1
        if self.did_block_action:
            r = r2 * -1
        else:
            r = r1
        self.algorithm.policy.update(self.initial_action, r)

        self.add_to_supervised_learning(r1)

        self.add_to_state_space(self.current_state)
        self.add_to_state_space(self.next_state)

        current_q = deepcopy(self.network)
        if self.learning_type == LearningType.Replay:
            self.replay_buffer.append(self.current_state, self.initial_action, r, 1 - int(self.active), self.next_state)
            if self.replay_buffer.size() > self.replay_buffer.minibatch_size:
                for _ in range(self.num_replay):
                    experiences = self.replay_buffer.sample()
                    self.optimize_network_bulk(experiences, current_q)
        else:
            self.optimize_network(self.current_state, self.initial_action, self.next_state, r, int(self.active),
                                  current_q)

        if self.active:
            self.current_state = self.next_state

    def add_to_supervised_learning(self, r):
        blocked_boolean = 1 if r <= self.algorithm.policy.min_penalty * -1 else 0
        if self.enable_action_blocking:
            self.action_blocker.add(self.current_state, self.initial_action, blocked_boolean)
        new_data = {}
        if self.state_dim == 1:
            new_data.update({'STATE': self.current_state})
        else:
            for i in range(self.state_dim):
                new_data.update({'STATE_VAR{0}'.format(i + 1): self.current_state[i]})

        hyperparameter_value = 0
        if type(self.algorithm.policy) == Softmax:
            hyperparameter_value = getattr(self.algorithm.policy, 'tau')
        elif type(self.algorithm.policy) == UCB:
            hyperparameter_value = getattr(self.algorithm.policy, 'ucb_c')
        elif type(self.algorithm.policy) == EpsilonGreedy:
            hyperparameter_value = getattr(self.algorithm.policy, 'epsilon')

        new_data.update({'INITIAL_ACTION': self.initial_action,
                         'REWARD': r,
                         'ALGORITHM': '{0}{1}'.format(self.algorithm.algorithm_name.name,
                                                      '_LAMBDA' if 'Lambda' in self.algorithm.__class__.__name__ else ''),
                         'GAMMA': self.algorithm.discount_factor,
                         'POLICY': self.algorithm.policy.__class__.__name__,
                         'NETWORK_TYPE': self.network.network_type.name,
                         'TARGET_VALUE': self.algorithm.calculate_target_value(self.initial_action,
                                                                               self.next_state, r,
                                                                               int(self.active), deepcopy(self.network),
                                                                               self.network.determine_coin_side()),
                         'HYPERPARAMETER': hyperparameter_value,
                         'BLOCKED?': blocked_boolean})
        self.historical_data = self.historical_data.append(new_data, ignore_index=True)

    def choose_next_action(self):
        action_values = self.network.get_action_values(self.current_state)
        self.initial_action = self.algorithm.policy.choose_action_based_from_values(action_values)

        if self.enable_action_blocking and self.initial_action is not None and self.action_blocker.should_we_block_action(
                self.current_state,
                self.initial_action):
            other_actions = [action for action in range(len(self.actions)) if not action == self.initial_action]
            self.actual_action = None
            for action in other_actions:
                if not self.action_blocker.should_we_block_action(self.current_state, action):
                    self.actual_action = action
                    break
            self.did_block_action = True
        else:
            self.actual_action = self.initial_action
            self.did_block_action = False

    def add_action(self, action):
        ids = map(id, self.actions)
        if id(action) in ids:
            return self.actions.index(action)
        else:
            self.actions.append(action)
            self.algorithm.add_action()
            self.algorithm.policy.add_action()
            self.network.add_action()
            self.optimizer.update_optimizer(self.network.layer_sizes)
            return len(self.actions) - 1

    def get_action(self, action_type):
        if action_type == ActionType.Initial:
            return self.initial_action
        elif action_type == ActionType.Actual:
            return self.actual_action
        else:
            return None

    def set_action(self, action_type, action):
        if action_type == ActionType.Initial:
            self.initial_action = action
        elif action_type == ActionType.Actual:
            self.actual_action = action
        else:
            self.initial_action = action
            self.actual_action = action

    def load_actions_from_csv(self, csv_file):
        df = pd.read_csv(csv_file)
        actions_from_csv = []
        for index, row in df.iterrows():
            if 'TYPE' in df.columns:
                action_type = row['TYPE']
                if action_type in ['int', 'float', 'str']:
                    if action_type == 'int':
                        action = int(df['ACTION'])
                    elif action_type == 'float':
                        action = float(df['ACTION'])
                    else:
                        action = str(df['ACTION'])
                else:
                    action_as_list = row['ACTION']
                    for a in action_as_list:
                        try:
                            if '.' in a:
                                a = float(a)
                            else:
                                a = int(a)
                        except TypeError:
                            continue
                    if action_type == 'tuple':
                        action = tuple(action_as_list)
                    else:
                        action = np.array([action_as_list])
            else:
                action = row['ACTION']
            actions_from_csv.append(action)
        if len(actions_from_csv) > 0:
            self.actions = actions_from_csv

    def optimize_network_bulk(self, experiences, current_q):
        coin_side = self.network.determine_coin_side()
        unique_idx_states_experienced = []
        delta_mat = np.zeros((0, self.network.num_actions))
        for experience in experiences:
            s, a, r, terminal, s_ = experience
            delta_vec = self.get_delta_vec(s, a, s_, r, 1 - terminal, current_q)
            state_index = self.state_space.index(s)
            if state_index in unique_idx_states_experienced:
                delta_mat[unique_idx_states_experienced.index(state_index)] += delta_vec
            else:
                unique_idx_states_experienced.append(state_index)
                delta_mat = np.append(delta_mat, delta_vec)

        states = np.zeros((len(unique_idx_states_experienced), len(list(self.current_state))))
        for i in range(len(unique_idx_states_experienced)):
            idx = unique_idx_states_experienced[i]
            states[i] = np.array([self.state_space[idx]])
        self.update_network_with_delta_mat(states, delta_mat, coin_side)

    def optimize_network(self, s, a, s_, r, active, current_q):
        delta_vec, coin_side = self.get_delta_vec(s, a, s_, r, active, current_q)
        states = np.zeros((1, self.state_dim))
        states[0] = np.array([s])
        self.update_network_with_delta_mat(states, delta_vec, coin_side)

    def update_network_with_delta_mat(self, states, delta_mat, coin_side):
        target_update = self.network.get_target_update(states, delta_mat, coin_side)
        weights = self.network.get_weights()

        weights[coin_side] = self.optimizer.update_weights(weights[coin_side], target_update)

        self.network.set_weights(weights)

    def get_delta_vec(self, s, a, s_, r, active, current_q):
        target_error, coin_side = self.algorithm.get_target_error(s, a, s_, r, active, self.network, current_q)
        delta_vec = np.zeros((1, len(self.actions)))
        delta_vec[0, a] = target_error
        return delta_vec, coin_side

    def get_total_reward(self):
        total_reward = 0

        for state in self.state_space:
            total_reward += np.sum(self.network.get_action_values(state))

        return total_reward

    def determine_final_policy(self):
        final_policy = {}

        for state in self.state_space:
            final_policy[state] = self.algorithm.policy.actions_with_max_value(self.network.get_action_values(state))

        return final_policy
