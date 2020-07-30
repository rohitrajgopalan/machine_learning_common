import enum
from os import listdir
from os.path import join, isfile

import numpy as np
import pandas as pd

from reinforcement_learning.environment.environment import ActionType
from reinforcement_learning.network.actionvaluenetwork import ActionValueNetwork
from reinforcement_learning.policy.epsilon_greedy import EpsilonGreedy
from reinforcement_learning.policy.softmax import Softmax
from reinforcement_learning.policy.ucb import UCB
from reinforcement_learning.replay.replay_buffer import ReplayBuffer
from reinforcement_learning.supervised.negative_action_blocker import NegativeActionBlocker
from reinforcement_learning.supervised.target_value_predictor import TargetValuePredictor


class LearningType(enum.Enum):
    ONLINE = 1,
    REPLAY = 2

    @staticmethod
    def all():
        return [LearningType.ONLINE, LearningType.REPLAY]


class TDAgent:
    learning_type = None
    is_double_agent = False

    algorithm = None

    policy_network = None
    target_network = None

    active = True
    actions = []
    action_dim = 0

    state_dim = 0
    initial_state = None
    current_state = None
    next_state = None

    initial_action = -1
    actual_action = -1

    experienced_states = []

    historical_data = None
    historical_data_columns = []

    enable_action_blocking = False
    enable_iterator = False
    iterator = None
    action_blocker = None

    num_replay = 0
    replay_buffer = None

    def __init__(self, args=None):
        # In an multi-agent setting, data from other agents might come and populate here which will
        # break the code and give unexpected results.
        # So a pre-clean-up is necessary
        if len(self.experienced_states) > 0:
            self.experienced_states = []
        if len(self.historical_data_columns) > 0:
            self.historical_data_columns = []

        self.did_block_action = False
        actions_csv_file = ''
        actions_npy_file = ''
        actions_dir = ''

        if args is None:
            args = {}
        for key in args:
            if key == 'actions_csv_file':
                actions_csv_file = args[key]
            elif key == 'actions_npy_file':
                actions_npy_file = args[key]
            elif key == 'actions_dir':
                actions_dir = args[key]
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
        self.add_state(self.current_state)

        if self.state_dim == 1:
            self.historical_data_columns.append('STATE')
        else:
            for i in range(1, self.state_dim + 1):
                self.historical_data_columns.append('STATE_VAR{0}'.format(i))
        if self.action_dim == 1:
            self.historical_data_columns.append('INITIAL_ACTION')
        else:
            for i in range(1, self.action_dim + 1):
                self.historical_data_columns.append('INITIAL_ACTION_VAR{0}'.format(i))
        self.historical_data_columns.append('REWARD')
        self.historical_data_columns.append('ALGORITHM')
        self.historical_data_columns.append('GAMMA')
        self.historical_data_columns.append('POLICY')
        self.historical_data_columns.append('HYPERPARAMETER')
        self.historical_data_columns.append('TARGET_VALUE')
        self.historical_data_columns.append('BLOCKED?')

        if len(actions_csv_file) > 0:
            self.load_actions_from_csv(actions_csv_file)
        elif len(actions_npy_file) > 0:
            self.load_actions_from_npy(actions_npy_file)
        elif len(actions_dir) > 0:
            self.load_actions_from_dir(actions_dir)

        self.reset()

    def blocker_init(self, csv_dir, dl_args=None):
        if not self.enable_action_blocking:
            pass
        self.action_blocker = NegativeActionBlocker(csv_dir, self.state_dim, dl_args)

    def buffer_init(self, num_replay, size, minibatch_size, random_seed):
        self.replay_buffer = ReplayBuffer(size, minibatch_size, random_seed)
        self.num_replay = num_replay

    def add_state(self, s):
        if s is None:
            pass
        if not type(s) == int:
            if type(s) == np.ndarray:
                s = tuple(s.reshape(1, -1)[0])
            elif type(s) == list:
                s = tuple(s)
        if s not in self.experienced_states:
            self.experienced_states.append(s)

    def reset(self):
        self.current_state = self.initial_state
        self.historical_data = pd.DataFrame(columns=self.historical_data_columns)
        self.active = True

    def load_actions_from_csv(self, csv_file):
        df = pd.read_csv(csv_file)
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
                    action_as_list = row['ACTION'].split(';')
                    actions = []
                    for a in action_as_list:
                        try:
                            if '.' in a:
                                a = float(a)
                            else:
                                a = int(a)
                            actions.append(a)
                        except TypeError:
                            continue
                    if action_type == 'tuple':
                        action = tuple(actions)
                    else:
                        action = np.array([actions])
            else:
                action = row['ACTION']
            a_index_, does_exist = self.does_action_already_exist(action)
            if not does_exist:
                self.actions.append(action)

    def load_actions_from_npy(self, npy_file):
        actions_from_npy = np.load(npy_file)
        if len(actions_from_npy.shape) == 3:
            _, _, n_elements = actions_from_npy.shape
            actions_from_npy = actions_from_npy.reshape(-1, n_elements)
        for a_index in range(actions_from_npy.shape[0]):
            action = actions_from_npy[a_index]
            a_index_, does_exist = self.does_action_already_exist(action)
            if not does_exist:
                self.actions.append(action)

    def load_actions_from_dir(self, actions_dir):
        data_files = [join(actions_dir, f) for f in listdir(actions_dir) if isfile(join(actions_dir, f))]
        for file in data_files:
            if file.endswith('.csv'):
                self.load_actions_from_csv(file)
            elif file.endswith('.npy'):
                self.load_actions_from_npy(file)

    def does_action_already_exist(self, action_in_question):
        for a, action in enumerate(self.actions):
            if type(action) == np.ndarray and type(action_in_question) == np.ndarray:
                if np.array_equal(action, action_in_question):
                    return a, True
            elif action == action_in_question:
                return a, True
        return -1, False

    def add_action(self, action):
        a_index, does_exist = self.does_action_already_exist(action)
        if not does_exist:
            self.actions.append(action)

    def network_init(self, action_network_args):
        action_network_args.update({'num_inputs': self.state_dim, 'num_outputs': len(self.actions)})
        self.policy_network = ActionValueNetwork(action_network_args)
        if self.is_double_agent:
            self.target_network = ActionValueNetwork(action_network_args)

    def iterator_init(self, csv_dir, dl_args=None):
        if not self.enable_iterator:
            pass
        self.iterator = TargetValuePredictor(csv_dir, self.state_dim, self.action_dim, self.algorithm, dl_args)

    def step(self, r1, r2):
        if self.did_block_action:
            r = r2 * -1
        else:
            r = r1
        self.algorithm.policy.update(self.initial_action, r)

        self.add_to_supervised_learning(r1)

        self.add_state(self.current_state)
        self.add_state(self.next_state)

        if self.is_double_agent:
            self.target_network.set_weights(self.policy_network.get_weights())

        self.replay_buffer.append(self.current_state, self.initial_action, self.next_state, r, 1 - int(self.active))
        if self.replay_buffer.size() >= self.replay_buffer.minibatch_size:
            self.n_update_steps += 1
            for _ in range(self.num_replay):
                experiences = self.replay_buffer.sample()
                self.optimize_action_network(experiences)

        if self.active:
            self.current_state = self.next_state

    def add_to_supervised_learning(self, r):
        blocked_boolean = 1 if r <= (self.algorithm.policy.min_penalty * -1) else 0
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
        if self.action_dim == 1:
            action = self.actions[self.initial_action]
            new_data.update({'INITIAL_ACTION': self.initial_action if type(action) == str else action})
        else:
            action = self.actions[self.initial_action]
            action = np.array([action])
            for i in range(self.action_dim):
                new_data.update({'INITIAL_ACTION_VAR{0}'.format(i+1): action[0, i]})

        new_data.update({'REWARD': r,
                         'ALGORITHM': '{0}{1}'.format(self.algorithm.algorithm_name.name,
                                                      '_LAMBDA' if 'Lambda' in self.algorithm.__class__.__name__ else ''),
                         'GAMMA': self.algorithm.discount_factor,
                         'POLICY': self.algorithm.policy.__class__.__name__,
                         'TARGET_VALUE': self.algorithm.calculate_target_value(self.initial_action,
                                                                               self.next_state, r,
                                                                               int(self.active), self.policy_network,
                                                                               self.target_network),
                         'HYPERPARAMETER': self.algorithm.policy.get_hyper_parameter(),
                         'BLOCKED?': blocked_boolean})
        self.historical_data = self.historical_data.append(new_data, ignore_index=True)

    def choose_next_action(self):
        self.initial_action = self.algorithm.policy.choose_action(self.current_state, self.policy_network)
        if self.initial_action is not None:
            action_val = self.actions[self.initial_action]
            if type(action_val) == str:
                action_val = self.initial_action
        else:
            action_val = None
        if self.enable_action_blocking and action_val is not None and self.action_blocker.should_we_block_action(
                self.current_state,
                action_val):
            other_actions = [action for action in range(len(self.actions)) if
                             not action == self.initial_action]
            self.actual_action = None
            for action in other_actions:
                action_val_ = self.actions[action]
                if type(action_val_) == str:
                    action_val_ = action
                if not self.action_blocker.should_we_block_action(self.current_state, action_val_):
                    self.actual_action = action
                    break
            self.did_block_action = True
        else:
            self.actual_action = self.initial_action
            self.did_block_action = False

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

    def optimize_action_network(self, experiences):
        q_target = np.zeros((len(experiences), len(self.actions)))
        states = np.zeros((len(experiences), self.state_dim))
        for batch_idx, experience in enumerate(experiences):
            s, a, s_, r, terminal = experience
            target_value = self.algorithm.calculate_target_value(a, s_, r, 1 - terminal, self.policy_network,
                                                                 self.target_network)
            action = self.actions[a]
            if type(action) == str:
                action = a
            if self.enable_iterator:
                predicted_value = self.iterator.predict(s, action)
                self.iterator.add(s, action, target_value)
                target_value = predicted_value
            try:
                q_target[batch_idx, a] = target_value
                states[batch_idx] = s
            except ValueError:
                print('Unable to set Target Value {0} for Batch Index {1}, State {2} Action Index {3}'.format(target_value, batch_idx, s, a))
        self.policy_network.update_network(states, q_target)

    def get_total_reward(self):
        total_reward = 0

        for state in self.experienced_states:
            total_reward += np.sum(self.policy_network.get_action_values(state))

        return total_reward

    def determine_final_policy(self):
        final_policy = {}

        for state in self.experienced_states:
            chosen_actions = self.algorithm.policy.actions_with_max_value(self.policy_network.get_action_values(state))
            final_policy[state] = chosen_actions

        return final_policy
