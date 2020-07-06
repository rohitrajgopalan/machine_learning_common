import enum
import numpy as np
import pandas as pd

from reinforcement_learning.environment.environment import ActionType
from reinforcement_learning.network.actionvaluenetwork import ActionValueNetwork
from reinforcement_learning.policy.epsilon_greedy import EpsilonGreedy
from reinforcement_learning.policy.softmax import Softmax
from reinforcement_learning.policy.ucb import UCB
from reinforcement_learning.replay.replay_buffer import ReplayBuffer
from reinforcement_learning.supervised.negative_action_blocker import NegativeActionBlocker


class LearningType(enum.Enum):
    ONLINE = 1,
    REPLAY = 2

    @staticmethod
    def all():
        return [LearningType.ONLINE, LearningType.REPLAY]


class Agent:
    learning_type = None
    is_double_agent = False

    algorithm = None

    policy_network = None
    target_network = None

    active = True
    actions = []

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

        if args is None:
            args = {}
        for key in args:
            if 'actions_csv_file' in key:
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
        self.add_state(self.current_state)

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
        self.historical_data_columns.append('HYPERPARAMETER')
        self.historical_data_columns.append('TARGET_VALUE')
        self.historical_data_columns.append('BLOCKED?')

        if len(actions_csv_file) > 0:
            self.load_actions_from_csv(actions_csv_file)

        self.reset()

    def blocker_init(self, csv_dir, dl_args=None):
        if not self.enable_action_blocking:
            pass
        self.action_blocker = NegativeActionBlocker(csv_dir, self.state_dim, dl_args)

    def buffer_init(self, num_replay, size, minibatch_size, random_seed):
        if self.learning_type == LearningType.REPLAY:
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

    def add_action(self, action):
        if action not in self.actions:
            self.actions.append(action)
        self.policy_network.add_action()
        if self.is_double_agent:
            self.target_network.add_action()
        self.algorithm.policy.add_action()

    def network_init(self, action_network_args):
        action_network_args.update({'num_inputs': self.state_dim, 'num_outputs': len(self.actions)})
        self.policy_network = ActionValueNetwork(action_network_args)
        if self.is_double_agent:
            self.target_network = ActionValueNetwork(action_network_args)

    def step(self, r1, r2):
        self.n_update_steps += 1
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

        if self.learning_type == LearningType.REPLAY:
            self.replay_buffer.append(self.current_state, self.initial_action, r, 1 - int(self.active), self.next_state)
            if self.replay_buffer.size() > self.replay_buffer.minibatch_size:
                for _ in range(self.num_replay):
                    experiences = self.replay_buffer.sample()
                    self.optimize_network_bulk(experiences)
        else:
            self.optimize_network(self.current_state, self.initial_action, self.next_state, r, 1 - int(self.active))

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
                         'TARGET_VALUE': self.algorithm.calculate_target_value(self.initial_action,
                                                                               self.next_state, r,
                                                                               int(self.active), self.policy_network,
                                                                               self.target_network),
                         'HYPERPARAMETER': hyperparameter_value,
                         'BLOCKED?': blocked_boolean})
        self.historical_data = self.historical_data.append(new_data, ignore_index=True)

    def choose_next_action(self):
        action_values = self.policy_network.get_action_values(self.current_state)
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

    def optimize_network(self, s, a, s_, r, terminal):
        q_target = np.zeros((1, len(self.actions)))
        q_target[0, a] = self.algorithm.get_target_value(s, a, s_, r, 1 - terminal, self.policy_network,
                                                         self.target_network)
        self.policy_network.update_network(np.array([s]), q_target)

    def optimize_network_bulk(self, experiences):
        q_target = np.zeros((len(experiences), len(self.actions)))
        states = np.zeros((len(experiences), self.state_dim))
        for batch_idx, experience in enumerate(experiences):
            s, a, s_, r, terminal = experience
            q_target[batch_idx, a] = self.algorithm.get_target_value(s, a, s_, r, 1 - terminal, self.policy_network,
                                                                     self.target_network)
            states[batch_idx] = s
        self.policy_network.update_network(states, q_target)

    def get_total_reward(self):
        total_reward = 0

        for state in self.experienced_states:
            total_reward += np.sum(self.policy_network.get_action_values(state))

        return total_reward

    def determine_final_policy(self):
        final_policy = {}

        for state in self.experienced_states:
            final_policy[state] = self.algorithm.policy.actions_with_max_value(
                self.policy_network.get_action_values(state))

        return final_policy
