from os import listdir
from os.path import join, isfile
import numpy as np
import pandas as pd

from reinforcement_learning.environment.environment import ActionType
from reinforcement_learning.replay.choose_buffer import choose_buffer
from reinforcement_learning.supervised.negative_action_blocker import NegativeActionBlocker
from .learning_type import LearningType
from ..replay.prioritized_replay_buffer import PrioritizedReplayBuffer


class Agent:
    algorithm = None
    discount_factor = 0.0
    active = True
    actions = []
    action_dim = 0
    action_type = None
    learning_type = None

    state_dim = 0
    state_type = None
    initial_state = None
    current_state = None
    next_state = None

    initial_action = -1
    actual_action = -1

    experienced_states = []

    action_blocking_data = None
    action_blocking_data_columns = []
    action_blocking_counter = 0

    experienced_samples = None
    experienced_samples_columns = []
    sample_counter = 0

    enable_action_blocking = False
    action_blocker = None

    num_replay = 0
    replay_buffer = None

    samples_dir = ''

    def __init__(self, args={}):
        # In an multi-agent setting, data from other agents might come and populate here which will
        # break the code and give unexpected results.
        # So a pre-clean-up is necessary
        if len(self.experienced_states) > 0:
            self.experienced_states = []
        if len(self.action_blocking_data_columns) > 0:
            self.action_blocking_data_columns = []
        if len(self.experienced_samples_columns) > 0:
            self.experienced_samples_columns = []

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
        self.current_state = self.initial_state
        self.add_state(self.current_state)

        if self.state_dim == 1:
            self.action_blocking_data_columns.append('STATE')
            self.experienced_samples_columns.append('STATE')
            self.experienced_samples_columns.append('NEXT_STATE')
        else:
            for i in range(1, self.state_dim + 1):
                self.action_blocking_data_columns.append('STATE_VAR{0}'.format(i))
                self.experienced_samples_columns.append('STATE_VAR{0}'.format(i))
                self.experienced_samples_columns.append('NEXT_STATE_VAR{0}'.format(i))
        if self.action_type in [int, float, str]:
            self.action_blocking_data_columns.append('INITIAL_ACTION')
            self.experienced_samples_columns.append('INITIAL_ACTION')
        else:
            if self.action_dim == 1:
                self.action_blocking_data_columns.append('INITIAL_ACTION')
                self.experienced_samples_columns.append('INITIAL_ACTION')
            else:
                for i in range(1, self.action_dim + 1):
                    self.action_blocking_data_columns.append('INITIAL_ACTION_VAR{0}'.format(i))
                    self.experienced_samples_columns.append('INITIAL_ACTION_VAR{0}'.format(i))
        self.experienced_samples_columns.append('REWARD')
        self.experienced_samples_columns.append('DONE?')
        self.action_blocking_data_columns.append('BLOCKED?')

        if len(actions_csv_file) > 0:
            self.load_actions_from_csv(actions_csv_file)
        elif len(actions_npy_file) > 0:
            self.load_actions_from_npy(actions_npy_file)
        elif len(actions_dir) > 0:
            self.load_actions_from_dir(actions_dir)

        if self.learning_type == LearningType.COMBINED:
            self.load_samples()

        self.reset()

    def blocker_init(self, csv_dir, dl_args=None):
        if not self.enable_action_blocking:
            pass
        self.action_blocker = NegativeActionBlocker(csv_dir, self.state_dim, self.action_dim, dl_args)

    def buffer_init(self, num_replay, buffer_type, size, minibatch_size, random_seed=None):
        self.replay_buffer = choose_buffer(buffer_type, size, minibatch_size, random_seed)
        self.num_replay = num_replay

    def add_state(self, s):
        if s is None:
            pass
        if not type(s) == int:
            if type(s) == list:
                s = tuple(s)
        if len(self.experienced_states) or s not in self.experienced_states:
            self.experienced_states.append(s)

    def reset(self):
        self.action_blocking_data = pd.DataFrame(columns=self.action_blocking_data_columns)
        if not self.learning_type == LearningType.OFF_POLICY:
            self.experienced_samples = pd.DataFrame(columns=self.experienced_samples_columns)
        self.current_state = self.initial_state
        self.active = True

    def load_actions_from_csv(self, csv_file):
        df = pd.read_csv(csv_file, engine='python', skip_blank_lines=True)
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
            self.action_type = type(action)
            if not does_exist:
                self.actions.append(action)

    def load_actions_from_npy(self, npy_file):
        actions_from_npy = np.load(npy_file)
        self.action_type = np.ndarray
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
            if self.action_type == np.ndarray:
                if np.array_equal(action, action_in_question):
                    return a, True
            elif action == action_in_question:
                return a, True
        return -1, False

    def add_action(self, action):
        a_index, does_exist = self.does_action_already_exist(action)
        if does_exist:
            return a_index
        else:
            self.actions.append(action)
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

    def assign_initial_action(self):
        pass

    def choose_next_action(self):
        self.assign_initial_action()
        if self.initial_action is not None:
            action_val = self.actions[self.initial_action]
            if self.action_type == str:
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
                if self.action_type == str:
                    action_val_ = action
                if not self.action_blocker.should_we_block_action(self.current_state, action_val_):
                    self.actual_action = action
                    break
            self.did_block_action = True
        else:
            self.actual_action = self.initial_action
            self.did_block_action = False

    def step(self, r1, r2, should_action_be_blocked=False):
        if self.did_block_action:
            r = r2 * -1
        else:
            r = r1
        if self.algorithm is not None:
            self.algorithm.policy.update(self.initial_action, should_action_be_blocked)

        self.add_to_supervised_learning(should_action_be_blocked)

        self.add_state(self.current_state)
        self.add_state(self.next_state)

        if not self.learning_type == LearningType.OFF_POLICY:
            self.add_to_experienced_samples(r)

        self.post_step_process(r)

    def post_step_process(self, r):
        if type(self.replay_buffer) == PrioritizedReplayBuffer:
            self.replay_buffer.append(self.current_state, self.initial_action, self.next_state, r, 1 - int(self.active), target_error=self.get_target_error(r))
        else:
            self.replay_buffer.append(self.current_state, self.initial_action, self.next_state, r, 1 - int(self.active))
        if self.replay_buffer.size() >= self.replay_buffer.minibatch_size:
            self.n_update_steps += 1
            for _ in range(self.num_replay):
                experiences = self.replay_buffer.sample()
                self.optimize_network(experiences)
        if self.active:
            self.current_state = self.next_state

    def add_to_experienced_samples(self, r):
        new_data = {}
        if self.state_dim == 1:
            new_data.update({'STATE': self.current_state, 'NEXT_STATE': self.next_state})
        else:
            for i in range(self.state_dim):
                state_val = self.current_state[i]
                next_state_val = self.next_state[i]
                if type(state_val) == bool:
                    state_val = int(state_val)
                if type(next_state_val) == bool:
                    next_state_val = int(next_state_val)
                new_data.update({'STATE_VAR{0}'.format(i + 1): state_val, 'NEXT_STATE_VAR{0}'.format(i+1): next_state_val})
        if self.action_type in [int, float, str]:
            action = self.actions[self.initial_action]
            new_data.update({'INITIAL_ACTION': self.initial_action if self.action_type == str else action})
        else:
            action = self.actions[self.initial_action]
            action = np.array([action])
            if self.action_dim == 1:
                new_data.update({'INITIAL_ACTION': action[0, 0]})
            else:
                for i in range(self.action_dim):
                    new_data.update({'INITIAL_ACTION_VAR{0}'.format(i + 1): action[0, i]})
        new_data.update({'REWARD': r, 'DONE?': 1-int(self.active)})
        try:
            df = pd.DataFrame(new_data)
            self.experienced_samples = pd.concat([self.experienced_samples, df], ignore_index=True)
        except MemoryError:
            print('Unable to add experience to sample due to memory issues')
        except ValueError:
            print('Duplicate Row issues as we tried to add a new row for sampling')
            print('New Row details: {0}'.format(new_data))

    def add_to_supervised_learning(self, should_action_be_blocked):
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
        if self.action_type in [int, float, str]:
            action = self.actions[self.initial_action]
            new_data.update({'INITIAL_ACTION': self.initial_action if self.action_type == str else action})
        else:
            action = self.actions[self.initial_action]
            action = np.array([action])
            if self.action_dim == 1:
                new_data.update({'INITIAL_ACTION': action[0, 0]})
            else:
                for i in range(self.action_dim):
                    new_data.update({'INITIAL_ACTION_VAR{0}'.format(i + 1): action[0, i]})

        new_data.update({'BLOCKED?': blocked_boolean})
        try:
            df = pd.DataFrame(new_data)
            self.action_blocking_data = pd.concat([self.action_blocking_data, df], ignore_index=True)
        except MemoryError:
            print('Unable to add row to action blocking data due to memory issues')
        except ValueError:
            print('Duplicate Row issues as we tried to add a new row for action blocking')
            print('New Row details: {0}'.format(new_data))
    
    def optimize_network(self, experiences):
        pass

    def get_results(self):
        return 0, {}

    def load_samples(self):
        data_files = [join(self.samples_dir, f) for f in listdir(self.samples_dir) if isfile(join(self.samples_dir, f))]
        for file in data_files:
            samples = pd.read_csv(file, engine='python', skip_blank_lines=True)
            samples = samples.dropna()
            for index, row in samples.iterrows():
                if self.state_dim == 1:
                    self.current_state = float(row['STATE']) if '.' in row['STATE'] else int(row['STATE'])
                    self.next_state = float(row['NEXT_STATE']) if '.' in row['NEXT_STATE'] else int(row['NEXT_STATE'])
                else:
                    self.current_state = []
                    self.next_state = []
                    for i in range(1, self.state_dim + 1):
                        state_col = 'STATE_VAR{0}'.format(i)
                        next_state_col = 'NEXT_{0}'.format(state_col)
                        self.current_state.append(float(row[state_col]) if '.' in row[state_col] else int(row[state_col]))
                        self.next_state.append(float(row[next_state_col]) if '.' in row[next_state_col] else int(row[next_state_col]))
                    if self.state_type == tuple:
                        self.current_state = tuple(self.current_state)
                        self.next_state = tuple(self.next_state)
                    else:
                        self.current_state = np.array(self.current_state)
                        self.next_state = np.array(self.next_state)

                if self.action_type == float:
                    action = float(row['ACTION'])
                elif self.action_type == int:
                    action = int(row['ACTION'])
                elif self.action_type == str:
                    action = row['ACTION']
                else:
                    if self.action_dim == 1:
                        action = float(row['ACTION']) if '.' in row['ACTION'] else int(row['ACTION'])
                    else:
                        action = []
                        for i in range(1, self.action_dim + 1):
                            action_col = 'INITIAL_ACTION_VAR{0}'.format(i)
                            action.append(float(row[action_col]) if '.' in row[action_col] else int(action_col))
                        if self.action_type == tuple:
                            action = tuple(action)
                        else:
                            action = np.array(action)
                a_index, does_exist = self.does_action_already_exist(action)
                if not does_exist:
                    self.actions.append(action)
                    self.initial_action = len(self.actions)-1
                else:
                    self.initial_action = a_index
                reward = float(row['REWARD'])
                self.active = int(row['DONE?']) == 0
                self.post_step_process(reward)

    def get_target_error(self, reward):
        return 0