from os import listdir
from os.path import isfile, join

import numpy as np
import pandas as pd

from .policy import Policy


class StaticPolicy(Policy):
    policy_table = {}
    state_dim = 0

    def __init__(self, args):
        super().__init__(args)
        if 'policy_csv_file' in args:
            self.load_policy_from_csv(args['policy_csv_file'])
        elif 'policy_dir' in args:
            files_dir = args['policy_dir']
            data_files = [join(files_dir, f) for f in listdir(files_dir) if isfile(join(files_dir, f))]
            for csv_file in data_files:
                self.load_policy_from_csv(csv_file)

    def load_policy_from_csv(self, csv_file):
        df = pd.read_csv(csv_file)
        state_cols = [col for col in df.columns if col.startswith('STATE')]
        for index, row in df.iterrows():
            if len(state_cols) == 1:
                val = row['STATE']
                if '.' in val:
                    val = float(val)
                else:
                    val = int(val)
                state = val
            else:
                state_as_list = []
                for col in state_cols:
                    val = row[col]
                    if '.' in val:
                        val = float(val)
                    else:
                        val = int(val)
                    state_as_list.append(val)
                state = np.array([state_as_list]).reshape(self.state_dim)
            if state not in self.policy_table:
                self.policy_table[state] = []
            action = int(row['ACTION'])
            if action not in self.policy_table[state]:
                self.policy_table[state].append(action)

    def derive(self, state, network, use_target=False):
        if type(state) == tuple:
            state = np.array([state])
        policy_probs = np.zeros(self.num_actions)
        actions_to_take = self.policy_table[state]
        for action in range(self.num_actions):
            if action in actions_to_take:
                policy_probs[action] = 1 / len(actions_to_take)
        return policy_probs

    def choose_action(self, state, network, use_target=False):
        if type(state) == tuple:
            state = np.array([state])
        if state in self.policy_table:
            return self.rand_generator.choice(self.policy_table[state])
        else:
            return None
