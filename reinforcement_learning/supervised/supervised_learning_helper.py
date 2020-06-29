import enum

import numpy as np
import pandas as pd

from .scikit_learn_helper import ScikitLearnHelper
from .deep_learning_helper import DeepLearningHelper


class MethodType(enum.Enum):
    Classification = 1
    Regression = 2


class SupervisedLearningHelper:
    method_type = None
    state_dim = 0
    csv_dir = ''
    filters = {}
    label = ''
    historical_data = None

    def __init__(self, method_type, csv_dir, state_dim, label, filters={}):
        self.method_type = method_type
        self.csv_dir = csv_dir
        self.features = []
        if state_dim == 1:
            self.features.append('STATE')
        else:
            for i in range(1, self.state_dim + 1):
                self.features.append('STATE_VAR{0}'.format(i))
        self.features.append('INITIAL_ACTION')
        self.state_dim = state_dim
        self.label = label
        self.filters = filters
        self.update()
        if self.historical_data is None:
            cols = self.features
            cols.append(self.label)
            self.historical_data = pd.DataFrame(columns=cols)

    def update(self):
        pass

    def add(self, state, action, target_value):
        new_data = {'INITIAL_ACTION': action, self.label: target_value}
        if self.state_dim == 1:
            new_data.update({'STATE': state})
        else:
            for i in range(1, self.state_dim + 1):
                new_data.update({'STATE_VAR{0}'.format(i): state[i]})
        self.historical_data = self.historical_data.append(new_data, ignore_index=True)
        x = self.historical_data[self.features]
        y = self.historical_data[self.label]
        self.fit(x, y)

    def fit(self, x, y):
        pass

    def predict(self, state, action):
        if self.state_dim == 1:
            input_x = [state]
        else:
            input_x = list(state)
        input_x.append(action)
        x = np.array([input_x])
        return self.get_predictions(x)

    def get_predictions(self, inputs):
        return None

    @staticmethod
    def choose_helper(method_type, csv_dir, state_dim, label, filters={}, dl_args=None):
        if dl_args is None:
            return ScikitLearnHelper(method_type, csv_dir, state_dim, label, filters)
        else:
            return DeepLearningHelper(method_type, csv_dir, state_dim, label, filters, dl_args)
