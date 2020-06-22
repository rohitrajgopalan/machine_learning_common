import enum

import numpy as np
import pandas as pd

from supervised_learning.common import select_best_regressor, select_best_classifier, randomly_select_classifier, \
    randomly_select_regressor


class MethodType(enum.Enum):
    Classification = 1
    Regression = 2


class SupervisedLearningHelper:
    method_type = None
    method = None
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

    def update(self):
        if self.method_type == MethodType.Regression:
            self.method, self.historical_data = select_best_regressor(self.csv_dir, features=self.features,
                                                                      label=self.label, filters=self.filters)
            if self.method is None:
                self.method = randomly_select_regressor()
        else:
            self.method, self.historical_data = select_best_classifier(self.csv_dir, features=self.features,
                                                                       label=self.label)
            if self.method is None:
                self.method = randomly_select_classifier()
        if self.historical_data is None:
            cols = self.features
            cols.append(self.label)
            self.historical_data = pd.DataFrame(columns=cols)

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
        self.method.fit(x, y)

    def predict(self, state, action):
        if self.state_dim == 1:
            input_x = [state]
        else:
            input_x = list(state)
        input_x.append(action)
        x = np.array([input_x])
        return self.method.predict(x)
