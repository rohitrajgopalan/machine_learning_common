from supervised_learning.supervised_learning_helper import SupervisedLearningHelper
import numpy as np


class RLSupervisedHelper:
    state_dim = 0
    supervised_learning_helper = None

    def __init__(self, method_type, csv_dir, state_dim, label, filters, dl_args=None):
        self.state_dim = state_dim
        features = []
        if self.state_dim == 1:
            features.append('STATE')
        else:
            for i in range(1, self.state_dim + 1):
                features.append('STATE_VAR{0}'.format(i))
        features.append('INITIAL_ACTION')
        self.supervised_learning_helper = SupervisedLearningHelper.choose_helper(method_type, csv_dir, features, label,
                                                                                 filters, dl_args)

    def add(self, state, action, target_value):
        new_data = {'INITIAL_ACTION': action}
        if self.state_dim == 1:
            new_data.update({'STATE': state})
        else:
            for i in range(1, self.state_dim + 1):
                new_data.update({'STATE_VAR{0}'.format(i): state[i]})
        self.supervised_learning_helper.add(new_data, target_value)

    def predict(self, state, action):
        if self.state_dim == 1:
            input_x = [state]
        else:
            input_x = list(state)
        input_x.append(action)
        inputs = np.array([input_x])
        return self.supervised_learning_helper.get_predictions(inputs)
