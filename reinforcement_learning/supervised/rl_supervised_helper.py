from supervised_learning.supervised_learning_helper import SupervisedLearningHelper
import numpy as np


class RLSupervisedHelper:
    state_dim = 0
    action_dim = 0
    supervised_learning_helper = None

    def __init__(self, method_type, csv_dir, state_dim, action_dim, label, dl_args=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        features = []
        if self.state_dim == 1:
            features.append('STATE')
        else:
            for i in range(1, self.state_dim + 1):
                features.append('STATE_VAR{0}'.format(i))
        if self.action_dim == 1:
            features.append('INITIAL_ACTION')
        else:
            for i in range(1, self.action_dim + 1):
                features.append('INITIAL_ACTION_VAR{0}'.format(i))
        self.supervised_learning_helper = SupervisedLearningHelper.choose_helper(method_type, files_dir=csv_dir, features=features, label=label, dl_args=dl_args)

    def add(self, state, action, target_value):
        new_data = {}
        if self.state_dim == 1:
            new_data.update({'STATE': state})
        else:
            for i in range(self.state_dim):
                new_data.update({'STATE_VAR{0}'.format(i+1): state[i]})
        if self.action_dim == 1:
            if type(action) == int or type(action) == float:
                new_data.update({'INITIAL_ACTION': action})
            else:
                action = np.array([action])
                new_data.update({'INITIAL_ACTION': action[0, 0]})
        else:
            action = np.array([action])
            for i in range(self.action_dim):
                new_data.update({'INITIAL_ACTION_VAR{0}'.format(i+1): action[0, i]})
        self.supervised_learning_helper.add(new_data, target_value)

    def predict(self, state, action):
        input_x = []
        if self.state_dim == 1:
            input_x.append(state)
        else:
            for i in range(self.state_dim):
                input_x.append(state[i])
        if type(action) == int or type(action) == float:
            input_x.append(action)
        else:
            action = np.array([action])
            for i in range(self.action_dim):
                input_x.append(action[0, i])
        inputs = np.array([input_x])
        predictions = self.supervised_learning_helper.predict(inputs)
        try:
            return predictions[0, 0]
        except IndexError:
            return predictions[0]
