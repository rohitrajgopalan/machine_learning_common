from reinforcement_learning.supervised.supervised_learning_helper import SupervisedLearningHelper, MethodType


class TargetValuePredictor:
    supervised_learning_helper = None

    def __init__(self, csv_dir, state_dim, algorithm, dl_args=None):
        filters = {'POLICY': algorithm.policy.__class__.__name__, 'GAMMA': algorithm.discount_factor,
                   'ALGORITHM': '{0}'.format(algorithm.algorithm_name.name)}
        hyperparameter_value = 0
        if filters['POLICY'] == 'Softmax':
            hyperparameter_value = getattr(algorithm.policy, 'tau')
        elif filters['POLICY'] == 'UCB':
            hyperparameter_value = getattr(algorithm.policy, 'ucb_c')
        elif filters['POLICY'] == 'EpsilonGreedy':
            hyperparameter_value = getattr(algorithm.policy, 'epsilon')
        filters.update({'HYPERPARAMETER': hyperparameter_value})
        self.supervised_learning_helper = SupervisedLearningHelper.choose_helper(MethodType.Regression, csv_dir, state_dim, 'TARGET_VALUE', filters, dl_args)

    def update_predictor(self, state, action, target_value):
        self.supervised_learning_helper.add(state, action, target_value)

    def get_target_value(self, state, action):
        self.supervised_learning_helper.predict(state, action)
