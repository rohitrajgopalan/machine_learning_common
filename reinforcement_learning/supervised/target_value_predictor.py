from reinforcement_learning.supervised.supervised_learning_helper import SupervisedLearningHelper, MethodType


class TargetValuePredictor(SupervisedLearningHelper):
    def __init__(self, csv_dir, state_dim, algorithm, network_type):
        filters = {'POLICY': algorithm.policy.__class__.__name__, 'GAMMA': algorithm.discount_factor, 'NETWORK_TYPE': network_type.name,
                   'ALGORITHM': '{0}{1}'.format(algorithm.algorithm_name.name, '_LAMBDA' if 'Lambda' in algorithm.__class__.__name__ else '')}
        hyperparameter_value = 0
        if filters['POLICY'] == 'Softmax':
            hyperparameter_value = getattr(algorithm.policy, 'tau')
        elif filters['POLICY'] == 'UCB':
            hyperparameter_value = getattr(algorithm.policy, 'ucb_c')
        elif filters['POLICY'] == 'EpsilonGreedy':
            hyperparameter_value = getattr(algorithm.policy, 'epsilon')
        filters.update({'HYPERPARAMETER': hyperparameter_value})
        super().__init__(MethodType.Regression, csv_dir, state_dim, 'TARGET_VALUE', filters)

