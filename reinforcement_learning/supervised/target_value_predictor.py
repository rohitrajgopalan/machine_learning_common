from supervised_learning.supervised_learning_helper import MethodType
from .rl_supervised_helper import RLSupervisedHelper


class TargetValuePredictor(RLSupervisedHelper):
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
        super().__init__(MethodType.Regression, csv_dir, state_dim, 'TARGET_VALUE', filters,
                         dl_args)

