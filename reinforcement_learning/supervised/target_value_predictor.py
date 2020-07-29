from supervised_learning.common import MethodType
from .rl_supervised_helper import RLSupervisedHelper


class TargetValuePredictor(RLSupervisedHelper):
    def __init__(self, csv_dir, state_dim, action_dim, algorithm, dl_args=None):
        if dl_args is not None:
            dl_args.update({'loss_function': 'mse'})
        filters = {'POLICY': algorithm.policy.__class__.__name__, 'GAMMA': algorithm.discount_factor,
                   'ALGORITHM': '{0}'.format(algorithm.algorithm_name.name),
                   'HYPERPARAMETER': algorithm.policy.get_hyper_parameter()}
        super().__init__(MethodType.Regression, csv_dir, state_dim, action_dim, 'TARGET_VALUE', filters,
                         dl_args)
