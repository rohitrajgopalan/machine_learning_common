from os.path import dirname, realpath

from reinforcement_learning.algorithm.base_algorithm import *
from reinforcement_learning.network.actionvaluenetwork import NetworkType
from reinforcement_learning.policy.epsilon_greedy import EpsilonGreedy
from reinforcement_learning.supervised.target_value_predictor import TargetValuePredictor
from supervised_learning.common import *

state_dim = 1
csv_dir = join(dirname(realpath('__file__')), 'algorithms', 'copy', 'ml_data', 'agent_1')
discount_factor = 0.975
policy_args = {'epsilon': 0.25, 'random_seed': 0, 'min_penalty': -0.5}
policy = EpsilonGreedy(policy_args)
algorithm_args = {'discount_factor': 0.975, 'enable_e_traces': True, 'lambda_val': 1.0, 'enable_regressor': False,
                  'policy': policy, 'algorithm_name': AlgorithmName.EXPECTED_SARSA}
algorithm = TDAlgorithm(algorithm_args)
network_type = NetworkType.SINGLE

iterator = TargetValuePredictor(csv_dir, state_dim, algorithm, network_type)
print(iterator.method.__class__.__name__)
print(iterator.predict(4, 4))
