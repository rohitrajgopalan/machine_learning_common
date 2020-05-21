from os.path import dirname, realpath

import numpy as np

from examples.cliffwalking.cliffwalking import CliffWalkingEnvironment
from reinforcement_learning.experiment.run_experiment import run_experiment

environment = CliffWalkingEnvironment(4, 4)
num_episodes = 5000

start_loc = (1, 1)
agent_info = {'start_loc': start_loc, 'actions': ['UP', 'DOWN', 'LEFT', 'RIGHT']}
agent_info_list = [agent_info]

chosen_types = {'learning_type': 'online',
                'agent_names': ['simple'],
                'algorithm_names': ['sarsa', 'sarsa_lambda']}

policy_hyperparameters = {'seed': 0,
                          'taus': [0.001, 0.01, 0.1, 1.0],
                          'epsilons': list(np.arange(0.1, 0.5, 0.1)),
                          'confidence_factors': [0.01, 0.1, 2]}

algorithm_hyperparamters = {'alphas': list(np.arange(0.01, 0.11, 0.01)),
                            'gammas': list(np.arange(0.95, 1.01, 0.01)),
                            'lambdas': [1.0]}

run_experiment(dirname(realpath('__file__')), environment, num_episodes, agent_info_list, chosen_types,
               policy_hyperparameters, algorithm_hyperparamters)
