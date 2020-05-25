from os.path import dirname, realpath

import numpy as np

from examples.fixedspace.fixedspaceenvironment import FixedSpaced
from reinforcement_learning.experiment.run_experiment import run_experiment

environment = FixedSpaced(4)
num_episodes = 1000

agent_info_list = [{'initial_state': (1, 1), 'actions': ['UP', 'DOWN', 'LEFT', 'RIGHT']}
                   , {'initial_state': (1, 4), 'actions': ['UP', 'DOWN', 'LEFT', 'RIGHT']}
                   , {'initial_state': (4, 1), 'actions': ['UP', 'DOWN', 'LEFT', 'RIGHT']}
                   , {'initial_state': (4, 4), 'actions': ['UP', 'DOWN', 'LEFT', 'RIGHT']}
                   ]

chosen_types = {'learning_type': 'online',
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

