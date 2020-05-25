from os.path import dirname, realpath
import numpy as np
from reinforcement_learning.experiment.run_experiment import run_experiment

from examples.maze.maze import MazeEnvironment

maze_dim = (6, 9)
obstacles = [(2, 3), (3, 3), (4, 3), (5, 6), (1, 8), (2, 8), (3, 8)]
final_location = (1, 9)
start_location = (3, 1)

environment = MazeEnvironment(maze_dim, obstacles, start_location, final_location)
num_episodes = 1000

agent_info_list = [{'initial_state': start_location, 'actions': ['UP', 'DOWN', 'LEFT', 'RIGHT']}]

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