import numpy as np

from reinforcement_learning.experiment.run_experiment import run_experiment


def run_demo(environment, agent_info_list, output_dir):
    num_episodes = 1000

    chosen_types = {'learning_type': 'online'}

    policy_hyperparameters = {'seed': 0,
                              'taus': [0.001, 0.01, 0.1, 1.0],
                              'epsilons': list(np.arange(0.1, 0.5, 0.1)),
                              'confidence_factors': [0.01, 0.1, 2]}

    algorithm_hyperparamters = {'alphas': list(np.arange(0.01, 0.10, 0.01)),
                                'gammas': list(np.arange(0.95, 1.00, 0.01)),
                                'lambdas': [1.0]}

    run_experiment(output_dir, environment, num_episodes, agent_info_list, chosen_types,
                   policy_hyperparameters, algorithm_hyperparamters)
