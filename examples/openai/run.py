from examples.openai.openai import OpenAIGymEnvironment
from reinforcement_learning.experiment.run_experiment import run_experiment
import numpy as np


def run(output_dir,open_ai_gym_name):
    environment = OpenAIGymEnvironment(open_ai_gym_name)
    agent_info_list = [
        {'actions': list(range(environment.openAI.action_space.n)), 'initial_state': environment.openAI.reset()}]

    num_episodes = 1000

    chosen_types = {'learning_type': 'online',
                    'algorithm_names': ['sarsa', 'sarsa_lambda', 'q', 'q_lambda', 'expected_sarsa',
                                        'expected_sarsa_lambda', 'mcql', 'mcq_lambda']}

    policy_hyperparameters = {'seed': 0,
                              'taus': [0.001, 0.01, 0.1, 1.0],
                              'epsilons': list(np.arange(0.1, 0.5, 0.1)),
                              'confidence_factors': [0.01, 0.1, 2]}

    algorithm_hyperparameters = {'alphas': list(np.arange(0.01, 0.11, 0.01)),
                                'gammas': list(np.arange(0.95, 1.01, 0.01)),
                                'lambdas': [1.0]}

    run_experiment(output_dir, environment, num_episodes, agent_info_list, chosen_types, policy_hyperparameters,
                   algorithm_hyperparameters)
