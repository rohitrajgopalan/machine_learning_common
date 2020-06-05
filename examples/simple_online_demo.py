from reinforcement_learning.experiment.run_experiment import run_experiment


def run_demo(environment, agent_info_list, output_dir):
    num_episodes = 1000

    chosen_types = {'learning_type': 'online',
                    'network_types': 'single',
                    'enable_e_traces': 'yes',
                    'enable_action_blocking': 'no',
                    'enable_regressors': 'no',
                    'algorithm_names': 'expected_sarsa'}

    policy_hyperparameters = {'seed': 0,
                              'taus': [1.0],
                              'epsilons': [0.25],
                              'confidence_factors': [2]}

    algorithm_hyperparamters = {'alphas': [0.05],
                                'gammas': [0.975],
                                'lambdas': [1.0]}

    run_experiment(output_dir, environment, num_episodes, agent_info_list, chosen_types,
                   policy_hyperparameters, algorithm_hyperparamters)
