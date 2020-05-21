from os import mkdir
from os.path import isdir, join

from reinforcement_learning.agent.base_agent import LearningType
from reinforcement_learning.agent.choose_agent import choose_agent
from reinforcement_learning.algorithm.choose_algorithm import choose_algorithm
from reinforcement_learning.experiment.run import run
from reinforcement_learning.network.actionvaluenetwork import NetworkType
from reinforcement_learning.policy.choose_policy import choose_policy


def generate_agent(agent_id, agent_name, agent_info, state_dim, learning_type, policy_name, policy_args, algorithm_name,
                   algorithm_args):
    policy_args.update({'num_actions': len(agent_info['actions'])})
    policy = choose_policy(policy_name, policy_args)
    algorithm_args.update({'policy': policy})
    algorithm = choose_algorithm(algorithm_name, algorithm_args)
    agent_info.update(
        {'state_dim': state_dim, 'learning_type': learning_type, 'algorithm': algorithm, 'agent_id': agent_id})
    return choose_agent(agent_name, agent_info)


def run_experiment(output_dir, environment, num_episodes, agents_info_list, chosen_types, policy_hyperparameters,
                   algorithm_hyperparameters,
                   replay_buffer_info=None):
    if replay_buffer_info is None:
        replay_buffer_info = {}
    output_dir = join(output_dir, 'out')
    if not isdir(output_dir):
        mkdir(output_dir)

    run_info = {'environment': environment,
                'num_episodes': num_episodes,
                'learning_type': chosen_types['learning_type'],
                'output_dir': output_dir,
                'num_hidden_units': 2,
                'random_seed': 0,
                'beta_m': 0.9,
                'beta_v': 0.99,
                'epsilon': 0.0001}

    all_policies = ['epsilon_greedy', 'softmax', 'thompson_sampling', 'ucb']

    all_agent_names = ['simple', 'dyna', 'dyna_plus', 'prioritized_sweeping']

    if 'agent_names' in chosen_types and len(chosen_types['agent_names']) > 0:
        if chosen_types['agent_names'][0] == 'all':
            chosen_agent_names = all_agent_names
        else:
            chosen_agent_names = chosen_types['agent_names']
    else:
        chosen_agent_names = all_agent_names

    if 'policies' in chosen_types and len(chosen_types['policies']) > 0:
        chosen_policies = chosen_types['policies']
    else:
        chosen_policies = all_policies

    for policy_name in chosen_policies:
        run_info['policy_name'] = policy_name
        policy_args = {'random_seed': policy_hyperparameters['seed']}
        # either a list of taus, epsilons or confidence factors
        policy_hyperparameter_list = []
        hyperparameter_type = ''
        if policy_name == 'epsilon_greedy':
            policy_hyperparameter_list = policy_hyperparameters['epsilons']
            hyperparameter_type = 'epsilon'
        elif policy_name == 'softmax':
            policy_hyperparameter_list = policy_hyperparameters['taus']
            hyperparameter_type = 'tau'
        elif policy_name == 'ucb':
            policy_hyperparameter_list = policy_hyperparameters['confidence_factors']
            hyperparameter_type = 'ucb_c'
        for policy_hyperparameter in policy_hyperparameter_list:
            policy_args[hyperparameter_type] = policy_hyperparameter
            run_info['policy_args'] = policy_args

            for learning_rate in algorithm_hyperparameters['alphas']:
                run_info['learning_rate'] = learning_rate
                for algorithm_name in chosen_types['algorithm_names']:
                    run_info['algorithm_name'] = algorithm_name
                    algorithm_args = {}
                    for discount_factor in algorithm_hyperparameters['gammas']:
                        algorithm_args['discount_factor'] = discount_factor
                        for lambda_val in algorithm_hyperparameters['lambdas']:
                            algorithm_args['lambda_val'] = lambda_val
                            for enable_e_traces in [True, False]:
                                algorithm_args['enable_e_traces'] = enable_e_traces
                                run_info['algorithm_args'] = algorithm_args

                                for agent_name in chosen_agent_names:
                                    run_info['agent_name'] = agent_name
                                    agents = []
                                    for i in range(len(agents_info_list)):
                                        agents.append(generate_agent(i + 1, agent_name, agents_info_list[i],
                                                                     environment.required_state_dim,
                                                                     chosen_types['learning_type'], policy_name,
                                                                     policy_args, algorithm_name, algorithm_args))

                                    run_info['agents'] = agents

                                    for network_type in [NetworkType.SINGLE, NetworkType.DOUBLE]:
                                        run_info['network_type'] = network_type

                                        if chosen_types['learning_type'] == LearningType.Replay:
                                            for key in replay_buffer_info:
                                                run_info[key] = replay_buffer_info[key]

                                        print(run_info)
                                        run(run_info)
