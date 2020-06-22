from datetime import datetime
from os import mkdir
from os.path import join, isdir

import numpy as np
import pandas as pd

from reinforcement_learning.agent.agent import Agent
from reinforcement_learning.agent.agent import LearningType
from reinforcement_learning.algorithm.base_algorithm import AlgorithmName
from reinforcement_learning.algorithm.choose_algorithm import choose_algorithm
from reinforcement_learning.experiment.run import run
from reinforcement_learning.network.actionvaluenetwork import NetworkType, NetworkInitializationType, \
    NetworkActivationFunction
from reinforcement_learning.policy.choose_policy import choose_policy

cols = ['LEARNING_TYPE', 'ALGORITHM', 'POLICY', 'HYPER_PARAMETER', 'ALPHA', 'GAMMA',
        'NETWORK_TYPE', 'NETWORK_INITIALIZER', 'ACTIVATION_FUNCTION', 'NETWORK_ALPHA', 'ENABLE_E_TRACES', 'LAMBDA',
        'ENABLE_ACTION_BLOCKING', 'ENABLE_REGRESSOR', 'AVG_TIMESTEP',
        'MAX_TIMESTEP', 'AVG_RUNTIME', 'MAX_RUNTIME']
hyper_parameters_data = None
agent_cols = ['AGENT_ID', 'TOTAL_REWARD', 'NUM_UPDATE_STEPS', 'FINAL_POLICY_FILE', 'ACTIONS_FILE']
agents_data = None
action_cols = ['ID', 'ACTION', 'TYPE']


def process_run(run_info, agents, runtimes, timesteps):
    global hyper_parameters_data
    hyper_parameter_val = 0
    if run_info['policy_name'] == 'epsilon_greedy':
        hyper_parameter_val = run_info['policy_args']['epsilon']
    elif run_info['policy_name'] == 'softmax':
        hyper_parameter_val = run_info['policy_args']['tau']
    elif run_info['policy_name'] == 'ucb':
        hyper_parameter_val = run_info['policy_args']['ucb_c']
    hyper_parameters_data = hyper_parameters_data.append({'LEARNING_TYPE': run_info['learning_type'].name,
                                                          'ALGORITHM': run_info['algorithm_name'],
                                                          'POLICY': run_info['policy_name'],
                                                          'HYPER_PARAMETER': hyper_parameter_val,
                                                          'ALPHA': run_info['learning_rate'],
                                                          'GAMMA': run_info['algorithm_args']['discount_factor'],
                                                          'NETWORK_TYPE': run_info['network_type'].name,
                                                          'NETWORK_INTIALIZER': run_info['network_initializer'].name,
                                                          'ACTIVATION_FUNCTION': run_info['activation_function'].name,
                                                          'NETWORK_ALPHA': run_info['network_alpha'],
                                                          'ENABLE_E_TRACES': 'Yes' if run_info['algorithm_args'][
                                                              'enable_e_traces'] else 'No',
                                                          'LAMBDA': run_info['algorithm_args']['lambda_val'] if
                                                          run_info['algorithm_args'][
                                                              'enable_e_traces'] else 0.0,
                                                          'ENABLE_ACTION_BLOCKING': 'Yes' if run_info[
                                                              'enable_action_blocking'] else 'No',
                                                          'ENABLE_REGRESSOR': 'Yes' if run_info['algorithm_args'][
                                                              'enable_regressor'] else 'No',
                                                          'MAX_RUNTIME': np.max(runtimes),
                                                          'AVG_RUNTIME': np.mean(runtimes),
                                                          'MAX_TIMESTEP': np.max(timesteps),
                                                          'AVG_TIMESTEP': np.mean(timesteps)}, ignore_index=True)

    output_dir = run_info['output_dir']
    actions_dir = join(output_dir, 'actions')
    final_policy_dir = join(output_dir, 'final_policy')

    global agents_data
    for agent in agents:
        agent_folder = 'agent_{0}'.format(agent.agent_id)
        agent_actions_dir = join(actions_dir, agent_folder)
        if not isdir(agent_actions_dir):
            mkdir(agent_actions_dir)

        file_name = "{0}.csv".format(datetime.now().strftime("%Y%m%d%H%M%S"))
        agent_action_file = join(agent_actions_dir, file_name)
        agent_actions = pd.DataFrame(columns=action_cols)
        for i, action in enumerate(agent.actions):
            if type(action) == str or type(action) == int:
                action_as_list = [action]
            else:
                action_as_list = list(action)
            agent_actions = agent_actions.append(
                {'ID': i + 1, 'ACTION': ';'.join(str(x) for x in action_as_list), 'TYPE': action.__class__.__name__},
                ignore_index=True)
        agent_actions.to_csv(agent_action_file, index=False)

        agent_final_policy_dir = join(final_policy_dir, agent_folder)
        agent_final_policy_file = join(agent_final_policy_dir, file_name)
        if not isdir(agent_final_policy_dir):
            mkdir(agent_final_policy_dir)
        final_policy = agent.determine_final_policy()

        agent_final_policy_cols = []
        if agent.state_dim == 1:
            agent_final_policy_cols.append('STATE')
        else:
            for i in range(1, agent.state_dim + 1):
                agent_final_policy_cols.append('STATE_VAR{0}'.format(i))
        agent_final_policy_cols.append('ACTION')
        agent_final_policy = pd.DataFrame(columns=agent_final_policy_cols)
        for state in final_policy:
            new_data = {}
            if agent.state_dim == 1:
                new_data.update({'STATE': state})
            else:
                for i in range(agent.state_dim):
                    new_data.update({'STATE_VAR{0}'.format(i): state[i]})
            for action in final_policy[state]:
                new_data.update({'ACTION': action})
                agent_final_policy = agent_final_policy.append(new_data, ignore_index=True)
        agent_final_policy.to_csv(agent_final_policy_file, index=False)

        agents_data = agents_data.append({'AGENT_ID': agent.agent_id, 'TOTAL_REWARD': agent.get_total_reward(),
                                          'NUM_UPDATE_STEPS': agent.n_update_steps, 'FINAL_POLICY_FILE': file_name,
                                          'ACTIONS_FILE': file_name}, ignore_index=True)


def generate_agent(agent_id, agent_info, state_dim, learning_type, policy_name, policy_args, algorithm_name,
                   algorithm_args, enable_action_blocking):
    policy_args.update({'num_actions': len(agent_info['actions'])})
    policy = choose_policy(policy_name, policy_args)

    if algorithm_name.endswith('lambda'):
        name = 'td_lambda'
    else:
        name = 'td'

    if algorithm_name.startswith('sarsa'):
        algorithm = AlgorithmName.SARSA
    elif algorithm_name.startswith('q'):
        algorithm = AlgorithmName.Q
    elif algorithm_name.startswith('expected_sarsa'):
        algorithm = AlgorithmName.EXPECTED_SARSA
    elif algorithm_name.startswith('mcq'):
        algorithm = AlgorithmName.MCQL
    else:
        algorithm = None

    algorithm_args.update({'policy': policy, 'algorithm_name': algorithm})

    algorithm = choose_algorithm(name, algorithm_args)
    agent_info.update(
        {'state_dim': state_dim, 'learning_type': learning_type, 'algorithm': algorithm, 'agent_id': agent_id,
         'enable_action_blocking': enable_action_blocking})
    return Agent(agent_info)


def choose_from_options(all_possible_options, chosen_types, key):
    chosen_options = []

    if key in chosen_types:
        if type(chosen_types[key]) == str:
            if chosen_types[key] == 'all':
                chosen_options = all_possible_options
            else:
                chosen_options = [chosen_types[key]]
        elif type(chosen_types[key]) == list and len(chosen_types[key]) > 0:
            chosen_options = chosen_types[key]
    else:
        chosen_options = all_possible_options

    return chosen_options


def choose_from_enums(all_possible_options, chosen_types, key):
    chosen_options = []

    if key in chosen_types:
        if type(chosen_types[key]) == str:
            if chosen_types[key] == 'all':
                chosen_options = all_possible_options
            else:
                for some_type in all_possible_options:
                    if some_type.name == chosen_types[key].upper():
                        chosen_options = [some_type]
                        break
        elif type(chosen_types[key]) == list and len(chosen_types[key]) > 0:
            chosen_options = chosen_types[key]
    else:
        chosen_options = all_possible_options

    return chosen_options


def create_boolean_list(chosen_types, key):
    chosen_options = []
    if key in chosen_types:
        if chosen_types[key] == 'yes':
            chosen_options = [True]
        elif chosen_types[key] == 'no':
            chosen_options = [False]
        elif chosen_types[key] == 'both':
            chosen_options = [False, True]
    else:
        chosen_options = [False, True]
    return chosen_options


def run_experiment(output_dir, environment, num_episodes, agents_info_list, chosen_types, policy_hyperparameters,
                   algorithm_hyperparameters,
                   network_hyperparameters=None,
                   replay_buffer_info=None):
    if network_hyperparameters is None:
        network_hyperparameters = {}
    global hyper_parameters_data
    hyper_parameters_data = pd.DataFrame(columns=cols)
    global agents_data
    agents_data = pd.DataFrame(columns=agent_cols)

    mkdir(join(output_dir, 'ml_data'))
    mkdir(join(output_dir, 'final_policy'))
    mkdir(join(output_dir, 'actions'))
    mkdir(join(output_dir, 'out'))

    if replay_buffer_info is None:
        replay_buffer_info = {}

    if chosen_types['learning_type'] == 'online':
        chosen_learning_type = LearningType.Online
    elif chosen_types['learning_type'] == 'replay':
        chosen_learning_type = LearningType.Replay

    run_info = {'environment': environment,
                'num_episodes': num_episodes,
                'learning_type': chosen_learning_type,
                'output_dir': output_dir,
                'random_seed': 0,
                'beta_m': 0.9,
                'beta_v': 0.99,
                'epsilon': 0.0001}

    chosen_algorithms = choose_from_options(['sarsa', 'sarsa_lambda', 'q', 'q_lambda', 'expected_sarsa',
                                             'expected_sarsa_lambda', 'mcql', 'mcq_lambda'], chosen_types,
                                            'algorithm_names')

    chosen_policies = choose_from_options(['epsilon_greedy', 'softmax', 'thompson_sampling', 'ucb'], chosen_types,
                                          'policies')
    chosen_e_traces_flags = create_boolean_list(chosen_types, 'enable_e_traces')

    if 'lambdas' not in algorithm_hyperparameters or len(algorithm_hyperparameters['lambdas']) == 0:
        algorithm_hyperparameters['lambdas'] = [0]

    chosen_network_types = choose_from_enums(NetworkType.all(), network_hyperparameters, 'network_types')
    chosen_network_initializers = choose_from_enums(NetworkInitializationType.all(), network_hyperparameters,
                                                    'network_initializers')
    chosen_activation_functions = choose_from_enums(NetworkActivationFunction.all(), network_hyperparameters,
                                                    'activation_functions')
    chosen_action_blockers = create_boolean_list(chosen_types, 'enable_action_blocking')
    chosen_enable_regressors = create_boolean_list(chosen_types, 'enable_regressors')

    if 'alphas' not in network_hyperparameters or len(network_hyperparameters['alphas']) == 0:
        network_hyperparameters['alphas'] = [0]

    for policy_name in chosen_policies:
        run_info['policy_name'] = policy_name
        policy_args = {'random_seed': policy_hyperparameters['seed'], 'min_penalty': environment.min_penalty}
        # either a list of taus, epsilons or confidence factors
        policy_hyperparameter_list = [0]
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
            if len(hyperparameter_type) > 0:
                policy_args[hyperparameter_type] = policy_hyperparameter

            run_info['policy_args'] = policy_args

            for learning_rate in algorithm_hyperparameters['alphas']:
                run_info['learning_rate'] = learning_rate
                for algorithm_name in chosen_algorithms:
                    run_info['algorithm_name'] = algorithm_name
                    algorithm_args = {}
                    for discount_factor in algorithm_hyperparameters['gammas']:
                        algorithm_args['discount_factor'] = discount_factor
                        for enable_e_traces in chosen_e_traces_flags:
                            algorithm_args['enable_e_traces'] = enable_e_traces
                            for lambda_val in algorithm_hyperparameters['lambdas']:
                                algorithm_args['lambda_val'] = lambda_val
                                for enable_regressor in chosen_enable_regressors:
                                    algorithm_args['enable_regressor'] = enable_regressor
                                    run_info['algorithm_args'] = algorithm_args
                                    for enable_action_blocking in chosen_action_blockers:
                                        run_info['enable_action_blocking'] = enable_action_blocking
                                        agents = []
                                        for i in range(len(agents_info_list)):
                                            agents.append(generate_agent(i + 1, agents_info_list[i],
                                                                         environment.required_state_dim,
                                                                         chosen_types['learning_type'], policy_name,
                                                                         policy_args, algorithm_name, algorithm_args,
                                                                         enable_action_blocking))

                                        run_info['agents'] = agents
                                        for network_type in chosen_network_types:
                                            run_info['network_type'] = network_type

                                            for initializer_type in chosen_network_initializers:
                                                run_info['initializer_type'] = initializer_type
                                                for activation_function in chosen_activation_functions:
                                                    run_info['activation_function'] = activation_function
                                                    for network_alpha in network_hyperparameters['alphas']:
                                                        run_info['network_alpha'] = network_alpha
                                                        if chosen_learning_type == LearningType.Replay:
                                                            for key in replay_buffer_info:
                                                                run_info[key] = replay_buffer_info[key]

                                                        agents, timesteps, runtimes = run(run_info)
                                                        process_run(run_info, agents, timesteps, runtimes)

    hyper_parameters_data.to_csv(join(output_dir, 'run_summary.csv'), index=False)
    agents_data.to_csv(join(output_dir, 'agents_data.csv'), index=False)
