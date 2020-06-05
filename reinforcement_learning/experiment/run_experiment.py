from reinforcement_learning.agent.agent import LearningType
from reinforcement_learning.agent.agent import Agent
from reinforcement_learning.algorithm.base_algorithm import AlgorithmName
from reinforcement_learning.algorithm.choose_algorithm import choose_algorithm
from reinforcement_learning.experiment.run import run
from reinforcement_learning.network.actionvaluenetwork import NetworkType
from reinforcement_learning.policy.choose_policy import choose_policy


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
                   replay_buffer_info=None):
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
                'num_hidden_units': 2,
                'random_seed': 0,
                'beta_m': 0.9,
                'beta_v': 0.99,
                'epsilon': 0.0001}

    chosen_algorithms = choose_from_options(['sarsa', 'sarsa_lambda', 'q', 'q_lambda', 'expected_sarsa',
                                             'expected_sarsa_lambda', 'mcql', 'mcq_lambda'], chosen_types,
                                            'algorithm_names')

    chosen_policies = choose_from_options(['epsilon_greedy', 'softmax', 'thompson_sampling', 'ucb'], chosen_types, 'policies')
    chosen_e_traces_flags = create_boolean_list(chosen_types, 'enable_e_traces')

    if 'lambdas' not in algorithm_hyperparameters or len(algorithm_hyperparameters['lambdas']) == 0:
        algorithm_hyperparameters['lambdas'] = [0]

    chosen_network_types = []

    if 'network_types' in chosen_types:
        if chosen_types['network_types'] == 'all':
            chosen_network_types = [NetworkType.SINGLE, NetworkType.DOUBLE]
        elif chosen_types['network_types'] == 'single':
            chosen_network_types = [NetworkType.SINGLE]
        elif chosen_types['network_types'] == 'double':
            chosen_network_types = [NetworkType.DOUBLE]
    else:
        chosen_network_types = [NetworkType.SINGLE, NetworkType.DOUBLE]

    chosen_action_blockers = create_boolean_list(chosen_types, 'enable_action_blocking')
    chosen_enable_regressors = create_boolean_list(chosen_types, 'enable_regressors')

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

                                            if chosen_learning_type == LearningType.Replay:
                                                for key in replay_buffer_info:
                                                    run_info[key] = replay_buffer_info[key]

                                            # agents, timesteps, runtimes = run(run_info)
                                            run(run_info)
                                            # populate data with results
