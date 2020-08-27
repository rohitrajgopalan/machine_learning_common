import enum
import os
from datetime import datetime
from os import mkdir
from os.path import join, isdir, isfile

import numpy as np
import pandas as pd

from neural_network.network_types import NetworkOptimizer
from reinforcement_learning.agent.ddpg_agent import DDPGAgent
from reinforcement_learning.agent.learning_type import LearningType
from reinforcement_learning.agent.td_agent import TDAgent
from reinforcement_learning.algorithm.td_algorithm import TDAlgorithm, TDAlgorithmName
from reinforcement_learning.policy.choose_policy import choose_policy
from reinforcement_learning.replay.buffer_type import BufferType


def create_boolean_list(chosen_types, key):
    chosen_options = []
    if key in chosen_types:
        if chosen_types[key].lower() == 'yes':
            chosen_options = [True]
        elif chosen_types[key].lower() == 'no':
            chosen_options = [False]
        elif chosen_types[key].lower() == 'both':
            chosen_options = [False, True]
    else:
        chosen_options = [False, True]
    return chosen_options


def choose_from_options(all_possible_options, chosen_types, key):
    chosen_options = []

    if key in chosen_types:
        if type(chosen_types[key]) == str:
            if chosen_types[key].lower() == 'all':
                chosen_options = all_possible_options
            elif chosen_types[key].lower() in all_possible_options:
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
                    if some_type.name.lower() == chosen_types[key].lower():
                        chosen_options = [some_type]
                        break
        elif type(chosen_types[key]) == list and len(chosen_types[key]) > 0:
            chosen_options = []
            for item in chosen_types[key]:
                for some_type in all_possible_options:
                    if (type(item) == str and some_type.name.lower() == item.lower()) or type(
                            item) == enum.Enum and item == some_type:
                        chosen_options.append(some_type)
                        break
        elif chosen_types[key] in all_possible_options:
            chosen_options = [chosen_types[key]]
    else:
        chosen_options = all_possible_options

    return chosen_options


class Experiment:
    hyper_parameters_data = None
    td_agent_cols = ['RUN_ID', 'NUM_EPISODES', 'ALGORITHM', 'POLICY', 'HYPER_PARAMETER', 'GAMMA',
                     'CONVERT_IMAGES_TO_GRAYSCALE',
                     'ADD_POOLING_FOR_IMAGES', 'ENABLE_DOUBLE_LEARNING',
                     'ENABLE_ACTION_BLOCKING', 'ACTION_BLOCKING_HELPER', 'OPTIMIZER',
                     'ALPHA', 'BETA_V', 'BETA_M', 'EPSILON', 'LEARNING_TYPE', 'REPLAY_TYPE', 'NUM_REPLAY',
                     'BUFFER_SIZE',
                     'MINI_BATCH_SIZE',
                     'AVG_TIME_STEP', 'MAX_TIME_STEP', 'AVG_RUNTIME', 'MAX_RUNTIME', 'NUM_COMPLETED', 'NUM_OUT_OF_TIME']
    ddpg_agent_cols = ['RUN_ID', 'NUM_EPISODES', 'CONVERT_IMAGES_TO_GRAYSCALE', 'ADD_POOLING_FOR_IMAGES', 'GAMMA',
                       'TAU',
                       'ENABLE_NOISE', 'ENABLE_ACTION_BLOCKING', 'ACTION_BLOCKING_HELPER', 'OPTIMIZER',
                       'ALPHA', 'BETA_V', 'BETA_M', 'EPSILON', 'LEARNING_TYPE', 'REPLAY_TYPE', 'NUM_REPLAY',
                       'BUFFER_SIZE',
                       'MINI_BATCH_SIZE',
                       'AVG_TIME_STEP', 'MAX_TIME_STEP', 'AVG_RUNTIME', 'MAX_RUNTIME', 'NUM_COMPLETED',
                       'NUM_OUT_OF_TIME']
    ac_agent_cols = ['RUN_ID', 'NUM_EPISODES', 'ALGORITHM', 'GAMMA', 'USE_TD_ERROR_AS_INPUT',
                     'CONVERT_IMAGES_TO_GRAYSCALE',
                     'ADD_POOLING_FOR_IMAGES',
                     'ENABLE_ACTION_BLOCKING', 'ACTION_BLOCKING_HELPER', 'OPTIMIZER',
                     'ALPHA', 'BETA_V', 'BETA_M', 'EPSILON', 'LEARNING_TYPE', 'REPLAY_TYPE', 'NUM_REPLAY',
                     'BUFFER_SIZE',
                     'MINI_BATCH_SIZE',
                     'AVG_TIME_STEP', 'MAX_TIME_STEP', 'AVG_RUNTIME', 'MAX_RUNTIME', 'NUM_COMPLETED',
                     'NUM_OUT_OF_TIME']
    agent_cols = ['RUN_ID', 'AGENT_ID', 'TOTAL_REWARD', 'NUM_UPDATE_STEPS', 'NUM_NEGATIVE_ACTIONS_TAKEN',
                  'FINAL_POLICY_FILE',
                  'ACTIONS_FILE']
    agents_data = None
    action_cols = ['ID', 'ACTION', 'TYPE']
    output_dir = ''
    dt_str = ''
    current_run_ID = 0

    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.current_run_ID = 0

    def process_run(self, run_info, agents, run_times, time_steps, num_completed, num_out_of_time):
        self.current_run_ID += 1
        hyper_parameter_val = 0
        is_td_agent = len([agent for agent in agents if type(agent) == TDAgent]) > 0
        is_ddpg_agent = len([agent for agent in agents if type(agent) == DDPGAgent]) > 0
        new_data = {'RUN_ID': self.current_run_ID,
                    'NUM_EPISODES': run_info['num_episodes'],
                    'CONVERT_IMAGES_TO_GRAYSCALE': 'Yes' if run_info['convert_to_grayscale'] else 'No',
                    'ADD_POOLING_FOR_IMAGES': 'Yes' if run_info['add_pooling'] else 'No',
                    'LEARNING_TYPE': run_info['learning_type'].name,
                    'REPLAY_TYPE': run_info['replay_type'].name,
                    'ENABLE_ACTION_BLOCKING': 'Yes' if run_info[
                        'enable_action_blocking'] else 'No',
                    'ACTION_BLOCKING_HELPER': 'Scikit-Learn' if
                    run_info[
                        'action_blocking_dl_args'] is None else 'Deep-Learning',
                    'NUM_REPLAY': run_info[
                        'num_replay'] if 'num_replay' in run_info else 0,
                    'BUFFER_SIZE': run_info[
                        'buffer_size'] if 'buffer_size' in run_info else 0,
                    'MINI_BATCH_SIZE': run_info[
                        'mini_batch_size'] if 'mini_batch_size' in run_info else 0,
                    'MAX_RUNTIME': np.max(run_times),
                    'AVG_RUNTIME': np.mean(run_times),
                    'MAX_TIME_STEP': np.max(time_steps),
                    'AVG_TIME_STEP': np.mean(time_steps),
                    'NUM_COMPLETED': num_completed,
                    'NUM_OUT_OF_TIME': num_out_of_time}
        if is_td_agent:
            if run_info['policy_name'] == 'epsilon_greedy':
                hyper_parameter_val = run_info['policy_args']['epsilon']
            elif run_info['policy_name'] == 'softmax':
                hyper_parameter_val = run_info['policy_args']['tau']
            elif run_info['policy_name'] == 'ucb':
                hyper_parameter_val = run_info['policy_args']['ucb_c']
            new_data.update({'ALGORITHM': run_info['algorithm_name'].name,
                             'ENABLE_DOUBLE_LEARNING': 'Yes' if run_info['enable_double_learning'] else 'No',
                             'POLICY': run_info['policy_name'],
                             'HYPER_PARAMETER': hyper_parameter_val,
                             'GAMMA': run_info['algorithm_args'][
                                 'discount_factor'],
                             'OPTIMIZER': run_info['action_network_args'][
                                 'optimizer_type'].name,
                             'ALPHA': run_info['action_network_args'][
                                 'optimizer_args']['learning_rate'],
                             'BETA_V': run_info['action_network_args'][
                                 'optimizer_args']['beta_v'],
                             'BETA_M': run_info['action_network_args'][
                                 'optimizer_args']['beta_m'],
                             'EPSILON': run_info['action_network_args'][
                                 'optimizer_args']['epsilon'],
                             })
        elif is_ddpg_agent:
            new_data.update({'GAMMA': run_info['discount_factor'],
                             'ENABLE_NOISE': 'Yes' if run_info['enable_noise'] else 'No',
                             'OPTIMIZER': run_info['actor_network_args']['optimizer_type'].name,
                             'ALPHA': run_info['optimizer_args']['learning_rate'],
                             'BETA_V': run_info['optimizer_args']['beta_v'],
                             'BETA_M': run_info['optimizer_args']['beta_m'],
                             'EPSILON': run_info['optimizer_args']['epsilon'],
                             'TAU': run_info['tau']
                             })
        else:
            new_data.update({'ALGORITHM': run_info['algorithm_name'].name,
                             'USE_TD_ERROR_AS_INPUT': 'Yes' if run_info['use_td_error_as_input'] else 'No',
                             'OPTIMIZER': run_info['actor_network_args']['optimizer_type'].name,
                             'GAMMA': run_info['algorithm_args']['discount_factor'],
                             'ALPHA': run_info['optimizer_args']['learning_rate'],
                             'BETA_V': run_info['optimizer_args']['beta_v'],
                             'BETA_M': run_info['optimizer_args']['beta_m'],
                             'EPSILON': run_info['optimizer_args']['epsilon']
                             })
        self.hyper_parameters_data = self.hyper_parameters_data.append(
            new_data,
            ignore_index=True)

        actions_dir = join(self.output_dir, 'actions')
        if not isdir(actions_dir):
            mkdir(actions_dir)
        final_policy_dir = join(self.output_dir, 'final_policy')
        if not isdir(final_policy_dir):
            mkdir(final_policy_dir)
        for agent in agents:
            agent_folder = 'agent_{0}'.format(agent.agent_id)
            agent_actions_dir = join(actions_dir, agent_folder)
            if not isdir(agent_actions_dir):
                mkdir(agent_actions_dir)

            date_str = datetime.now().strftime("%Y%m%d%H%M%S")

            file_name = '{0}.csv'.format(date_str)
            num_tuples_in_actions = len([a for a in agent.actions if type(a) == tuple])
            num_arrays_in_actions = len([a for a in agent.actions if type(a) == np.ndarray])
            if num_arrays_in_actions > 0 and num_tuples_in_actions == 0:
                agent_action_file = join(agent_actions_dir, date_str)
                actions = [a for a in agent.actions if a != 'SAMPLE']
                actions = np.array([actions])
                np.save(agent_action_file, actions)
            else:
                agent_action_file = join(agent_actions_dir, file_name)
                agent_actions = pd.DataFrame(columns=self.action_cols)
                for i, action in enumerate(agent.actions):
                    if type(action) == str or type(action) == int:
                        action_as_list = [action]
                    else:
                        action_as_list = list(action)
                    agent_actions = agent_actions.append(
                        {'ID': i + 1, 'ACTION': ';'.join(str(x) for x in action_as_list),
                         'TYPE': action.__class__.__name__},
                        ignore_index=True)
                agent_actions.to_csv(agent_action_file, index=False)

            agent_final_policy_dir = join(final_policy_dir, agent_folder)
            agent_final_policy_file = join(agent_final_policy_dir, file_name)
            if not isdir(agent_final_policy_dir):
                mkdir(agent_final_policy_dir)
            total_reward, final_policy = agent.get_results()

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
                        state_val = state[i]
                        if type(state_val) == bool:
                            state_val = int(state_val)
                        new_data.update({'STATE_VAR{0}'.format(i + 1): state_val})
                for action in final_policy[state]:
                    new_data.update({'ACTION': action})
                    agent_final_policy = agent_final_policy.append(new_data, ignore_index=True)
            agent_final_policy.to_csv(agent_final_policy_file, index=False)
            self.agents_data = self.agents_data.append(
                {'RUN_ID': self.current_run_ID, 'AGENT_ID': agent.agent_id, 'TOTAL_REWARD': total_reward,
                 'NUM_UPDATE_STEPS': agent.n_update_steps, 'FINAL_POLICY_FILE': file_name,
                 'ACTIONS_FILE': file_name, 'NUM_NEGATIVE_ACTIONS_TAKEN': agent.num_negative_actions_taken},
                ignore_index=True)

    def perform_experiment_td(self, experimental_parameters, specifics):
        self.current_run_ID = 0
        self.dt_str = datetime.now().strftime("%Y%m%d%H%M%S")
        self.agents_data = pd.DataFrame(columns=self.agent_cols)
        self.hyper_parameters_data = pd.DataFrame(columns=self.td_agent_cols)
        # Gather specifics
        agents_info_list = specifics['agent_info_list']
        action_network_args = specifics['action_network_args']
        num_episodes = specifics['num_episodes']
        environment = specifics['environment']
        random_seed = specifics['seed'] if 'seed' in specifics else 0
        action_blocking_dl_args = specifics[
            'action_blocking_dl_args'] if 'action_blocking_dl_args' in specifics else None
        policy_network_args = specifics['policy_network_args'] if 'policy_network_args' in specifics else None
        run_info = {'environment': environment,
                    'num_episodes': num_episodes,
                    'random_seed': random_seed}
        # Process Experimental Parameters
        chosen_add_pooling_flags = create_boolean_list(experimental_parameters,
                                                       'add_pooling') if environment.are_states_images else [False]
        chosen_grayscale_flags = create_boolean_list(experimental_parameters,
                                                     'convert_to_grayscale') if environment.are_states_images else [
            False]
        chosen_replay_types = choose_from_enums(BufferType.all(), experimental_parameters, 'replay_types')
        chosen_learning_types = choose_from_enums(LearningType.all(), experimental_parameters, 'learning_types')
        chosen_algorithms = choose_from_enums(TDAlgorithmName.all(), experimental_parameters, 'algorithm_names')
        chosen_policies = choose_from_options(['epsilon_greedy', 'softmax', 'thompson_sampling', 'ucb', 'network'],
                                              experimental_parameters, 'policies')
        chosen_action_blockers = create_boolean_list(experimental_parameters, 'enable_action_blocking')
        chosen_double_agent_flags = create_boolean_list(experimental_parameters, 'enable_double_learning')
        chosen_optimizers = choose_from_enums(NetworkOptimizer.all(), experimental_parameters, 'optimizers')
        policy_hyper_parameters = experimental_parameters['policy_hyper_parameters']
        optimizer_hyper_parameters = experimental_parameters[
            'optimizer_hyper_parameters'] if 'optimizer_hyper_parameters' in experimental_parameters else {}
        replay_buffer_hyper_parameters = experimental_parameters[
            'replay_buffer_hyper_parameters'] if 'replay_buffer_hyper_parameters' in experimental_parameters else {}
        if 'gammas' not in experimental_parameters or len(experimental_parameters['gammas']) == 0:
            experimental_parameters['gammas'] = [1.0]
        if 'alphas' not in optimizer_hyper_parameters or len(optimizer_hyper_parameters['alphas']) == 0:
            optimizer_hyper_parameters['alphas'] = [0.001]
        if 'beta_ms' not in optimizer_hyper_parameters or len(optimizer_hyper_parameters['beta_ms']) == 0:
            optimizer_hyper_parameters['beta_ms'] = [0.9]
        if 'beta_vs' not in optimizer_hyper_parameters or len(optimizer_hyper_parameters['beta_vs']) == 0:
            optimizer_hyper_parameters['beta_vs'] = [0.99]
        if 'epsilons' not in optimizer_hyper_parameters or len(optimizer_hyper_parameters['epsilons']) == 0:
            optimizer_hyper_parameters['epsilons'] = [1e-07]
        for convert_to_grayscale in chosen_grayscale_flags:
            run_info['convert_to_grayscale'] = convert_to_grayscale
            for add_pooling in chosen_add_pooling_flags:
                run_info['add_pooling'] = add_pooling
                for optimizer_type in chosen_optimizers:
                    action_network_args.update({'optimizer_type': optimizer_type})
                    if action_blocking_dl_args is not None:
                        action_blocking_dl_args['optimizer_type'] = optimizer_type
                    if policy_network_args is not None:
                        policy_network_args['optimizer_type'] = optimizer_type
                    optimizer_args = {}
                    for learning_rate in optimizer_hyper_parameters['alphas']:
                        optimizer_args['learning_rate'] = learning_rate
                        for beta_m in optimizer_hyper_parameters['beta_ms']:
                            optimizer_args['beta_m'] = beta_m
                            for beta_v in optimizer_hyper_parameters['beta_vs']:
                                optimizer_args['beta_v'] = beta_v
                                for epsilon in optimizer_hyper_parameters['epsilons']:
                                    optimizer_args['epsilon'] = epsilon
                                    run_info['optimizer_args'] = optimizer_args
                                    action_network_args['optimizer_args'] = optimizer_args
                                    if action_blocking_dl_args is not None:
                                        action_blocking_dl_args['optimizer_args'] = optimizer_args
                                    if policy_network_args is not None:
                                        policy_network_args['optimizer_args'] = optimizer_args
                                    run_info.update({'action_network_args': action_network_args,
                                                     'action_blocking_dl_args': action_blocking_dl_args})
                                    for algorithm_name in chosen_algorithms:
                                        run_info['algorithm_name'] = algorithm_name
                                        algorithm_args = {'algorithm_name': algorithm_name}
                                        for discount_factor in experimental_parameters['gammas']:
                                            algorithm_args['discount_factor'] = discount_factor
                                            run_info['algorithm_args'] = algorithm_args
                                            for policy_name in chosen_policies:
                                                run_info['policy_name'] = policy_name
                                                policy_args = {'random_seed': random_seed,
                                                               'min_penalty': environment.min_penalty}
                                                if policy_name == 'network':
                                                    assert policy_network_args is not None
                                                    policy_args.update({'network_args': policy_network_args})
                                                policy_hyper_parameter_list = [0]
                                                hyper_parameter_type = ''
                                                if policy_name == 'epsilon_greedy':
                                                    policy_hyper_parameter_list = policy_hyper_parameters[
                                                        'epsilons']
                                                    hyper_parameter_type = 'epsilon'
                                                elif policy_name == 'softmax':
                                                    policy_hyper_parameter_list = policy_hyper_parameters['taus']
                                                    hyper_parameter_type = 'tau'
                                                elif policy_name == 'ucb':
                                                    policy_hyper_parameter_list = policy_hyper_parameters[
                                                        'confidence_factors']
                                                    hyper_parameter_type = 'ucb_c'
                                                for policy_hyper_parameter in policy_hyper_parameter_list:
                                                    if len(hyper_parameter_type) > 0:
                                                        policy_args[hyper_parameter_type] = policy_hyper_parameter
                                                    for enable_action_blocking in chosen_action_blockers:
                                                        run_info['enable_action_blocking'] = enable_action_blocking
                                                        for learning_type in chosen_learning_types:
                                                            run_info['learning_type'] = learning_type
                                                            for is_double_agent in chosen_double_agent_flags:
                                                                run_info['enable_double_learning'] = is_double_agent
                                                                agents = []
                                                                for i, agent_info in enumerate(agents_info_list):
                                                                    agent_info.update({'agent_id': i + 1,
                                                                                       'is_double_agent': is_double_agent,
                                                                                       'learning_type': learning_type,
                                                                                       'state_dim': environment.required_state_dim,
                                                                                       'action_dim': environment.required_action_dim,
                                                                                       'enable_action_blocking': enable_action_blocking})
                                                                    policy_args.update(
                                                                        {'num_actions': len(agent_info['actions'])})
                                                                    run_info['policy_args'] = policy_args
                                                                    policy = choose_policy(policy_name, policy_args)
                                                                    algorithm_args['policy'] = policy
                                                                    algorithm = TDAlgorithm(algorithm_args)
                                                                    agent_info.update({'algorithm': algorithm})
                                                                    agents.append(TDAgent(agent_info))

                                                                run_info['agents'] = agents
                                                                if learning_type in [LearningType.REPLAY,
                                                                                     LearningType.OFF_POLICY,
                                                                                     LearningType.COMBINED]:
                                                                    for replay_type in chosen_replay_types:
                                                                        run_info['replay_type'] = replay_type
                                                                        for num_replay in replay_buffer_hyper_parameters['num_replay']:
                                                                            run_info['num_replay'] = num_replay
                                                                            for buffer_size in replay_buffer_hyper_parameters['buffer_size']:
                                                                                run_info['buffer_size'] = buffer_size
                                                                                for mini_batch_size in replay_buffer_hyper_parameters['mini_batch_size']:
                                                                                    run_info['mini_batch_size'] = mini_batch_size
                                                                                    agents, run_times, time_steps, num_completed, num_out_of_time = self.perform_run(
                                                                                        run_info)
                                                                                    self.process_run(run_info,
                                                                                                     agents,
                                                                                                     run_times,
                                                                                                     time_steps,
                                                                                                     num_completed,
                                                                                                     num_out_of_time)
                                                                elif learning_type == LearningType.ONLINE:
                                                                    run_info['replay_type'] = BufferType.BASIC
                                                                    agents, run_times, time_steps, num_completed, num_out_of_time = self.perform_run(
                                                                        run_info)
                                                                    self.process_run(run_info, agents, run_times,
                                                                                     time_steps, num_completed,
                                                                                     num_out_of_time)

        self.hyper_parameters_data.to_csv(
            '{0}'.format(os.path.join(self.output_dir, 'td_run_summary_{0}.csv'.format(self.dt_str))), index=False)
        self.agents_data.to_csv(
            '{0}'.format(os.path.join(self.output_dir, 'td_agents_data_{0}.csv'.format(self.dt_str))),
            index=False)

    def perform_experiment_ddpg(self, experimental_parameters, specifics):
        self.current_run_ID = 0
        self.dt_str = datetime.now().strftime("%Y%m%d%H%M%S")
        self.agents_data = pd.DataFrame(columns=self.agent_cols)
        self.hyper_parameters_data = pd.DataFrame(columns=self.ddpg_agent_cols)
        agents_info_list = specifics['agent_info_list']
        actor_network_args = {}
        critic_network_args = {}
        if 'actor_network_args' in specifics and 'critic_network_args' in specifics:
            actor_network_args = specifics['actor_network_args']
            critic_network_args = specifics['critic_network_args']
        elif 'network_args' in specifics:
            actor_network_args = specifics['network_args']
            critic_network_args = specifics['network_args']
        num_episodes = specifics['num_episodes']
        environment = specifics['environment']
        random_seed = specifics['seed'] if 'seed' in specifics else 0
        action_blocking_dl_args = specifics[
            'action_blocking_dl_args'] if 'action_blocking_dl_args' in specifics else None
        run_info = {'environment': environment,
                    'num_episodes': num_episodes,
                    'random_seed': random_seed}
        # Process Experimental Parameters
        chosen_add_pooling_flags = create_boolean_list(experimental_parameters,
                                                       'add_pooling') if environment.are_states_images else [False]
        chosen_grayscale_flags = create_boolean_list(experimental_parameters,
                                                     'convert_to_grayscale') if environment.are_states_images else [
            False]
        chosen_optimizers = choose_from_enums(NetworkOptimizer.all(), experimental_parameters, 'optimizers')
        chosen_replay_types = choose_from_enums(BufferType.all(), experimental_parameters, 'replay_types')
        chosen_learning_types = choose_from_enums(LearningType.all(), experimental_parameters, 'learning_types')
        chosen_action_blockers = create_boolean_list(experimental_parameters, 'enable_action_blocking')
        chosen_enable_noise_flags = create_boolean_list(experimental_parameters, 'enable_noise')
        optimizer_hyper_parameters = experimental_parameters[
            'optimizer_hyper_parameters'] if 'optimizer_hyper_parameters' in experimental_parameters else {}
        replay_buffer_hyper_parameters = experimental_parameters[
            'replay_buffer_hyper_parameters'] if 'replay_buffer_hyper_parameters' in experimental_parameters else {}
        if 'gammas' not in experimental_parameters or len(experimental_parameters['gammas']) == 0:
            experimental_parameters['gammas'] = [1.0]
        if 'alphas' not in optimizer_hyper_parameters or len(optimizer_hyper_parameters['alphas']) == 0:
            optimizer_hyper_parameters['alphas'] = [0.001]
        if 'beta_ms' not in optimizer_hyper_parameters or len(optimizer_hyper_parameters['beta_ms']) == 0:
            optimizer_hyper_parameters['beta_ms'] = [0.9]
        if 'beta_vs' not in optimizer_hyper_parameters or len(optimizer_hyper_parameters['beta_vs']) == 0:
            optimizer_hyper_parameters['beta_vs'] = [0.99]
        if 'epsilons' not in optimizer_hyper_parameters or len(optimizer_hyper_parameters['epsilons']) == 0:
            optimizer_hyper_parameters['epsilons'] = [1e-07]
        if 'taus' not in experimental_parameters or len(experimental_parameters['taus']) == 0:
            experimental_parameters['taus'] = [0.4]
        for convert_to_grayscale in chosen_grayscale_flags:
            run_info['convert_to_grayscale'] = convert_to_grayscale
            actor_network_args['convert_to_grayscale'] = convert_to_grayscale
            critic_network_args['convert_to_grayscale'] = convert_to_grayscale
            for add_pooling in chosen_add_pooling_flags:
                run_info['add_pooling'] = add_pooling
                actor_network_args['add_pooling'] = add_pooling
                critic_network_args['add_pooling'] = add_pooling
                for tau in experimental_parameters['taus']:
                    actor_network_args['tau'] = tau
                    critic_network_args['tau'] = tau
                    run_info['tau'] = tau
                    for optimizer_type in chosen_optimizers:
                        actor_network_args['optimizer_type'] = optimizer_type
                        critic_network_args['optimizer_type'] = optimizer_type
                        if action_blocking_dl_args is not None:
                            action_blocking_dl_args['optimizer_type'] = optimizer_type
                        optimizer_args = {}
                        for learning_rate in optimizer_hyper_parameters['alphas']:
                            optimizer_args['learning_rate'] = learning_rate
                            for beta_m in optimizer_hyper_parameters['beta_ms']:
                                optimizer_args['beta_m'] = beta_m
                                for beta_v in optimizer_hyper_parameters['beta_vs']:
                                    optimizer_args['beta_v'] = beta_v
                                    for epsilon in optimizer_hyper_parameters['epsilons']:
                                        optimizer_args['epsilon'] = epsilon
                                        actor_network_args['optimizer_args'] = optimizer_args
                                        critic_network_args['optimizer_args'] = optimizer_args
                                        run_info['optimizer_args'] = optimizer_args
                                        if action_blocking_dl_args is not None:
                                            action_blocking_dl_args['optimizer_args'] = optimizer_args
                                        run_info.update({'actor_network_args': actor_network_args,
                                                         'critic_network_args': critic_network_args,
                                                         'action_blocking_dl_args': action_blocking_dl_args})
                                        for enable_action_blocking in chosen_action_blockers:
                                            run_info['enable_action_blocking'] = enable_action_blocking
                                            for learning_type in chosen_learning_types:
                                                run_info['learning_type'] = learning_type
                                                for enable_noise in chosen_enable_noise_flags:
                                                    run_info['enable_noise'] = enable_noise
                                                    for gamma in experimental_parameters['gammas']:
                                                        run_info['discount_factor'] = gamma
                                                        agents = []
                                                        for i, agent_info in enumerate(agents_info_list):
                                                            agent_info.update({'agent_id': i + 1,
                                                                               'enable_noise': enable_noise,
                                                                               'learning_type': learning_type,
                                                                               'discount_factor': gamma,
                                                                               'state_dim': environment.required_state_dim,
                                                                               'action_dim': environment.required_action_dim,
                                                                               'enable_action_blocking': enable_action_blocking})
                                                            agents.append(DDPGAgent(agent_info))
                                                        run_info['agents'] = agents
                                                        if learning_type in [LearningType.REPLAY,
                                                                             LearningType.OFF_POLICY,
                                                                             LearningType.COMBINED]:
                                                            for replay_type in chosen_replay_types:
                                                                run_info['replay_type'] = replay_type
                                                                for num_replay in replay_buffer_hyper_parameters['num_replay']:
                                                                    run_info['num_replay'] = num_replay
                                                                    for buffer_size in replay_buffer_hyper_parameters['buffer_size']:
                                                                        run_info['buffer_size'] = buffer_size
                                                                        for mini_batch_size in replay_buffer_hyper_parameters['mini_batch_size']:
                                                                            run_info[
                                                                                'mini_batch_size'] = mini_batch_size
                                                                            agents, run_times, time_steps, num_completed, num_out_of_time = self.perform_run(
                                                                                run_info)
                                                                            self.process_run(run_info, agents,
                                                                                             run_times,
                                                                                             time_steps, num_completed,
                                                                                             num_out_of_time)
                                                        elif learning_type == LearningType.ONLINE:
                                                            run_info['replay_type'] = BufferType.BASIC
                                                            agents, run_times, time_steps, num_completed, num_out_of_time = self.perform_run(
                                                                run_info)
                                                            self.process_run(run_info, agents, run_times, time_steps,
                                                                             num_completed, num_out_of_time)

        self.hyper_parameters_data.to_csv(
            '{0}'.format(os.path.join(self.output_dir, 'ddpg_run_summary_{0}.csv'.format(self.dt_str))), index=False)
        self.agents_data.to_csv(
            '{0}'.format(os.path.join(self.output_dir, 'ddpg_agents_data_{0}.csv'.format(self.dt_str))),
            index=False)

    def perform_run(self, run_info):
        agents = run_info['agents']
        random_seed = run_info['random_seed'] if 'random_seed' in run_info else None
        learning_type = run_info['learning_type'] if 'learning_type' in run_info else LearningType.ONLINE
        environment = run_info['environment']

        ml_data_dir = join(self.output_dir, 'ml_data')
        if not isdir(ml_data_dir):
            mkdir(ml_data_dir)

        samples_dir = join(self.output_dir, 'samples')
        if not isdir(samples_dir):
            mkdir(samples_dir)

        enable_action_blocking = run_info['enable_action_blocking'] if 'enable_action_blocking' in run_info else False
        action_blocking_dl_args = run_info['action_blocking_dl_args'] if 'action_blocking_dl_args' in run_info else None

        for agent in agents:
            agent_ml_data_dir = join(ml_data_dir, 'agent_{0}'.format(agent.agent_id))
            if not isdir(agent_ml_data_dir):
                mkdir(agent_ml_data_dir)

            agent_samples_dir = join(samples_dir, 'agent_{0}'.format(agent.agent_id))
            if not isdir(agent_samples_dir):
                mkdir(agent_samples_dir)

            if learning_type in [LearningType.REPLAY, LearningType.OFF_POLICY, LearningType.COMBINED]:
                agent.buffer_init(run_info['replay_type'], run_info['num_replay'], run_info['buffer_size'],
                                  run_info['mini_batch_size'],
                                  random_seed)
            elif learning_type == LearningType.ONLINE:
                agent.buffer_init(BufferType.BASIC, 1, 1, 1, None)
            if enable_action_blocking:
                agent.blocker_init(csv_dir=agent_ml_data_dir, dl_args=action_blocking_dl_args)
            if type(agent) == TDAgent:
                action_network_args = run_info['action_network_args']
                agent.network_init(action_network_args=action_network_args)
            elif type(agent) == DDPGAgent:
                agent.actor_network_init(run_info['actor_network_args'])
                agent.critic_network_init(run_info['critic_network_args'])
                agent.exploration_noise_init(
                    run_info['exploration_noise_args'] if 'exploration_noise_args' in run_info else {})

        if learning_type == LearningType.OFF_POLICY:
            run_info['num_episodes'] = 1
            time_steps = np.zeros(1)
            run_times = np.zeros(1)
            start = datetime.now()
            for agent in agents:
                agent.load_samples()
            end = datetime.now()
            diff = end - start
            run_times[0] = diff.seconds
            return agents, run_times, time_steps
        else:
            environment.set_agents(agents)
            num_episodes = run_info['num_episodes']
            time_steps = np.zeros(num_episodes)
            run_times = np.zeros(num_episodes)
            num_completed = 0
            num_out_of_time = 0

            for episode in range(num_episodes):
                environment.reset()
                done = False
                value = 0
                start = datetime.now()
                while not done:
                    done, value = environment.step()

                if value == 1:
                    num_completed += 1
                elif value == 2:
                    num_out_of_time += 1

                time_steps[episode] = environment.current_time_step

                end = datetime.now()
                diff = end - start
                run_times[episode] = diff.seconds

                for agent in environment.agents:
                    agent_ml_data_dir = join(ml_data_dir, 'agent_{0}'.format(agent.agent_id))
                    file_name = join(agent_ml_data_dir,
                                     'log{0}_run{1}_episode{2}.csv'.format(datetime.now().strftime("%Y%m%d%H%M%S"),
                                                                           self.current_run_ID + 1,
                                                                           episode + 1))
                    if isfile(file_name):
                        file_name = join(file_name, 'duplicate')
                    agent.action_blocking_data.to_csv(
                        file_name,
                        index=False)

                    if not learning_type == LearningType.OFF_POLICY:
                        agent_samples_dir = join(samples_dir, 'agent_{0}'.format(agent.agent_id))
                        file_name = join(agent_samples_dir,
                                         'log{0}_run{1}_episode{2}.csv'.format(datetime.now().strftime("%Y%m%d%H%M%S"),
                                                                               self.current_run_ID + 1,
                                                                               episode + 1))
                        if isfile(file_name):
                            file_name = join(file_name, 'duplicate')
                        agent.experienced_samples.to_csv(file_name,
                                                         index=False)

            return environment.agents, run_times, time_steps, num_completed, num_out_of_time
