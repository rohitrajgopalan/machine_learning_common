import enum
import os
from datetime import datetime
from os import mkdir
from os.path import join, isdir

import numpy as np
import pandas as pd

from neural_network.network_types import NetworkOptimizer
from reinforcement_learning.agent.agent import LearningType, Agent
from reinforcement_learning.algorithm.algorithm import Algorithm, AlgorithmName
from reinforcement_learning.policy.choose_policy import choose_policy


class Experiment:
    hyper_parameters_data = None
    cols = ['ALGORITHM', 'POLICY', 'HYPER_PARAMETER', 'GAMMA',
            'ENABLE_ACTION_BLOCKING', 'ACTION_BLOCKING_HELPER', 'ENABLE_ITERATOR', 'ITERATOR_HELPER', 'OPTIMIZER',
            'ALPHA', 'BETA_V', 'BETA_M', 'EPSILON', 'LEARNING_TYPE', 'NUM_REPLAY', 'BUFFER_SIZE', 'MINI_BATCH_SIZE',
            'AVG_TIME_STEP', 'MAX_TIME_STEP', 'AVG_RUNTIME', 'MAX_RUNTIME']
    agent_cols = ['AGENT_ID', 'TOTAL_REWARD', 'NUM_UPDATE_STEPS', 'FINAL_POLICY_FILE', 'ACTIONS_FILE']
    agents_data = None
    action_cols = ['ID', 'ACTION', 'TYPE']
    output_dir = ''
    dt_str = ''

    def __init__(self, output_dir):
        self.hyper_parameters_data = pd.DataFrame(columns=self.cols)
        self.output_dir = output_dir
        self.agents_data = pd.DataFrame(columns=self.agent_cols)
        self.dt_str = datetime.now().strftime("%Y%m%d%H%M%S")

    def choose_from_options(self, all_possible_options, chosen_types, key):
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

    def choose_from_enums(self, all_possible_options, chosen_types, key):
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
            elif type(chosen_types[key]) == enum.Enum:
                chosen_options = [chosen_types[key]]
        else:
            chosen_options = all_possible_options

        return chosen_options

    def create_boolean_list(self, chosen_types, key):
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

    def process_run(self, run_info, agents, run_times, time_steps):
        hyper_parameter_val = 0
        if run_info['policy_name'] == 'epsilon_greedy':
            hyper_parameter_val = run_info['policy_args']['epsilon']
        elif run_info['policy_name'] == 'softmax':
            hyper_parameter_val = run_info['policy_args']['tau']
        elif run_info['policy_name'] == 'ucb':
            hyper_parameter_val = run_info['policy_args']['ucb_c']
        self.hyper_parameters_data = self.hyper_parameters_data.append({'LEARNING_TYPE': run_info['learning_type'].name,
                                                                        'ALGORITHM': run_info['algorithm_name'].name,
                                                                        'POLICY': run_info['policy_name'],
                                                                        'HYPER_PARAMETER': hyper_parameter_val,
                                                                        'GAMMA': run_info['algorithm_args'][
                                                                            'discount_factor'],
                                                                        'ENABLE_ACTION_BLOCKING': 'Yes' if run_info[
                                                                            'enable_action_blocking'] else 'No',
                                                                        'ACTION_BLOCKING_HELPER': 'Scikit-Learn' if
                                                                        run_info[
                                                                            'action_blocking_dl_args'] is None else 'Deep-Learning',
                                                                        'ENABLE_ITERATOR': 'Yes' if
                                                                        run_info['algorithm_args'][
                                                                            'enable_iterator'] else 'No',
                                                                        'ITERATOR_HELPER': 'Scikit-Learn' if run_info[
                                                                                                                 'regression_dl_args'] is None else 'Deep-Learning',
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
                                                                        'NUM_REPLAY': run_info[
                                                                            'num_replay'] if 'num_replay' in run_info else 0,
                                                                        'BUFFER_SIZE': run_info[
                                                                            'buffer_size'] if 'buffer_size' in run_info else 0,
                                                                        'MINI_BATCH_SIZE': run_info[
                                                                            'mini_batch_size'] if 'mini_batch_size' in run_info else 0,
                                                                        'MAX_RUNTIME': np.max(run_times),
                                                                        'AVG_RUNTIME': np.mean(run_times),
                                                                        'MAX_TIME_STEP': np.max(time_steps),
                                                                        'AVG_TIME_STEP': np.mean(time_steps)},
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

            file_name = "{0}.csv".format(datetime.now().strftime("%Y%m%d%H%M%S"))
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
                        new_data.update({'STATE_VAR{0}'.format(i+1): state[i]})
                for action in final_policy[state]:
                    new_data.update({'ACTION': action})
                    agent_final_policy = agent_final_policy.append(new_data, ignore_index=True)
            agent_final_policy.to_csv(agent_final_policy_file, index=False)

            self.agents_data = self.agents_data.append(
                {'AGENT_ID': agent.agent_id, 'TOTAL_REWARD': agent.get_total_reward(),
                 'NUM_UPDATE_STEPS': agent.n_update_steps, 'FINAL_POLICY_FILE': file_name,
                 'ACTIONS_FILE': file_name}, ignore_index=True)

    def perform_experiment(self, experimental_parameters, specifics):
        # Gather specifics
        agents_info_list = specifics['agent_info_list']
        action_network_args = specifics['action_network_args']
        num_episodes = specifics['num_episodes']
        environment = specifics['environment']
        random_seed = specifics['seed'] if 'seed' in specifics else 0
        action_blocking_dl_args = specifics[
            'action_blocking_dl_args'] if 'action_blocking_dl_args' in specifics else None
        regression_dl_args = specifics['regression_dl_args'] if 'regression_dl_args' in specifics else None
        run_info = {'environment': environment,
                    'num_episodes': num_episodes,
                    'random_seed': random_seed}
        # Process Experimental Parameters
        chosen_learning_types = self.choose_from_enums(LearningType.all(), experimental_parameters, 'learning_types')
        chosen_algorithms = self.choose_from_enums(AlgorithmName.all(), experimental_parameters, 'algorithm_names')
        chosen_policies = self.choose_from_options(['epsilon_greedy', 'softmax', 'thompson_sampling', 'ucb'],
                                                   experimental_parameters, 'policies')
        chosen_action_blockers = self.create_boolean_list(experimental_parameters, 'enable_action_blocking')
        chosen_enable_regressors = self.create_boolean_list(experimental_parameters, 'enable_regressors')
        chosen_double_agent_flags = self.create_boolean_list(experimental_parameters, 'enable_double_learning')
        chosen_optimizers = self.choose_from_enums(NetworkOptimizer.all(), experimental_parameters, 'optimizers')
        policy_hyper_parameters = experimental_parameters['policy_hyper_parameters']
        algorithm_hyper_parameters = experimental_parameters['algorithm_hyper_parameters']
        replay_buffer_hyper_parameters = experimental_parameters[
            'replay_buffer_hyper_parameters'] if 'replay_buffer_hyper_parameters' in experimental_parameters else {}
        if 'alphas' not in algorithm_hyper_parameters or len(algorithm_hyper_parameters['alphas']) == 0:
            algorithm_hyper_parameters['alphas'] = [0.001]
        if 'gammas' not in algorithm_hyper_parameters or len(algorithm_hyper_parameters['gammas']) == 0:
            algorithm_hyper_parameters['gammas'] = [1.0]
        if 'beta_ms' not in algorithm_hyper_parameters or len(algorithm_hyper_parameters['beta_ms']) == 0:
            algorithm_hyper_parameters['beta_ms'] = [0.9]
        if 'beta_vs' not in algorithm_hyper_parameters or len(algorithm_hyper_parameters['beta_vs']) == 0:
            algorithm_hyper_parameters['beta_vs'] = [0.99]
        if 'epsilons' not in algorithm_hyper_parameters or len(algorithm_hyper_parameters['epsilons']) == 0:
            algorithm_hyper_parameters['epsilons'] = [1e-07]
        for optimizer_type in chosen_optimizers:
            action_network_args.update({'optimizer_type': optimizer_type})
            if action_blocking_dl_args is not None:
                action_blocking_dl_args['optimizer_type'] = optimizer_type
            if regression_dl_args is not None:
                regression_dl_args['optimizer_type'] = optimizer_type
            optimizer_args = {}
            for learning_rate in algorithm_hyper_parameters['alphas']:
                optimizer_args['learning_rate'] = learning_rate
                for beta_m in algorithm_hyper_parameters['beta_ms']:
                    optimizer_args['beta_m'] = beta_m
                    for beta_v in algorithm_hyper_parameters['beta_vs']:
                        optimizer_args['beta_v'] = beta_v
                        for epsilon in algorithm_hyper_parameters['epsilons']:
                            optimizer_args['epsilon'] = epsilon
                            action_network_args['optimizer_args'] = optimizer_args
                            if action_blocking_dl_args is not None:
                                action_blocking_dl_args['optimizer_args'] = optimizer_args
                            if regression_dl_args is not None:
                                regression_dl_args['optimizer_args'] = optimizer_args
                            run_info.update({'action_network_args': action_network_args,
                                             'action_blocking_dl_args': action_blocking_dl_args,
                                             'regression_dl_args': regression_dl_args})
                            for algorithm_name in chosen_algorithms:
                                run_info['algorithm_name'] = algorithm_name
                                algorithm_args = {'algorithm_name': algorithm_name}
                                for discount_factor in algorithm_hyper_parameters['gammas']:
                                    algorithm_args['discount_factor'] = discount_factor
                                    for enable_regressor in chosen_enable_regressors:
                                        algorithm_args['enable_iterator'] = enable_regressor
                                        run_info['algorithm_args'] = algorithm_args
                                        for policy_name in chosen_policies:
                                            run_info['policy_name'] = policy_name
                                            policy_args = {'random_seed': random_seed,
                                                           'min_penalty': environment.min_penalty}
                                            policy_hyper_parameter_list = [0]
                                            hyper_parameter_type = ''
                                            if policy_name == 'epsilon_greedy':
                                                policy_hyper_parameter_list = policy_hyper_parameters['epsilons']
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
                                                            agents = []
                                                            for i, agent_info in enumerate(agents_info_list):
                                                                agent_info.update({'agent_id': i + 1,
                                                                                   'is_double_agent': is_double_agent,
                                                                                   'learning_type': learning_type,
                                                                                   'state_dim': environment.required_state_dim,
                                                                                   'enable_action_blocking': enable_action_blocking})
                                                                policy_args.update(
                                                                    {'num_actions': len(agent_info['actions'])})
                                                                run_info['policy_args'] = policy_args
                                                                policy = choose_policy(policy_name, policy_args)
                                                                algorithm_args['policy'] = policy
                                                                algorithm = Algorithm(algorithm_args)
                                                                agent_info.update({'algorithm': algorithm})
                                                                agents.append(Agent(agent_info))

                                                            run_info['agents'] = agents
                                                            if learning_type == LearningType.REPLAY:
                                                                for num_replay in replay_buffer_hyper_parameters['num_replay']:
                                                                    run_info['num_replay'] = num_replay
                                                                    for buffer_size in replay_buffer_hyper_parameters['buffer_size']:
                                                                        run_info['buffer_size'] = buffer_size
                                                                        for mini_batch_size in replay_buffer_hyper_parameters['mini_batch_size']:
                                                                            run_info['mini_batch_size'] = mini_batch_size
                                                                            agents, run_times, time_steps = self.perform_run(
                                                                                run_info)
                                                                            self.process_run(run_info, agents,
                                                                                             run_times, time_steps)
                                                            else:
                                                                agents, run_times, time_steps = self.perform_run(
                                                                    run_info)
                                                                self.process_run(run_info, agents, run_times,
                                                                                 time_steps)

        self.hyper_parameters_data.to_csv(
            '{0}'.format(os.path.join(self.output_dir, 'run_summary_{0}.csv'.format(self.dt_str))), index=False)
        self.agents_data.to_csv(
            '{0}'.format(os.path.join(self.output_dir, 'agents_data_{0}.csv'.format(self.dt_str))), index=False)

    def perform_run(self, run_info={}):
        agents = run_info['agents']
        random_seed = run_info['random_seed'] if 'random_seed' in run_info else 0
        learning_type = run_info['learning_type'] if 'learning_type' in run_info else LearningType.ONLINE
        environment = run_info['environment']

        ml_data_dir = join(self.output_dir, 'ml_data')
        if not isdir(ml_data_dir):
            mkdir(ml_data_dir)

        enable_action_blocking = run_info['enable_action_blocking'] if 'enable_action_blocking' in run_info else False
        action_blocking_dl_args = run_info['action_blocking_dl_args'] if 'action_blocking_dl_args' in run_info else None

        algorithm_args = run_info['algorithm_args']
        enable_regression = algorithm_args['enable_iterator'] if 'enable_iterator' in algorithm_args else False
        regression_dl_args = algorithm_args['regression_dl_args'] if 'regression_dl_args' in algorithm_args else None
        action_network_args = run_info['action_network_args']

        for agent in agents:
            agent_dir = join(ml_data_dir, 'agent_{0}'.format(agent.agent_id))
            if not isdir(agent_dir):
                mkdir(agent_dir)
            if learning_type == LearningType.REPLAY:
                agent.buffer_init(run_info['num_replay'], run_info['buffer_size'], run_info['mini_batch_size'],
                                  random_seed)
            if enable_action_blocking:
                agent.blocker_init(csv_dir=agent_dir, dl_args=action_blocking_dl_args)
            if enable_regression:
                agent.algorithm.iterator_init(csv_dir=agent_dir, state_dim=agent.state_dim, dl_args=regression_dl_args)
            agent.network_init(action_network_args=action_network_args)
        environment.set_agents(agents)

        num_episodes = run_info['num_episodes']
        time_steps = np.zeros(num_episodes)
        run_times = np.zeros(num_episodes)

        for episode in range(num_episodes):
            environment.reset()
            done = False
            start = datetime.now()
            while not done:
                done = environment.step()

            time_steps[episode] = environment.current_time_step

            end = datetime.now()
            diff = end - start
            run_times[episode] = diff.seconds

            for agent in environment.agents:
                agent_dir = join(ml_data_dir, 'agent_{0}'.format(agent.agent_id))
                agent.historical_data.to_csv(
                    join(agent_dir,
                         'log{0}_episode{1}.csv'.format(datetime.now().strftime("%Y%m%d%H%M%S"), episode + 1)),
                    index=False)

        return environment.agents, run_times, time_steps
