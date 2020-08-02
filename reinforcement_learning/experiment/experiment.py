import enum
import os
from datetime import datetime
from os import mkdir
from os.path import join, isdir

import numpy as np
import pandas as pd

from neural_network.network_types import NetworkOptimizer
from reinforcement_learning.agent.ac_agent import ACAgent
from reinforcement_learning.agent.td_agent import TDAgent
from reinforcement_learning.agent.learning_type import LearningType
from reinforcement_learning.algorithm.algorithm import Algorithm, AlgorithmName
from reinforcement_learning.policy.choose_policy import choose_policy


class Experiment:
    hyper_parameters_data = None
    td_agent_cols = ['ALGORITHM', 'POLICY', 'HYPER_PARAMETER', 'GAMMA', 'ENABLE_DOUBLE_LEARNING',
                     'ENABLE_ACTION_BLOCKING', 'ACTION_BLOCKING_HELPER', 'OPTIMIZER',
                     'ALPHA', 'BETA_V', 'BETA_M', 'EPSILON', 'LEARNING_TYPE', 'NUM_REPLAY', 'BUFFER_SIZE',
                     'MINI_BATCH_SIZE',
                     'AVG_TIME_STEP', 'MAX_TIME_STEP', 'AVG_RUNTIME', 'MAX_RUNTIME']
    ac_agent_cols = ['GAMMA', 'ENABLE_NOISE', 'ENABLE_ACTION_BLOCKING', 'ACTION_BLOCKING_HELPER', 'OPTIMIZER',
                     'ALPHA', 'BETA_V', 'BETA_M', 'EPSILON', 'LEARNING_TYPE', 'NUM_REPLAY', 'BUFFER_SIZE',
                     'MINI_BATCH_SIZE',
                     'AVG_TIME_STEP', 'MAX_TIME_STEP', 'AVG_RUNTIME', 'MAX_RUNTIME']
    agent_cols = ['AGENT_ID', 'TOTAL_REWARD', 'NUM_UPDATE_STEPS', 'FINAL_POLICY_FILE', 'ACTIONS_FILE']
    agents_data = None
    action_cols = ['ID', 'ACTION', 'TYPE']
    output_dir = ''
    dt_str = ''

    def __init__(self, output_dir):
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
        is_td_agent = len([agent for agent in agents if type(agent) == TDAgent]) > 0
        is_ac_agent = len([agent for agent in agents if type(agent) == ACAgent]) > 0
        new_data = {'LEARNING_TYPE': run_info['learning_type'].name,
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
                    'AVG_TIME_STEP': np.mean(time_steps)}
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
        if is_ac_agent:
            new_data.update({'GAMMA': run_info['discount_factor'],
                             'ENABLE_NOISE': 'Yes' if run_info['enable_noise'] else 'No',
                             'OPTIMIZER': NetworkOptimizer.ADAM.name,
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
                {'AGENT_ID': agent.agent_id, 'TOTAL_REWARD': total_reward,
                 'NUM_UPDATE_STEPS': agent.n_update_steps, 'FINAL_POLICY_FILE': file_name,
                 'ACTIONS_FILE': file_name}, ignore_index=True)

    def perform_experiment_td(self, experimental_parameters, specifics):
        self.hyper_parameters_data = pd.DataFrame(columns=self.td_agent_cols)
        # Gather specifics
        agents_info_list = specifics['agent_info_list']
        action_network_args = specifics['action_network_args']
        num_episodes = specifics['num_episodes']
        environment = specifics['environment']
        random_seed = specifics['seed'] if 'seed' in specifics else 0
        action_blocking_dl_args = specifics[
            'action_blocking_dl_args'] if 'action_blocking_dl_args' in specifics else None
        run_info = {'environment': environment,
                    'num_episodes': num_episodes,
                    'random_seed': random_seed}
        # Process Experimental Parameters
        chosen_learning_types = self.choose_from_enums(LearningType.all(), experimental_parameters, 'learning_types')
        chosen_algorithms = self.choose_from_enums(AlgorithmName.all(), experimental_parameters, 'algorithm_names')
        chosen_policies = self.choose_from_options(['epsilon_greedy', 'softmax', 'thompson_sampling', 'ucb'],
                                                   experimental_parameters, 'policies')
        chosen_action_blockers = self.create_boolean_list(experimental_parameters, 'enable_action_blocking')
        chosen_double_agent_flags = self.create_boolean_list(experimental_parameters, 'enable_double_learning')
        chosen_optimizers = self.choose_from_enums(NetworkOptimizer.all(), experimental_parameters, 'optimizers')
        policy_hyper_parameters = experimental_parameters['policy_hyper_parameters']
        algorithm_hyper_parameters = experimental_parameters['algorithm_hyper_parameters']
        optimizer_hyper_parameters = experimental_parameters[
            'optimizer_hyper_parameters'] if 'optimizer_hyper_parameters' in experimental_parameters else {}
        replay_buffer_hyper_parameters = experimental_parameters[
            'replay_buffer_hyper_parameters'] if 'replay_buffer_hyper_parameters' in experimental_parameters else {}
        if 'gammas' not in algorithm_hyper_parameters or len(algorithm_hyper_parameters['gammas']) == 0:
            algorithm_hyper_parameters['gammas'] = [1.0]
        if 'alphas' not in optimizer_hyper_parameters or len(optimizer_hyper_parameters['alphas']) == 0:
            optimizer_hyper_parameters['alphas'] = [0.001]
        if 'beta_ms' not in optimizer_hyper_parameters or len(optimizer_hyper_parameters['beta_ms']) == 0:
            optimizer_hyper_parameters['beta_ms'] = [0.9]
        if 'beta_vs' not in optimizer_hyper_parameters or len(optimizer_hyper_parameters['beta_vs']) == 0:
            optimizer_hyper_parameters['beta_vs'] = [0.99]
        if 'epsilons' not in optimizer_hyper_parameters or len(optimizer_hyper_parameters['epsilons']) == 0:
            optimizer_hyper_parameters['epsilons'] = [1e-07]
        for optimizer_type in chosen_optimizers:
            action_network_args.update({'optimizer_type': optimizer_type})
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
                            run_info['optimizer_args'] = optimizer_args
                            action_network_args['optimizer_args'] = optimizer_args
                            if action_blocking_dl_args is not None:
                                action_blocking_dl_args['optimizer_args'] = optimizer_args
                            run_info.update({'action_network_args': action_network_args,
                                             'action_blocking_dl_args': action_blocking_dl_args})
                            for algorithm_name in chosen_algorithms:
                                run_info['algorithm_name'] = algorithm_name
                                algorithm_args = {'algorithm_name': algorithm_name}
                                for discount_factor in algorithm_hyper_parameters['gammas']:
                                    algorithm_args['discount_factor'] = discount_factor
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
                                                            algorithm = Algorithm(algorithm_args)
                                                            agent_info.update({'algorithm': algorithm})
                                                            agents.append(TDAgent(agent_info))

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
        self.agents_data.to_csv('{0}'.format(os.path.join(self.output_dir, 'agents_data_{0}.csv'.format(self.dt_str))),
                                index=False)

    def perform_experiement_ac(self, experimental_parameters, specifics):
        self.hyper_parameters_data = pd.DataFrame(columns=self.ac_agent_cols)
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
        if action_blocking_dl_args is not None:
            action_blocking_dl_args.update({'optimizer_type': NetworkOptimizer.ADAM})
        run_info = {'environment': environment,
                    'num_episodes': num_episodes,
                    'random_seed': random_seed}
        # Process Experimental Parameters
        chosen_learning_types = self.choose_from_enums(LearningType.all(), experimental_parameters, 'learning_types')
        chosen_action_blockers = self.create_boolean_list(experimental_parameters, 'enable_action_blocking')
        chosen_enable_noise_flags = self.create_boolean_list(experimental_parameters, 'enable_noise')
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
        for tau in experimental_parameters['taus']:
            actor_network_args['tau'] = tau
            critic_network_args['tau'] = tau
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
                                                agents.append(ACAgent(agent_info))
                                            run_info['agents'] = agents
                                            if learning_type == LearningType.REPLAY:
                                                for num_replay in replay_buffer_hyper_parameters['num_replay']:
                                                    run_info['num_replay'] = num_replay
                                                    for buffer_size in replay_buffer_hyper_parameters['buffer_size']:
                                                        run_info['buffer_size'] = buffer_size
                                                        for mini_batch_size in replay_buffer_hyper_parameters['mini_batch_size']:
                                                            run_info[
                                                                'mini_batch_size'] = mini_batch_size
                                                            agents, run_times, time_steps = self.perform_run(
                                                                run_info)
                                                            self.process_run(run_info, agents, run_times, time_steps)
                                            else:
                                                agents, run_times, time_steps = self.perform_run(
                                                    run_info)
                                                self.process_run(run_info, agents, run_times, time_steps)

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

        for agent in agents:
            agent_dir = join(ml_data_dir, 'agent_{0}'.format(agent.agent_id))
            if not isdir(agent_dir):
                mkdir(agent_dir)
            if learning_type == LearningType.REPLAY:
                agent.buffer_init(run_info['num_replay'], run_info['buffer_size'], run_info['mini_batch_size'],
                                  random_seed)
            else:
                agent.buffer_init(1, 1, 1, random_seed)
            if enable_action_blocking:
                agent.blocker_init(csv_dir=agent_dir, dl_args=action_blocking_dl_args)
            if type(agent) == TDAgent:
                action_network_args = run_info['action_network_args']
                agent.network_init(action_network_args=action_network_args)
            elif type(agent) == ACAgent:
                agent.actor_network_init(run_info['actor_network_args'])
                agent.critic_network_init(run_info['critic_network_args'])
                agent.exploration_noise_init(
                    run_info['exploration_noise_args'] if 'exploration_noise_args' in run_info else {})
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
