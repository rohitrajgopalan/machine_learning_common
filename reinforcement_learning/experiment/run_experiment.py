from run import *
from choose_agent import *
from agent import *
from actionvaluenetwork import *
import numpy as np
from os import listdir,mkdir
from os.path import basename, isdir,isfile, join, dirname, realpath

def generate_agents(agent_name,agents_info_list=[]):
    agents = []
    for agent_info in agents_info_list:
        agents.append(choose_agent(agent_name,agent_info))
    return agents

def generate_policy_args(policy_name,policy_hyperparamters={}):

    return policy_args

def run_experiment(output_dir,environment,num_episodes,agents_info_list,chosen_types,policy_hyperparameters, algorithm_hyperparameters, replay_buffer_info={})
    output_dir = join(output_dir,'out')
    if not isdir(output_dir):
        mkdir(output_dir)

    agents = []


    run_info = {'environment':environment,
                'num_episodes':num_episodes,
                'learning_type':chosen_type['learning_type'],
                'output_dir':output_dir,
                'num_hidden_units': 2,
                'random_seed': 0,
                'beta_m': 0.9,
                'beta_v':0.99
                'epsilon':0.0001}

    all_policies = ['epsilon_greedy','softmax','thompson_sampling','ucb']
    chosen_policies = []

    all_agent_names = ['simple','dyna','dyna_plus','prioritized_sweeping']
    chosen_agent_names = []

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
        policy_args = {'random_seed':policy_hyperparameters['seed']}
        #either a list of taus, epsilons or confidence factors
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
                for algorithm_name in chosen_type['algorithm_names']:
                    run_info['algorithm_name'] = algorithm_name
                    algorithm_args = {}
                    for discount_factor in algorithm_hyperparameters['gammas']:
                        algorithm_args['discount_factor'] = discount_factor
                        for lambda_val in algorithm_hyperparameters['lambdas']:
                            algorithm_args['lambda_val'] = lambda_val
                            for enable_e_traces in [True,False]:
                                algorithm_args['enable_e_traces'] = enable_e_traces
                                run_info['algorithm_args'] = algorithm_args
                
                                for agent_name in chosen_agent_names:
                                    run_info['agent_name'] = agent_name
                                    run_info['agents'] = generate_agents(agent_name,agents_info_list)

                                    for network_type in [NetworkType.Single,NetworkType.Double]:
                                        run_info['network_type'] = network_type

                                        if chosen_type['learning_type'] == LearningType.Replay:
                                            for key in replay_buffer_info:
                                                run_info[key] = replay_buffer_info[key]

                                        print(run_info)
                                        run(run_info)


                                
                                        

                            
                
                    
                    
                
