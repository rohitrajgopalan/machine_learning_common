import os
import numpy as np
import datetime as dt
from os import listdir,mkdir
from os.path import basename, isdir,isfile, join, dirname, realpath
from choose_policy import *
from choose_algorithm import *
from environment import *
from agent import *

def run(run_info={}):
    lines_to_write = []
    agents = run_info['agents']

    environment = run_info['environment']

    output_dir = run_info['output_dir']
    dt_str = dt.date.today().strftime("%Y%m%d_%H%M%S")
    output_run_dir = join(output_dir,dt_str)

    mkdir(output_run_dir)
    
    policy_name = run_info['policy_name']
    policy_args = run_info['policy_args']
    

    algorithm_args = run_info['algorithm_args']
    algorithm_name = run_info['algorithm_name']

    network_type = run_info['network_type']
    num_hidden_units = run_info['num_hidden_units']
    random_seed = run_info['random_seed']

    learning_rate = run_info['learning_rate']
    beta_m = run_info['beta_m']
    beta_v = run_info['beta_v']
    epsilon = run_info['epsilon']

    lines_to_write.append('Number of Agents: {0}\n'.format(num_agents))
    lines_to_write.append('For Each Agent\nType: {0}\nLearning Type:{1}\n'.format(agent_name,learning_type.name))
    if learning_type == LearningType.Replay:
        buffer_size = run_info['buffer_size']
        minibatch_size = run_info['minibatch_size']
        num_replay = run_info['num_replay']
        lines.to_write.append('Buffer Size: {0}\nMini-Batch Size: {1}\nNum Replay Steps: {2}\n'.format(buffer_size,minibatch_size,num_replay))
    lines_to_write.append('\n')
    lines_to_write.append('Algorithm:\nName: {0}\n'.format(algorithm_name))
    lines_to_write.append('Policy: {0}'.format(policy_name))
    if policy_name == 'softmax':
        lines_to_write.append(' with tau: {0}\n'.format(policy.tau))
    elif policy_name == 'epsilon_greedy':
        lines_to_write.append(' with epsilon: {0}\n'.format(policy.epsilon))
    elif policy_name == 'ucb':
        lines_to_write.append(' with confidence factor: {0}\n'.format(policy.ucb_c))
    else:
        lines_to_write.append('\n')
    lines_to_write.append('Discount Factor: {0}\n'.format(algorithm.discount_factor))
    if algorithm.enable_e_traces:
        lines_to_write.append('Enable Eligibility Traces: Yes\nLambda:{0}\n'.format(algorithm.lambda_val))
    else:
        lines_to_write.append('Enable Eligibility Traces: No\n')
    lines_to_write.append('Network: {0}\n'.format(network_type.name))
    lines_to_write.append('Adam Optimizer:\nLearning Rate: {1}\nBeta M: {2}\nBeta W:{3}\nEpsilon:{4}\n\n'.format(learning_rate,beta_m,beta_w,epsilon))
    for i, agent in enumerate(agents):
        mkdir(join(output_run_dir,'agent_{0}'.format(i+1)))
        agent.agent_id = i+1
        
        policy_args.update({'num_actions':len(agent.actions)})
        policy = choose_policy(policy_name,policy_args)

        algorithm_args.update({'policy':policy})
        agent.algorithm = choose_algorithm(algorithm_name,algorithm_args)
        agent.learning_type = learning_type
        agent.network_init(network_type,num_hidden_units,random_seed)
        agent.optimizer_init(learning_rate,beta_m,beta_v,epsilon)
        if learning_type == LearningType.Replay:
            agent.buffer_init(num_replay,buffer_size,minibatch_size,random_seed)
        agents.append(agent)

    environment.agents = agents

    num_epsiodes = run_info['num_episodes']
    timesteps = np.zeros(num_episodes)
    runtimes = np.zeros(num_episodes)

    for episode in range(num_episodes):
        env.reset()
        done = False
        start = dt.datetime.now()
        t = 0
        while not done:
            done = env.step()
            t += 1

        timesteps[episode] = t
        
        end = dt.datetime.now()
        diff = end - start
        runtimes[episode] = diff.seconds

        lines_to_write.append('Completed episode {0}. Time taken: {1} seconds, Number of steps: {2}'.format(episode,runtimes[episode],timesteps[episode]))

        for agent in environment.agents:
            agent_dir = join(output_run_dir,'agent_{0}'.format(agent.agent_id))
            agent.historical_data.to_csv(join(agent_dir,'epsiode_{0}.csv'.format(episode+1)),header=True)

    agents = environment.agents

    lines_to_write.append('Results after {0} episodes\n'.format(num_episodes))
    lines_to_write.append('Maximum number of time steps: {0}\nAverage number of steps: {1}\n'.format(np.max(timesteps),np.mean(timesteps)))
    lines_to_write.append('Maximum run-time in seconds: {0}\nAverage run-time in seconds: {1}\n'.format(np.max(runtimes),np.mean(runtimes)))
    lines_to_write.append('Results for each agent\n')
    for agent in agents:
        lines_to_write.append('Agent {0}\n'.format(agent.agent_id))
        lines_to_write.append('Number of Update Steps: {0}\n'.format(agent.n_update_steps))
        lines_to_write.append('Total Reward: {0}\n'.format(agent.get_total_reward()))
        positive_actions = agent.get_all_positive_actions()
        for state in agent.algorithm.state_space:
            if state in positive_actions:
                lines_to_write.append('State {0}: {1}\n'.format(state,positive_actions[state]))
            else:
                lines_to_write.append('State {0}\n'.format(state))
        lines_to_write.append('\n')

    run_output_file = open(join(output_run_dir,'out.txt','w'))
    run_output_file.writelines(lines_to_write)
    run_output_file.close()
