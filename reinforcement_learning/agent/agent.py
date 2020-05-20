from policy as policy
from algorithm import Algorithm
from actionvaluenetwork import *
from replay_buffer import ReplayBuffer
import numpy as np
import enum

class LearningType(enum.Enum):
    Online = 1
    Replay = 2

class Agent:
    agent_id = 0
    n_update_steps = 0
    learning_type = None
    replay_buffer = None
    num_replay = 1
    
    algorithm = None
    lambda_val = 0.0
    
    active = True

    actions = []

    initial_state = None
    current_state = None
    next_state = None

    initial_action = -1
    actual_action = -1

    optimizer = None
    network = None

    def __init__(self,**args):
        for key in args:
            if 'random_seed' in key:
                setattr(self,key[:key.index('random_seed')]+'rand_generator',np.random.RandomState(args[key]))
            elif key == 'state_dim':
                state_dim = args['state_dim']
            elif key == 'start_loc':
                start_loc = args['state_loc'] 
            else:
                setattr(self,key,args[key])
        self.initialise_state(state_dim,start_loc)
        self.e_traces = np.empty((0,len(self.actions)))
        self.reset()        

    def network_init(self,network_type,num_hidden_units,random_seed):
        self.network = ActionValueNetwork(network_type,len(list(self.current_state)),num_hidden_units,self.n_actions,random_seed)

    def buffer_init(self,replay_buffer_size,minibatch_size,random_seed):
        if not self.learning_type == LearningType.Replay:
            pass
        self.replay_buffer = ReplayBuffer(replay_buffer_size,minibatch_size,random_seed)

    def initialize_state(self,state_dim, start_loc):
        l = list(np.array([start_loc]))

        if state_dim > len(l):
            for _ in range(len(l),state_dim):
                l.append(0)
            self.initial_state = tuple(l)
        else:
            self.initial_state = tuple(start_loc)

    def reset(self):
        self.current_state = initial_state
        self.active = True
        self.algorithm.add_state(self.current_state,self.network.network_type)
        self.n_update_steps = 0

    def optimizer_init(self,learning_rate,beta_m,beta_v,epsilon):
        self.optimizer = Adam(self.network.layer_sizes,learning_rate,beta_m,beta_v,epsilon)

    def step(self,r,terminal=False):
        self.n_update_steps += 1
        if not self.initial_action == self.actual_action:
            r *= -1
        self.algorithm.policy.update(r,self.initial_action)
        self.active = not terminal

        self.algorithm.add_state(self.next_state,self.network.network_type)

        current_q = deepcopy(self.network)
        if self.learning_type == LearningType.Replay:
            self.replay_buffer.append(self.current_state,self.initial_action,r,int(terminal),self.next_state)
            if self.replay_buffer.size() > self.replay_buffer.minibatch_size:
                for _ in range(self.num_replay):
                    experiences = self.replay_buffer.sample()
                    self.optimize_network(experiences,current_q)
        else:
            self.optimize_network(self.current_state,self.initial_action,self.next_state,r,int(terminal),current_q)

        #TODO: Add data to supervised learning

        if self.active:
            self.current_state = self.next_state

    def choose_next_action(self):
        action_values = self.network.get_action_values(self.current_state)
        self.initial_action = self.algorithm.policy.choose_action(action_values)
        #TODO: add supervised learning to block potential negative actions
        self.actual_action = self.initial_action

    def optimize_network(self,experiences,current_q):
        coin_side = self.network.determine_coin_side()
        unique_idx_states_experienced = []
        delta_mat = np.zeros((0,network.num_actions))
        for experience in experiences:
            s,a,r,terminal,s_ = experience
            delta_vec = self.get_delta_vec(s,a,s_,r,terminal,coin_side,current_q)
            state_index = self.get_state_index(s)
            if state_index in unique_idx_states_experienced:
                delta_mat[unique_idx_states_experienced.index(state_index),:] += delta_vec
            else:
                unique_idx_states_experienced.append(state_index)
                delta_mat = np.append(delta_mat,delta_vec)
        states = [self.state_space[idx] for idx in unique_idx_states_experienced]
        self.update_network_with_delta_mat(states,delta_mat,coin_side)
                
    def optimize_network(self,s,a,s_,r,terminal,current_q):
        delta_vec,coin_side = self.get_delta_vec(s,a,s_,r,terminal,current_q)
        delta_mat = np.zeros((1,network.num_actions))
        delta_mat[0,:] = delta_vec

        states = np.zeros((0,len(list(s))))
        states = np.append(states,np.array([s]))

        self.update_network_with_delta_mat(states,delta_mat,coin_side) 

    def update_network_with_delta_mat(self,states,delta_mat,coin_side):
        target_update = self.network.get_target_update(states,delta_mat,coin_side)
        weights = self.network.get_weights()

        for i in range(len(weights)):
            weights[i] = self.optimizer.update_weights(weights[i],target_update)
    
        self.network.set_weights(weights)        

    def get_delta_vec(self,s,a,s_,r,terminal,current_q):
        target_error,coin_side = self.algorithm.get_target_error(s,a,s_,r,terminal,self.network,current_q)
        delta_vec = np.zeros(len(self.actions))
        delta_vec[a] = target_error
        return delta_vec,coin_side
    
    def get_total_reward(self):
        total_reward = 0

        for state in self.algorithm.state_space:
            action_values = self.network.get_action_values(state)
            total_reward += np.sum(action_values)

        return total_reward

    def get_all_positive_actions(self):
        positive_actions = {}

        for state in self.algorithm.state_space:
            action_values = self.network.get_action_values(state)
            pa_list = list((np.where(action_values > 0))[0])
            positive_actions[state] = [action for action in self.actions if self.actions.index(action) in pa_list]  
        return positive_actions

    def print_agent_info(self):
        print('Information on Agent No. {0}'.format(self.agent_id))
        print('Total reward: {0}'.format(self.get_total_reward()))
        positive_actions = self.get_all_positive_actions()
        print('States visited with any potential actions:')
        for state in self.algorithm.state_space:
            if state in positive_actions:
                print('State {0}: {1}'.format(state,positive_actions[state]))
            else:
                print('State {0}'.format(state))
            
        
