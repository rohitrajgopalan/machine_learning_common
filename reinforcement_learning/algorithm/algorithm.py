import numpy as np
from actionvaluenetwork import *

class Algorithm:
    discount_factor = 0.0
    policy = None
    lambda_val = 0.0
    e_traces = None
    state_space = []
    enable_e_traces = False
    
    def __init__(self,**args):
        for key in args:
            setattr(self,key,args[key])
        if self.enable_e_traces:
            self.e_traces = np.zeros((0,self.policy.num_actions))

    def get_state_index(self,s):
        if not s is None and tuple(s) in self.state_space:
            return self.state_space.index(tuple(s))
        else:
            return -1
        
    def add_state(self,s,network_type):
        if self.get_state_index(s) == -1 and not s is None:
            state_space.append(tuple(s))
            if self.enable_e_traces:
                self.e_traces = np.append(self.e_traces,np.zeros((1,self.policy.num_actions)),axis=0)
    
    def get_target_error(self,s,a,s_,r,terminal,network,current_q):
        return (0,0)

    def get_target_value(self,a,s_,r,terminal,current_q,coin_side):
        a_ = self.get_potential_action(current_q,coin_side,s_,a)
        return r + (self.discount_factor * self.get_scalar(s_,a_,current_q.network_type.value-1-coin_side,current_q) * (1-terminal))

    def get_potential_action(self,current_q,coin_side,s,a):
        return -1

    def get_scalar(self,s,a,coin_side,network):
        q_mat = network.get_action_values(s,coin_side)
        return q_mat[a]
