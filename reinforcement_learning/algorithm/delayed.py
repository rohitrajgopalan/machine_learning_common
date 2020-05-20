from algorithm import Algorithm
from actionvaluenetwork import *
import numpy as np

class DelayedAlgorithm(Algorithm):
    #used for attempted updates
    U = []
    #counters
    l = None
    #time of last attempted update
    t = None
    LEARN = None
    timesteps = 0
    #time of most recent Q-value change
    t_ = 0
    #max counter value for any state-action pair
    m = 0
    #threshold delta between Q and update values
    epsilon = 0.0

    def __init__(self,**args):
        super().__init__(args)
        self.l = np.zeros((0,self.policy.num_actions))
        self.t = np.zeros((0,self.policy.num_actions))
        self.LEARN = np.ones((0,self.policy.num_actions))

    def add_state(self,s,network_type):
        if self.get_state_index(s) == -1 and not s is None:
            super().add_state(s,network_type)
            if len(self.U) == 0:
                for _ in network_type.value:
                    self.U.append(np.zeros((0,self.policy.num_actions)))
            for i in network_type.value:
                self.U[i] = np.append(self.U[i],np.zeros((1,self.policy.num_actions)),axis=0)
            self.l = np.append(self.l,np.zeros((1,self.policy.num_actions)),axis=0)
            self.t = np.append(self.t,np.zeros((1,self.policy.num_actions)),axis=0)
            self.LEARN = np.append(self.LEARN,np.ones((1,self.policy.num_actions)),axis=0)
    
    def get_target_error(self,s,a,s_,r,terminal,network,current_q):
        coin_side = network.determine_coin_side()
        target_error = 0
        self.timesteps += 1
        if self.LEARN[self.get_state_index(s),a] == 1:
            self.U[coin_side][self.get_state_index(s),a] += self.get_target_value(a,s_,r,terminal,current_q,coin_side)
            self.l[self.get_state_index(s),a] += 1
            if self.l[self.get_state_index(s),a] == self.m:
                q_mat = network.get_action_values(s,coin_side)
                if (q_mat[a] - self.U[coin_side][self.get_state_index(s),a])/self.m >= (2*self.epsilon):
                    target_error = self.U[coin_side][self.get_state_index(s),a]/self.m
                    if coin_side == 0:
                        target_error += self.epsilon
                    else:
                        target_error -= self.epsilon
                    self.t_ = self.timesteps
                elif self.t[self.get_state_index(s),a] >= self.t_:
                    self.LEARN[self.get_state_index(s),a] = 0
                self.t[self.get_state_index(s),a] = 0
                self.U[coin_side][self.get_state_index(s),a] = 0
                self.l[self.get_state_index(s),a] = 0
        elif self.t[self.get_state_index(s),a] < self.t_:
            self.LEARN[self.get_state_index(s),a] = 1
        return (target_error,coin_side)        

class DelayedSARSA(DelayedAlgorithm):
    def get_potential_action(self,current_q,coin_side,s_,a):
        return self.policy.choose_action(current_q.get_action_values(s_))
    
class DelayedQ(DelayedAlgorithm):
    def get_potential_action(self,current_q,coin_side,s_,a):
        return self.argmax(current_q.get_action_values(s_,coin_side))

class DelayedExpectedSARSA(DelayedAlgorithm):
    def get_scalar(self,s,a,coin_side,network):
        q_mat = network.get_action_values(s,coin_side)
        policy_mat = self.policy.derive(q_mat)
        return np.dot(policy_mat,q_mat.T)[0]
        
class DelayedMCQL(DelayedAlgorithm):
    def get_potential_next_action(self,current_q,coin_side,s_,a):
        return a
