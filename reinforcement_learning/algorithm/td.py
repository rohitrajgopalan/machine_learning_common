from algorithm import Algorithm
import numpy as np

class TDAlgorithm(Algorithm):
    def get_target_error(self,s,a,s_,r,terminal,network,current_q):
        coin_side = network.determine_coin_side()

        if self.enable_e_traces:
            self.e_traces[self.get_state_index(s),a] += 1

        target_error = self.get_target_value(a,s_,r,terminal,current_q,coin_side) - self.get_scalar(s,a,coin_side,network)

        if self.enable_e_traces:
            target_error *= self.e_traces[self.get_state_index(s)]
            self.e_traces *= (self.lambda_val*self.discount_factor)
    
        return target_error,coin_side
    
class SARSA(TDAlgorithm):
    def get_potential_next_action(self,current_q,coin_side,s_,a):
        return self.policy.choose_action(current_q.get_action_values(s_))
    
class Q(TDAlgorithm):
    def get_potential_next_action(self,current_q,coin_side,s_,a):
        return self.argmax(current_q.get_action_values(s_,coin_side))

class ExpectedSARSA(TDAlgorithm):
    def get_scalar(self,s,a,coin_side,network):
        q_mat = network.get_action_values(s,coin_side)
        policy_mat = self.policy.derive(q_mat)
        return np.dot(policy_mat,q_mat.T)[0]
        
class MCQL(TDAlgorithm):
    def get_potential_action(self,current_q,coin_side,s,a):
        return a
