import numpy as np

from reinforcement_learning.agent.agent import Agent
from reinforcement_learning.network.action_value_network import ActionValueNetwork


class TDAgent(Agent):
    is_double_agent = False
    policy_network = None
    target_network = None

    def __init__(self, args={}):
        super().__init__(args)

    def network_init(self, action_network_args):
        action_network_args.update({'num_inputs': self.state_dim, 'num_outputs': len(self.actions)})
        self.policy_network = ActionValueNetwork(action_network_args)
        if self.is_double_agent:
            self.target_network = ActionValueNetwork(action_network_args)

    def step(self, r1, r2, should_action_be_blocked=False):
        if self.is_double_agent:
            self.target_network.set_weights(self.policy_network.get_weights())
        super().step(r1, r2, should_action_be_blocked)

    def assign_initial_action(self):
        self.initial_action = self.algorithm.policy.choose_action(self.current_state, self.policy_network)

    def optimize_network(self, experiences):
        q_target = np.zeros((len(experiences), len(self.actions)))
        states = np.zeros((len(experiences), self.state_dim))
        for batch_idx, experience in enumerate(experiences):
            s, a, s_, r, terminal = experience
            target_value = self.algorithm.calculate_target_value(a, s_, r, 1 - terminal, self.policy_network,
                                                                 self.target_network)
            try:
                q_target[batch_idx, a] = target_value
                states[batch_idx] = s
            except ValueError:
                print('Unable to set Target Value {0} for Batch Index {1}, State {2} Action Index {3}'.format(
                    target_value, batch_idx, s, a))
        self.policy_network.update_network(states, q_target)

    def get_results(self):
        total_reward = 0
        final_policy = {}

        for state in self.experienced_states:
            action_values = self.policy_network.get_action_values(state)
            total_reward += np.sum(action_values)
            final_policy[tuple(state) if type(state) == np.ndarray else state] = self.algorithm.policy.actions_with_max_value(action_values)

        return total_reward, final_policy

    def get_target_error(self, reward):
        return self.algorithm.get_target_error(self.current_state, self.initial_action, self.next_state, reward, self.active, self.policy_network, self.target_network)