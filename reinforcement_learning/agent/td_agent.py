import numpy as np

from reinforcement_learning.agent.agent import Agent
from reinforcement_learning.network.action_value_network import ActionValueNetwork


class TDAgent(Agent):
    is_double_agent = False
    action_value_network = None

    def __init__(self, args={}):
        super().__init__(args)

    def network_init(self, action_network_args):
        action_network_args.update({'num_outputs': len(self.actions),
                                    'num_inputs' if type(self.state_dim) == int else 'input_shape': self.state_dim})
        self.action_value_network = ActionValueNetwork(action_network_args, self.is_double_agent)

    def step(self, r1, r2, should_action_be_blocked=False):
        self.action_value_network.update_target_weights()
        super().step(r1, r2, should_action_be_blocked)

    def assign_initial_action(self):
        self.initial_action = self.algorithm.policy.choose_action(self.current_state, self.action_value_network)

    def optimize_network(self, experiences):
        q_target = np.zeros((len(experiences), len(self.actions)))
        if type(self.state_dim) == tuple:
            state_shape = [len(experiences)]
            for s in self.state_dim:
                state_shape.append(s)
            state_shape = tuple(state_shape)
        else:
            state_shape = (len(experiences), self.state_dim)
        states = np.zeros(state_shape)
        for batch_idx, experience in enumerate(experiences):
            s, a, s_, r, terminal = experience
            target_value = self.algorithm.calculate_target_value(a, s_, r, 1 - terminal, self.action_value_network)
            try:
                q_target[batch_idx, a] = target_value
                states[batch_idx] = s
            except ValueError:
                print('Unable to set Target Value {0} for Batch Index {1}, State {2} Action Index {3}'.format(
                    target_value, batch_idx, s, a))
        self.action_value_network.update_network(states, q_target)

    def get_results(self):
        total_reward = 0
        final_policy = {}

        for state in self.experienced_states:
            action_values = self.action_value_network.get_action_values(state)
            total_reward += np.sum(action_values)
            if self.flatten_state:
                state = state.flatten()
            if type(state) == np.ndarray:
                state = tuple(state)
            final_policy[state] = self.algorithm.policy.actions_with_max_value(action_values)

        return total_reward, final_policy

    def get_target_error(self, s, a, s_, r, active):
        return self.algorithm.get_target_error(s, a, s_, r, active, self.action_value_network)
