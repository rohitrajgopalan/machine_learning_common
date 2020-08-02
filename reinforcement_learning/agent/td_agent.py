import numpy as np

from reinforcement_learning.agent.agent import Agent
from reinforcement_learning.network.action_value_network import ActionValueNetwork


class TDAgent(Agent):
    is_double_agent = False
    policy_network = None
    target_network = None

    def __init__(self, args={}):
        super().__init__(args)
        self.historical_data_columns.append('ALGORITHM')
        self.historical_data_columns.append('POLICY')
        self.historical_data_columns.append('HYPERPARAMETER')
        self.reset()

    def network_init(self, action_network_args):
        action_network_args.update({'num_inputs': self.state_dim, 'num_outputs': len(self.actions)})
        self.policy_network = ActionValueNetwork(action_network_args)
        if self.is_double_agent:
            self.target_network = ActionValueNetwork(action_network_args)

    def step(self, r1, r2, should_action_be_blocked=False):
        if self.is_double_agent:
            self.target_network.set_weights(self.policy_network.get_weights())
        super().step(r1, r2, should_action_be_blocked)

    def add_to_supervised_learning(self, r, should_action_be_blocked):
        blocked_boolean = 1 if should_action_be_blocked else 0
        if self.enable_action_blocking:
            self.action_blocker.add(self.current_state, self.initial_action, blocked_boolean)
        new_data = {}
        if self.state_dim == 1:
            new_data.update({'STATE': self.current_state})
        else:
            for i in range(self.state_dim):
                state_val = self.current_state[i]
                if type(state_val) == bool:
                    state_val = int(state_val)
                new_data.update({'STATE_VAR{0}'.format(i + 1): state_val})
        if self.action_dim == 1:
            action = self.actions[self.initial_action]
            new_data.update({'INITIAL_ACTION': self.initial_action if type(action) == str else action})
        else:
            action = self.actions[self.initial_action]
            action = np.array([action])
            for i in range(self.action_dim):
                new_data.update({'INITIAL_ACTION_VAR{0}'.format(i + 1): action[0, i]})

        new_data.update({'REWARD': r,
                         'ALGORITHM': '{0}{1}'.format(self.algorithm.algorithm_name.name,
                                                      '_LAMBDA' if 'Lambda' in self.algorithm.__class__.__name__ else ''),
                         'GAMMA': self.algorithm.discount_factor,
                         'POLICY': self.algorithm.policy.__class__.__name__,
                         'TARGET_VALUE': self.algorithm.calculate_target_value(self.initial_action,
                                                                               self.next_state, r,
                                                                               int(self.active), self.policy_network,
                                                                               self.target_network),
                         'HYPERPARAMETER': self.algorithm.policy.get_hyper_parameter(),
                         'BLOCKED?': blocked_boolean})
        self.historical_data = self.historical_data.append(new_data, ignore_index=True)

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
            final_policy[state] = self.algorithm.policy.actions_with_max_value(action_values)

        return total_reward, final_policy
