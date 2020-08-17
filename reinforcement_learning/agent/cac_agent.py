from reinforcement_learning.agent.agent import Agent
from reinforcement_learning.network.cac_actor_network import CACActorNetwork
from reinforcement_learning.network.cac_critic_network import CACCriticNetwork
import numpy as np


class CACAgent(Agent):
    actor_network = None
    critic_network = None
    discount_factor = 0
    action_min = None
    action_max = None

    def __init__(self, args={}):
        super().__init__(args)

    def actor_network_init(self, actor_network_args, normal_dist_std_dev):
        actor_network_args.update({'num_outputs': self.action_dim,
                                   'num_inputs' if type(self.state_dim) == int else 'input_shape': self.state_dim})
        self.actor_network = CACActorNetwork(actor_network_args, normal_dist_std_dev)

    def critic_network_init(self, critic_network_args):
        critic_network_args.update({'num_outputs': 1,
                                    'num_inputs' if type(self.state_dim) == int else 'input_shape': self.state_dim})
        self.critic_network = CACCriticNetwork(critic_network_args)

    def assign_initial_action(self):
        action = self.actor_network.action(self.current_state)
        action = np.clip(action, self.action_min, self.action_max)
        self.initial_action = self.add_action(action)

    def optimize_network(self, experiences):
        if type(self.state_dim) == tuple:
            state_shape = [len(experiences)]
            for s in self.state_dim:
                state_shape.append(s)
            state_shape = tuple(state_shape)
        else:
            state_shape = (len(experiences), self.state_dim)
        states = np.zeros(state_shape)
        actions = np.zeros((len(experiences), self.action_dim))
        rewards = np.zeros((len(experiences)))
        target_values = np.zeros((len(experiences)))
        target_errors = np.zeros((len(experiences)))
        for batch_idx, experience in enumerate(experiences):
            s, a, s_, r, terminal = experience
            states[batch_idx] = s if self.state_type == np.ndarray else np.array([s])
            action = self.actions[a]
            actions[batch_idx] = action if self.action_type == np.ndarray else np.array([action])
            s_ = s_ if self.state_dim > 1 else np.array([s_])
            target_values[batch_idx] = self.calculate_target_value(s_, r, 1 - terminal)
            target_errors[batch_idx] = self.get_target_error(s, a, s_, r, 1 - terminal)
            rewards[batch_idx] = r

        self.critic_network.update(states, target_values)
        self.actor_network.update(states, actions, target_errors)

    def get_target_error(self, s, a, s_, r, active):
        return self.calculate_target_value(s_, r, active) - self.critic_network.get_value(s)

    def calculate_target_value(self, s_, r, active):
        return r + (self.discount_factor * self.critic_network.get_value(s_) * active)

    def get_results(self):
        self.actions = []
        action_counter = 0
        total_reward = 0
        final_policy = {}
        for state in self.experienced_states:
            action = self.actor_network.action(state)
            action_counter += 1
            total_reward += self.critic_network.get_value(state)
            self.actions.append(action)
            if self.flatten_state:
                state = state.flatten()
            if type(state) == np.ndarray:
                state = tuple(state)
            final_policy[state] = [action_counter - 1]
        return total_reward, final_policy
