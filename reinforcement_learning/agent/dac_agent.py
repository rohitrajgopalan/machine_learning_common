from .agent import Agent
from reinforcement_learning.network.dac_critic_network import DACCriticNetwork
from reinforcement_learning.network.dac_actor_network import DACActorNetwork
import numpy as np


class DACAgent(Agent):
    actor_network = None
    critic_network = None
    discount_factor = 0

    def __init__(self, args={}):
        super().__init__(args)

    def actor_network_init(self, actor_network_args):
        actor_network_args.update({'num_outputs': len(self.actions),
                                   'num_inputs' if type(self.state_dim) == int else 'input_shape': self.state_dim})
        self.actor_network = DACActorNetwork(actor_network_args)

    def critic_network_init(self, critic_network_args):
        critic_network_args.update({'num_outputs': 1,
                                    'num_inputs' if type(self.state_dim) == int else 'input_shape': self.state_dim})
        self.critic_network = DACCriticNetwork(critic_network_args)

    def optimize_network(self, experiences):
        if type(self.state_dim) == tuple:
            state_shape = [len(experiences)]
            for s in self.state_dim:
                state_shape.append(s)
            state_shape = tuple(state_shape)
        else:
            state_shape = (len(experiences), self.state_dim)
        states = np.zeros(state_shape)
        actions = np.zeros((len(experiences), len(self.actions)))
        rewards = np.zeros((len(experiences)))
        for batch_idx, experience in enumerate(experiences):
            s, a, s_, r, terminal = experience
            states[batch_idx] = s if self.state_type == np.ndarray else np.array([s])
            action_one_hot = np.zeros(len(self.actions))
            action_one_hot[a] = 1
            actions[batch_idx] = action_one_hot
            rewards[batch_idx] = r
        discounted_rewards = self.discount(rewards)
        state_values = self.critic_network.get_values(states)
        advantages = discounted_rewards - state_values
        self.actor_network.update(states, actions, advantages)
        self.critic_network.update(states, discounted_rewards)

    def assign_initial_action(self):
        self.initial_action = self.actor_network.action(self.current_state)

    def get_target_error(self, s, a, s_, r, active):
        return r - self.critic_network.get_value(s_)

    def discount(self, rewards):
        discounted_r = np.zeros_like(rewards)
        cumulative_r = 0
        for t in reversed(range(0, rewards.shape[0])):
            cumulative_r = rewards[t] + cumulative_r * self.discount_factor
            discounted_r[t] = cumulative_r
        return discounted_r

    def get_results(self):
        total_reward = 0
        final_policy = {}
        for state in self.experienced_states:
            best_actions = self.actor_network.best_actions(state)
            if self.flatten_state:
                state = state.flatten()
            if type(state) == np.ndarray:
                state = tuple(state)
            if len(best_actions) > 0:
                final_policy[state] = best_actions
            total_reward += self.critic_network.get_value(state)
        return total_reward, final_policy
