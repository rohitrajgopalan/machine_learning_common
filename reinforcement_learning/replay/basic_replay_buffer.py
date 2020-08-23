from .replay_buffer import ReplayBuffer


class BasicReplayBuffer(ReplayBuffer):
    def append(self, state, action, next_state, reward, terminal, picked_action_prob=0, **args):
        """
        Args:
            state (Numpy array): The state.
            action (integer): The action.
            reward (float): The reward.
            terminal (integer): 1 if the next state is a terminal state and 0 otherwise.
            next_state (Numpy array): The next state.
            picked_action_prob (integer): Probability of action that was chosen. Used for Policy networks
        """
        if len(self.buffer) == self.max_size:
            del self.buffer[0]
        self.buffer.append([state, action, next_state, reward, terminal, picked_action_prob])

    def sample(self):
        return [self.buffer[idx] for idx in range(self.minibatch_size)]
