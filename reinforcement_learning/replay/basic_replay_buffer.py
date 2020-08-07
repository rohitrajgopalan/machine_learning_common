from .replay_buffer import ReplayBuffer


class BasicReplayBuffer(ReplayBuffer):
    def append(self, state, action, next_state, reward, terminal, **args):
        """
        Args:
            state (Numpy array): The state.
            action (integer): The action.
            reward (float): The reward.
            terminal (integer): 1 if the next state is a terminal state and 0 otherwise.
            next_state (Numpy array): The next state.
        """
        if len(self.buffer) == self.max_size:
            del self.buffer[0]
        self.buffer.append([state, action, next_state, reward, terminal])

    def sample(self):
        return [self.buffer[idx] for idx in range(self.minibatch_size)]
