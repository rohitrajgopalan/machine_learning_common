from .replay_buffer import ReplayBuffer
import numpy as np


class RandomizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, minibatch_size, **args):
        """
        Args:
            size (integer): The size of the replay buffer.
            minibatch_size (float/integer): The sample size.
            seed (integer): The seed for the random number generator.
        """
        super().__init__(size, minibatch_size)
        self.rand_generator = np.random.RandomState(args['seed'] if 'seed' in args else 0)

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
        """
        Returns:
            A list of transition tuples including state, action, reward, terinal, and next_state
        """
        ids = self.rand_generator.choice(np.arange(len(self.buffer)), size=self.minibatch_size)
        return [self.buffer[idx] for idx in ids]

