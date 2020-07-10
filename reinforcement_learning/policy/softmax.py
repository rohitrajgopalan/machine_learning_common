import numpy as np

from .policy import Policy


class Softmax(Policy):
    # tau (float): The temperature parameter scalar.
    tau = 0.0

    def derive_policy_based_from_values(self, action_values):
        """
        Args:
            action_values (Numpy array): A 2D array of shape (batch_size, num_actions). 
                           The action-values computed by an action-value network.
        Returns:
            A 2D array of shape (batch_size, num_actions). Where each column is a probability distribution over
            the actions representing the policy.
        """
        if action_values.shape() == (0, 1):
            preferences = np.zeros((1, self.num_actions)) / self.tau
            exp_preferences = np.exp(preferences)
            sum_of_exp_preferences = np.sum(exp_preferences, axis=1, keepdims=True)
            return exp_preferences / sum_of_exp_preferences
        else:
            # Compute the preferences by dividing the action-values by the temperature parameter tau
            preferences = action_values / self.tau
            # Compute the maximum preference across the actions
            max_preference = np.max(preferences, axis=1, keepdims=True)

            # Reshape max_preference array which has shape [Batch,] to [Batch, 1]. This allows NumPy broadcasting
            # when subtracting the maximum preference from the preference of each action.
            reshaped_max_preference = max_preference.reshape((-1, 1))

            # Compute the numerator, i.e., the exponential of the preference - the max preference.
            exp_preferences = np.exp(preferences - reshaped_max_preference)
            # Compute the denominator, i.e., the sum over the numerator along the actions axis.
            sum_of_exp_preferences = np.sum(exp_preferences, axis=1, keepdims=True)

            # Reshape sum_of_exp_preferences array which has shape [Batch,] to [Batch, 1] to  allow for NumPy broadcasting
            # when dividing the numerator by the denominator.
            reshaped_sum_of_exp_preferences = sum_of_exp_preferences.reshape((-1, 1))

            # Compute the action probabilities according to the equation in the previous cell.
            action_probs = exp_preferences / reshaped_sum_of_exp_preferences

            # squeeze() removes any singleton dimensions. It is used here because this function is used in the
            # agent policy when selecting an action (for which the batch dimension is 1.) As np.random.choice is used in
            # the agent policy and it expects 1D arrays, we need to remove this singleton batch dimension.
            action_probs = action_probs.squeeze()
            return action_probs

    def choose_action_based_from_values(self, action_values):
        probs_batch = self.derive_policy_based_from_values(action_values)
        probs_batch = np.array([probs_batch])
        try:
            return self.rand_generator.choice(self.num_actions, p=probs_batch.squeeze())
        except TypeError:
            return self.rand_generator.choice(self.num_actions, p=probs_batch)
