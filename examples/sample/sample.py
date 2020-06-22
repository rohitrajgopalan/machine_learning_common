from reinforcement_learning.environment.environment import Environment, RewardType


class SampleEnvironment(Environment):

    # TODO: Add variables

    def __init__(self,  # Enter constructor parameters
                 ):
        """
        Enter three variables as follows:
        1. RewardType.Immediate or RewardType.Delayed - This is to indicate whether we will reward an agent
        immediately after they carried out an action (RewardType.Immediate) or after all agents have carried out an action (RewardType.Delayed)
        2. What is the state dimensions for a state within this environment? e.g. if a state is to represent a 2-coordinate location then that value is 2
        This value gets carried across to the agents for processing especially for creating NNs for their action values
        3. What is the minimum penalty value that an agent can get within this environment. This value is the negative equivalent of a positive reward obtained.
        The library uses this value to determine whether in theory, an agent should be blocked from undertaking a particular action if the reward obtained is less
        than the minimum penalty
        """
        super().__init__(RewardType.Immediate, 0, 0)
        # TODO: Enter setters

    def determine_next_state(self, agent, action_type):
        if agent.get_action(action_type) is None:
            return None
        else:
            # TODO: Add logic to determine next state based for an agent
            # return next_state
            pass

    def calculate_reward(self, agent, state, action_type):
        if agent.get_action(action_type) is None:
            return 0, False
        else:
            # TODO: Add logic to determine the reward for an agent and whether it is terminal
            reward = 0
            terminal = False
            return reward, terminal
