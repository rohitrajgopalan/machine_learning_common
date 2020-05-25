from reinforcement_learning.environment.environment import Environment, RewardType
import gym


class OpenAIGymEnvironment(Environment):
    openAI = None
    reset_required = False

    result_steps = {}

    def __init__(self, env_name):
        self.openAI = gym.make(env_name)
        super().__init__(RewardType.Immediate, self.openAI.observation_space.shape[0])

    def reset(self):
        self.openAI.reset()
        super().reset()

    def is_complete(self):
        return self.reset_required and super().is_complete()

    def determine_next_state(self, agent):
        observation, reward, terminal, _ = self.openAI.step(agent.actual_action)
        self.result_steps.update({agent.agent_id: (observation, reward, terminal)})

    def calculate_reward(self, agent):
        observation, reward, terminal = self.result_steps[agent.agent_id]
        agent.next_state = tuple(observation)
        if terminal:
            agent.active = False
            self.openAI.env.close()
            self.reset_required = True
        return reward
