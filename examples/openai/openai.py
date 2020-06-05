from gym.spaces import Tuple, Box

from reinforcement_learning.environment.environment import Environment, RewardType, ActionType
import gym


class OpenAIGymEnvironment(Environment):
    openAI = None
    reset_required = False

    result_steps = {}

    def __init__(self, env_name, min_penalty):
        self.openAI = gym.make(env_name)
        self.reward_type = RewardType.Immediate
        if type(self.openAI.observation_space) == Tuple:
            self.required_state_dim = len(self.openAI.observation_space)
        elif type(self.openAI.observation_space) == Box:
            self.required_state_dim = self.openAI.observation_space.shape[0]
        else:
            self.required_state_dim = 1
        self.min_penalty = min_penalty

    def reset(self):
        super().reset()
        self.openAI.reset()
        for agent in self.agents:
            agent.current_state = self.openAI.reset()
        self.reset_required = False

    def is_complete(self):
        if super().is_complete():
            self.reset_required = True
        return self.reset_required

    def determine_next_state(self, agent, action_type):
        if action_type == ActionType.Actual and agent.actual_action is None:
            return None
        else:
            # if action_type == ActionType.Actual:
            #     self.openAI.render()
            action_ = agent.get_action(action_type)

            if agent.actions[action_] == 'SAMPLE':
                a = self.openAI.action_space.sample()
                action_ = agent.add_action(a)
                agent.set_action(action_type, action_)
            else:
                a = agent.actions[action_]

            observation, reward, terminal, _ = self.openAI.step(a)
            self.result_steps.update({agent.agent_id: {action_: (reward, terminal)}})
            self.openAI.close()
            return observation

    def calculate_reward(self, agent, state, action_type):
        if action_type == ActionType.Actual:
            if agent.actual_action is None:
                return 0, False
            else:
                return self.result_steps[agent.agent_id][agent.actual_action]
        else:
            return self.result_steps[agent.agent_id][agent.initial_action]
