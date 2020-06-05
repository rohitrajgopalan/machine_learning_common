from reinforcement_learning.environment.environment import Environment, RewardType, ActionType


class FixedSpaced(Environment):
    square_length = 0
    final_location = (0, 0)

    def __init__(self, square_length):
        super().__init__(RewardType.Delayed, 2, -10)
        self.square_length = square_length
        self.final_location = (square_length / 2, square_length / 2)

    def calculate_reward(self, agent, state, action_type):
        if agent.get_action(action_type) is None:
            return 0, False
        else:
            if state == self.final_location:
                reward = 100
                terminal = True
            else:
                terminal = False
                num_collisions = 0
                other_agents = [other for other in self.agents if not other.agent_id == agent.agent_id]
                for other in other_agents:
                    other_action = other.get_action(action_type)
                    other_next_state, _ = self.agents_step_info[other.agent_id][other_action]
                    if state == other_next_state:
                        num_collisions += 1

                if num_collisions == 0:
                    if state == agent.current_state:
                        reward = 0
                    else:
                        reward = 1
                else:
                    reward = -10 * num_collisions
            return reward, terminal

    def determine_next_state(self, agent, action_type):
        if agent.get_action(action_type) is None:
            return None
        else:
            chosen_action = agent.actions[agent.get_action(action_type)]
            x, y = agent.current_state
            if chosen_action == 'UP':
                x = max(x - 1, 1)
            elif chosen_action == 'DOWN':
                x = min(x + 1, self.square_length)
            elif chosen_action == 'LEFT':
                y = max(y - 1, 1)
            elif chosen_action == 'RIGHT':
                y = min(y + 1, self.square_length)
            return x, y
