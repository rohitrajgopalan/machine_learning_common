import enum


class RewardType(enum.Enum):
    Immediate = 1
    Delayed = 2


class ActionType(enum.Enum):
    Initial = 1
    Actual = 2


class Environment:
    agents = []
    reward_type = None
    required_state_dim = 0

    min_penalty = -1

    # agent_id: (next_state_1, rewards_1, next_state_2, rewards_2) or
    agents_step_info = {}

    def __init__(self, reward_type, required_state_dim, min_penalty):
        self.reward_type = reward_type
        self.required_state_dim = required_state_dim
        self.min_penalty = min_penalty

    def set_agents(self, agents):
        self.agents = agents

    def perform_agent_step(self, agent):
        agent_info = {}
        if not agent.did_block_action:
            next_state = self.determine_next_state(agent, ActionType.Actual)
            agent_info.update({agent.initial_action: (next_state, 0), agent.actual_action: (next_state, 0)})
        else:
            ns1 = self.determine_next_state(agent, ActionType.Initial)
            ns2 = self.determine_next_state(agent, ActionType.Actual)
            agent_info.update({agent.initial_action: (ns1, 0), agent.actual_action: (ns2, 0)})

        self.agents_step_info.update({agent.agent_id: agent_info})

    def determine_next_state(self, agent, action_type):
        return None

    def reset(self):
        for agent in self.agents:
            agent.reset()
        self.agents_step_info = {}

    def is_complete(self):
        active_agents = [agent for agent in self.agents if agent.active]
        return len(active_agents) == 0

    def generate_rewards(self):
        for agent in self.agents:
            if not agent.active:
                continue
            if not agent.did_block_action:
                next_state, _ = self.agents_step_info[agent.agent_id][agent.actual_action]
                r, terminal = self.calculate_reward(agent, next_state, ActionType.Actual)
                self.agents_step_info[agent.agent_id].update(
                    {agent.initial_action: (next_state, r), agent.actual_action: (next_state, r)})
                agent.next_state = next_state
                agent.active = not terminal
            else:
                ns1, _ = self.agents_step_info[agent.agent_id][agent.initial_action]
                r1, _ = self.calculate_reward(agent, ns1, ActionType.Initial)
                ns2, _ = self.agents_step_info[agent.agent_id][agent.actual_action]
                r2, terminal = self.calculate_reward(agent, ns2, ActionType.Actual)
                self.agents_step_info[agent.agent_id].update(
                    {agent.initial_action: (ns1, r1), agent.actual_action: (ns2, r2)})
                if agent.actual_action is None:
                    agent.next_state = agent.current_state
                else:
                    agent.next_state = ns2
                    agent.active = not terminal

    def calculate_reward(self, agent, state, action_type):
        return 0, False

    def step(self):
        if self.is_complete():
            return True
        active_agents = [agent for agent in self.agents if agent.active]
        for agent in active_agents:
            agent.choose_next_action()
            self.perform_agent_step(agent)
            if self.reward_type == RewardType.Immediate:
                self.generate_rewards()
                _, r1 = self.agents_step_info[agent.agent_id][agent.initial_action]
                _, r2 = self.agents_step_info[agent.agent_id][agent.actual_action]
                agent.step(r1, r2)
                if self.is_complete():
                    return True

        if self.reward_type == RewardType.Delayed:
            self.generate_rewards()
            for agent in active_agents:
                _, r1 = self.agents_step_info[agent.agent_id][agent.initial_action]
                _, r2 = self.agents_step_info[agent.agent_id][agent.actual_action]
                agent.step(r1, r2)
            return self.is_complete()
        else:
            return False
