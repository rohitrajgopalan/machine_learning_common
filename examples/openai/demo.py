from os import mkdir
from os.path import join, dirname, realpath, isdir

# from examples.common_online_demo import run_demo
from gym.spaces import Box, Discrete, Tuple

from examples.openai.openai import OpenAIGymEnvironment
from examples.simple_online_demo import run_demo

open_ai_problems = {
    'algorithms': ['Copy-v0', 'DuplicatedInput-v0', 'RepeatCopy-v0', 'Reverse-v0', 'ReversedAddition-v0',
                   'ReversedAddition3-v0']
    # 'control': ['Acrobot-v1', 'CartPole-v1', 'MountainCar-v0', 'MountainCarContinuous-v0'],
    # 'box2d': ['BipedalWalker-v3', 'BipedalWalkerHardcore-v3', 'CarRacing-v0', 'LunarLander-v2', 'LunarLanderContinuous-v2']
    # 'toy_text': ['CliffWalking-v0', 'FrozenLake-v0', 'FrozenLake8x8-v0', 'GuessingGame-v0', 'HotterColder-v0','NChain-v0', 'Roulette-v0', 'Taxi-v3']
}

enable_negative_blocking = {'algorithms': False, 'control': False, 'box2d': True, 'toy_text': True}

for category in open_ai_problems:
    min_penalty = -2
    if category == 'algorithms':
        min_penalty = -0.5
    elif category == 'box2d':
        min_penalty = -100
    elif category == 'toy_text':
        min_penalty = -1
    if not isdir(join(dirname(realpath('__file__')), category)):
        mkdir(join(dirname(realpath('__file__')), category))
    for open_ai_gym_env in open_ai_problems[category]:
        if not isdir(join(dirname(realpath('__file__')), category, open_ai_gym_env.split('-')[0].lower())):
            mkdir(join(dirname(realpath('__file__')), category, open_ai_gym_env.split('-')[0].lower()))
        if open_ai_gym_env in ['Taxi-v3']:
            min_penalty = -10
        environment = OpenAIGymEnvironment(open_ai_gym_env, min_penalty)

        actions = []
        if type(environment.openAI.action_space) == Tuple:
            actions = []
            for a1 in range(environment.openAI.action_space[0].n):
                for a2 in range(environment.openAI.action_space[1].n):
                    for a3 in range(environment.openAI.action_space[2].n):
                        actions.append((a1, a2, a3))
        elif type(environment.openAI.action_space) == Box:
            actions = ['SAMPLE']
        elif type(environment.openAI.action_space) == Discrete:
            actions = list(range(environment.openAI.action_space.n))

        agent_info_list = [
            {'actions': actions, 'initial_state': environment.openAI.reset()}]

        run_demo(environment, agent_info_list,
                 join(dirname(realpath('__file__')), category, open_ai_gym_env.split('-')[0].lower()))
