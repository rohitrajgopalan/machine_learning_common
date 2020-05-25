from os.path import dirname, realpath
from examples.openai.run import run

run(dirname(realpath(__file__)), 'CartPole-v1')
