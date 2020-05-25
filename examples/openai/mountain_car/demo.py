from os.path import dirname, realpath

from examples.openai.run import run

run(dirname(realpath(__file__)), 'MountainCar-v0')
run(dirname(realpath(__file__)), 'MountainCarContinuous-v0')
