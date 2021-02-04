import numpy as np
import gym
import gym_minigrid
from gym_minigrid.envs import CrossingEnv

# Change the seed to be your full student ID
env = CrossingEnv(size=11, num_crossings=5, seed=6188068)
obs = env.reset()
render_data = env.render()
np.save('render_data.npy', render_data)
env.close()