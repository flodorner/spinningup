import safety_gym
import gym
from spinup import td3_lagrange_pytorch

env = gym.make('Safexp-PointGoal1-v0')

td3_lagrange_pytorch(lambda: env, epochs=1000, steps_per_epoch=10000, start_steps=10000,
                     act_noise=0.0, ac_kwargs=dict(hidden_sizes=[256,256],hidden_sizes_policy=[256,256]),num_test_episodes=10,
                                                   batch_size=100, shift_oac=4,beta_oac=4)