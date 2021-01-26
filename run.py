import safety_gym
import gym
from wrapper import constraint_wrapper
from spinup import td3_lagrange_pytorch,sac_lagrange_pytorch

env = constraint_wrapper(gym.make('Safexp-PointGoal1-v0'))

sac_lagrange_pytorch(lambda: env, epochs=25, steps_per_epoch=10000, start_steps=10000,
                     ac_kwargs=dict(hidden_sizes=[256,256],hidden_sizes_policy=[256,256]),num_test_episodes=0,
                                                   batch_size=100,data_aug=False,threshold=False,lambda_soft=-10000.0)