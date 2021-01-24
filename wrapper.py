from gym.spaces import Box
import numpy as np
import scipy.stats as stat
import torch

# Used to represent accumulated cost in the form of discrete buckets.
def bucketize(x,n_buckets,max_x):
    out = np.zeros(n_buckets)
    for i in range(1,n_buckets+1):
        #All buckets below the cost value are set to 1.
        if x>i*max_x/n_buckets:
            out[i-1]=1
    return out

# Wrapper around the safety-gym env class
class constraint_wrapper:
    def __init__(self, env,threshold=25,
                 buckets=None):
        self.base_env = env # Use safety-gym environement as the base env
        self.buckets = buckets # no. of buckets for discretization
        # Adding cost dimension to observation space
        if self.buckets is None: # If scalar cost
            low = np.concatenate([env.observation_space.low,np.array([0])])
            high = np.concatenate([env.observation_space.high,np.array([np.inf])])
        else: # If discretized cost
            low = np.concatenate([env.observation_space.low,np.array([0 for i in range(self.buckets)])])
            high = np.concatenate([env.observation_space.high,np.array([np.inf for i in range(self.buckets)])])
        self.observation_space = Box(low=low,high=high,dtype=np.float32) # Augment observation space domain with cost domain
        self.action_space = env.action_space
        self.total_rews = [] # To store total episode returns
        self.total_costs = [] # To store total episode costs
        self.t = -1
        self.threshold = threshold # threshold value for cost

    def reset(self):
        if self.t > 0:
            self.total_rews.append(self.reward_counter) # Add total episode rewards to rewards buffer.
            self.total_costs.append(self.cost_counter) # Add total episode cost to rewards buffer.
        obs = self.base_env.reset()
        # Reset episode time, cost, and returns
        self.t = 0
        self.cost_counter = 0
        self.reward_counter = 0
        # Return new cost-augmented observation
        if self.buckets is None:
            self.obs_old = np.concatenate([obs, [self.cost_counter]])
        else:
            self.obs_old = np.concatenate([obs, bucketize(self.cost_counter,self.buckets,self.threshold)])
        return self.obs_old

    def step(self,action):
        if self.base_env.done:
            self.base_env.reset()
        obs, reward, done, info = self.base_env.step(action) # Base environment step
        self.reward_counter += reward # Update total episode reward
        self.cost_counter += info["cost"] # Update total episode cost
        self.t += 1
        # Calculate the cost adjusted reward
        # Augment observation space with accumulated cost
        if self.buckets is None:
            self.obs_old = np.concatenate([obs, [min(self.cost_counter,self.threshold+1)]])
        else:
            self.obs_old = np.concatenate([obs,bucketize(self.cost_counter,self.buckets,self.threshold)])
        return self.obs_old, reward, done, {"unsafe":int(self.cost_counter==self.threshold+1 and info["cost"]==1),"cost":info["cost"]}
    def render(self, mode='human'):
        return self.base_env.render(mode,camera_id=1)