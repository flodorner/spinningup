from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import spinup.algos.pytorch.hybrid.core as core
from spinup.utils.logx import EpochLogger
from spinup.utils.test_policy import load_policy_and_env

def bucketize_vec(x,n_buckets,max_x):
    # Discretize cost into n_buckets buckets
    out = np.zeros((len(x),n_buckets))
    # All buckets below the accumulated cost are set to 1
    for i in range(1,n_buckets+1):
        out[:, i - 1] = (x>i*max_x/n_buckets).astype(np.int)
    return out


class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, size, gamma=0.99):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.cost_buf = np.zeros(size, dtype=np.float32)
        self.cost_ret_buf = np.zeros(size, dtype=np.float32)
        self.gamma = gamma
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs,  rew, cost):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.rew_buf[self.ptr] = rew
        self.cost_buf[self.ptr] = cost

        self.ptr += 1

    def finish_path(self, last_val=0,last_val_cost=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        costs = np.append(self.cost_buf[path_slice], last_val_cost)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]
        self.cost_ret_buf[path_slice] = core.discount_cumsum(costs, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer. Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        data = dict(obs=self.obs_buf,  ret=self.ret_buf, cost_ret=self.cost_ret_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.cost_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done, cost):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.cost_buf[self.ptr] = cost
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        # Note in case we pursue this further after the submission: Get rid of the wrapper and do everything in here!
        # This way, we can also implement adaptive penalties to get around the tuning problem.
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     cost=self.cost_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}



def hybrid_lagrange(env_fn, actor_critic=core.MLPActorCritic,cost_critic=core.MLPCritic,ac_kwargs=dict(), seed=0,
        steps_per_epoch=10000, epochs=100, replay_size=int(1e6), gamma=0.99,
        lr=1e-3,batch_size=100,start_steps=10000,
        update_after=1000, update_every=50,v_update_every=1000, act_noise=0.1,
        num_test_episodes=0, max_ep_len=1000,
        logger_kwargs=dict(), save_freq=1,lambda_delay=25,n_updates=1,train_v_iters=20,
        lambda_soft=0.0):

    #Can I get faster/better V-estimates? Options:  mix in off-policy data (Laser/Vtrace)?
    #Use dedicated exploration episodes that don't feed directly into V? (And disable the normal action noise?)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    torch.manual_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit_high = env.action_space.high[0]
    act_limit_low = env.action_space.low[0]

    # Create actor-critic module and target networks
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    ac.pi=ac.pi.to(device)
    ac.a=ac.a.to(device)
    ac.v = ac.v.to(device)

    cc = cost_critic(env.observation_space, env.action_space, **ac_kwargs)
    cc.a = cc.a.to(device)
    cc.v = cc.v.to(device)


    # List of parameters for both Q-networks (save this for convenience)
    a_params = itertools.chain(ac.a.parameters(), cc.a.parameters())
    v_params = itertools.chain(ac.v.parameters(), cc.v.parameters())

    soft_lambda_base = torch.tensor(float(lambda_soft), requires_grad=True)
    softplus = torch.nn.Softplus().to(device)

    # Experience buffer
    replay_buffer_advantage = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)
    replay_buffer_value = PPOBuffer(obs_dim=obs_dim, size=v_update_every,gamma=gamma)


    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.a, ac.v,cc.a,cc.v])
    logger.log('\nNumber of parameters: \t pi: %d, \t a: %d, \t v: %d, \t ac: %d, \t vc: %d\n'%var_counts)


    # Set up function for computing TD3 Q-losses
    def compute_loss_q(data):
        o, a, r, c, o2, d = data['obs'], data['act'],data['rew'], data['cost'], data['obs2'], data['done']

        ad = ac.a(o.to(device),a.to(device))
        ad_c = cc.a(o.to(device),a.to(device))

        # Bellman backup for Q functions
        with torch.no_grad():
            target = gamma*ac.v(o2.to(device))+r.to(device)-ac.v(o.to(device))
            target_c = gamma*cc.v(o2.to(device))+c.to(device)-cc.v(o.to(device))

        # MSE loss against Bellman backup

        loss_ad = ((ad - target) ** 2).mean()
        loss_ad_c = ((ad_c - target_c) ** 2).mean()
        loss_total = loss_ad+loss_ad_c

        # Useful info for logging
        loss_info = dict(ADVals=ad.detach().cpu().numpy(),
                         ADCVals=ad_c.detach().cpu().numpy(),
                         ADLoss = loss_ad.detach().cpu().numpy(),
                         ADCLoss = loss_ad_c.detach().cpu().numpy()
                         )

        return loss_total, loss_info


    # Set up function for computing TD3 pi loss
    def compute_loss_pi(data):
        o = data['obs']
        soft_lambda = soft_lambda_base.to(device)
        lambda_var = softplus(soft_lambda)
        q1_pi = ac.a(o.to(device), ac.pi(o.to(device)))-lambda_var*cc.a(o.to(device), ac.pi(o.to(device)))
        return -q1_pi.mean()/(1+lambda_var)

    def compute_loss_lambda(data):
        cost_limit=25
        ep_len = 1000
        o = data['obs']
        soft_lambda = soft_lambda_base.to(device)
        lambda_var = softplus(soft_lambda)
        vc = cc.v(o.to(device))
        vc_constraint = cost_limit/(ep_len*(1-gamma))
        lambda_loss = (lambda_var * (vc_constraint - vc)).mean()
        lambda_info = dict(Lambda=lambda_var.detach().cpu().numpy())
        return lambda_loss, lambda_info

    def compute_loss_v(data):
        obs, ret, cost_ret = data['obs'], data['ret'], data["cost_ret"]
        v = ac.v(obs.to(device))
        vc = cc.v(obs.to(device))
        loss_v = ((v - ret.to(device))**2).mean()
        loss_v_c = ((vc - cost_ret.to(device))**2).mean()
        loss_total = loss_v+loss_v_c

        # Useful info for logging
        loss_info = dict(VVals=v.detach().cpu().numpy(),
                         VCVals=vc.detach().cpu().numpy(),
                         VLoss = loss_v.detach().cpu().numpy(),
                         VCLoss = loss_v_c.detach().cpu().numpy()
                         )
        return loss_total,loss_info

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(ac.pi.parameters(), lr=lr)
    a_optimizer = Adam(a_params, lr=lr)
    v_optimizer = Adam(v_params, lr=lr)
    lambda_optimizer = Adam([soft_lambda_base],lr=0.0)

    # Set up model saving
    logger.setup_pytorch_saver([ac,cc])

    def update(data, timer):
        # First run one gradient descent step for Q1 and Q2
        a_optimizer.zero_grad()
        total_loss, loss_info = compute_loss_q(data)
        total_loss.backward()
        a_optimizer.step()
        # Record things
        logger.store(**loss_info)

        # Freeze A-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in a_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        pi_optimizer.zero_grad()
        loss_pi = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()

        if timer % lambda_delay == 0:
            lambda_optimizer.zero_grad()
            loss_lambda, lambda_info = compute_loss_lambda(data)
            loss_lambda.backward()
            lambda_optimizer.step()
            logger.store(**lambda_info)
        # Unfreeze A-networks so you can optimize it at next  step.
        for p in a_params:
            p.requires_grad = True

        # Record things
        logger.store(LossPi=loss_pi.item())

    def update_v():
        data = replay_buffer_value.get()
        # Value function learning
        for i in range(train_v_iters):
            v_optimizer.zero_grad()
            loss_v,loss_info = compute_loss_v(data)
            loss_v.backward()
            v_optimizer.step()
            logger.store(**loss_info)
        return None


    def get_action(o, noise_scale):
        #Idea: flip coin to decide whether to only improve cost or only improve reward?
        a = ac.act(torch.as_tensor(o, dtype=torch.float32).to(device))
        a += noise_scale * np.random.randn(act_dim)
        return np.clip(a, act_limit_low, act_limit_high)



    def test_agent():
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_cost, ep_len = test_env.reset(), False, 0, 0, 0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                o, r, d, info = test_env.step(get_action(o, 0))
                ep_ret += r
                ep_cost += info.get("cost",0)
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len,TestEpCost=ep_cost)
        test_env.reset()

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    o, ep_ret, ep_cost, ep_len= env.reset(), 0, 0,0

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards,
        # use the learned policy (with some noise, via act_noise).
        if t>start_steps:
            a = get_action(o, act_noise)
        else:
            a = env.action_space.sample()
        # Step the env
        o2, r, d, info  = env.step(a)
        cost = info.get("cost",0)
        ep_ret += r
        ep_cost += cost
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==max_ep_len else d
        # Store experience to replay buffer
        replay_buffer_advantage.store(o, a, r, o2, d, cost)
        if t>start_steps:
            replay_buffer_value.store(o, r, cost)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2

        if t % v_update_every == v_update_every-1 and t>start_steps:
            #Assumes no terminal states...
            v = ac.v(torch.as_tensor(o, dtype=torch.float32).to(device)).detach().cpu().numpy()
            vc = cc.v(torch.as_tensor(o, dtype=torch.float32).to(device)).detach().cpu().numpy()
            replay_buffer_value.finish_path(v, vc)
            update_v()

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            #update value function
            logger.store(EpRet=ep_ret, EpLen=ep_len,EpCost=ep_cost)
            o, ep_ret, ep_cost, ep_len = env.reset(), 0, 0, 0

        # Update handling
        if t >= update_after and t % update_every == 0 and t>start_steps:
            for j in range(update_every*n_updates):
                batch = replay_buffer_advantage.sample_batch(batch_size)
                update(data=batch, timer=j)

        # End of epoch handling
        if (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs):
                logger.save_state({'env': env}, None)

            # Test the performance of the deterministic version of the agent.
            test_agent()
            a = env.action_space.sample()

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('EpCost', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            if num_test_episodes>0:
                logger.log_tabular('TestEpRet', with_min_and_max=True)
                logger.log_tabular('TestEpCost', with_min_and_max=True)
                logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('ADVals', with_min_and_max=True)
            logger.log_tabular('ADCVals', with_min_and_max=True)
            logger.log_tabular('VVals', with_min_and_max=True)
            logger.log_tabular('VCVals', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('ADLoss', average_only=True)
            logger.log_tabular('ADCLoss', average_only=True)
            logger.log_tabular('VLoss', average_only=True)
            logger.log_tabular('VCLoss', average_only=True)
            logger.log_tabular('Lambda', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()


"""
import gym
from gym.spaces import Box
class test_env:
    def __init__(self):
        self.observation_space = Box(low=np.array([0]), high=np.array([1]), dtype=np.float32)
        self.action_space = Box(low=np.array([0]), high=np.array([1]), dtype=np.float32)
        self.t=0
    def step(self,action):
        self.t += 1
        return np.array([0]),0.01*action,self.t%1000==-1,{"cost":0.02*action}
    def reset(self):
        return np.array([0])

hybrid_lagrange(lambda: test_env(),ac_kwargs={"hidden_sizes":[10,10],"hidden_sizes_policy":[10,10]},lambda_soft=-100)"""


