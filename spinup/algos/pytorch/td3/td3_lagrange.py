from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import spinup.algos.pytorch.td3.core as core
from spinup.utils.logx import EpochLogger
from spinup.utils.test_policy import load_policy_and_env

def bucketize_vec(x,n_buckets,max_x):
    # Discretize cost into n_buckets buckets
    out = np.zeros((len(x),n_buckets))
    # All buckets below the accumulated cost are set to 1
    for i in range(1,n_buckets+1):
        out[:, i - 1] = (x>i*max_x/n_buckets).astype(np.int)
    return out


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, env, obs_dim, act_dim, size):
        self.env = env
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



def td3_lagrange(env_fn, actor_critic=core.MLPActorCritic,cost_critic=core.MLPCritic,ac_kwargs=dict(), seed=0,
        steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99,
        polyak=0.995, pi_lr=1e-3, q_lr=1e-3, batch_size=100, start_steps=10000,
        update_after=1000, update_every=50, act_noise=0.0, target_noise=0.4,
        noise_clip=0.5, policy_delay=2, num_test_episodes=10, max_ep_len=1000,
        logger_kwargs=dict(), save_freq=1,shift_oac=4,beta_oac=4,lambda_delay=25,n_updates=1,discor_critic=core.MLPCritic):
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3)


    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with an ``act``
            method, a ``pi`` module, a ``q1`` module, and a ``q2`` module.
            The ``act`` method and ``pi`` module should accept batches of
            observations as inputs, and ``q1`` and ``q2`` should accept a batch
            of observations and a batch of actions as inputs. When called,
            these should return:

            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each
                                           | observation.
            ``pi``       (batch, act_dim)  | Tensor containing actions from policy
                                           | given observations.
            ``q1``       (batch,)          | Tensor containing one current estimate
                                           | of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ``q2``       (batch,)          | Tensor containing the other current
                                           | estimate of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object
            you provided to TD3.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs)
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target
            networks. Target networks are updated towards main networks
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually
            close to 1.)

        pi_lr (float): Learning rate for policy.

        q_lr (float): Learning rate for Q-networks.

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.

        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long
            you wait between updates, the ratio of env steps to gradient steps
            is locked to 1.

        act_noise (float): Stddev for Gaussian exploration noise added to
            policy at training time. (At test time, no noise is added.)

        target_noise (float): Stddev for smoothing noise added to target
            policy.

        noise_clip (float): Limit for absolute value of target policy
            smoothing noise.

        policy_delay (int): Policy will only be updated once every
            policy_delay times for each update of the Q-networks.

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """
    assert (shift_oac==None and beta_oac==None) or (shift_oac!=None and beta_oac!=None)
    use_oac = (shift_oac!=None and beta_oac!=None)
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
    ac_targ = deepcopy(ac)
    ac.pi=ac.pi.to(device)
    ac_targ.pi = ac_targ.pi.to(device)
    ac.q1=ac.q1.to(device)
    ac_targ.q1 = ac_targ.q1.to(device)
    ac.q2=ac.q2.to(device)
    ac_targ.q2 = ac_targ.q2.to(device)
    cc = cost_critic(env.observation_space, env.action_space, **ac_kwargs)
    cc_targ = deepcopy(cc)
    cc.q1 = cc.q1.to(device)
    cc.q2 = cc.q2.to(device)
    cc_targ.q1 = cc_targ.q1.to(device)
    cc_targ.q2 = cc_targ.q2.to(device)

    if discor_critic is not None:
        #Use additional layer as described in discor paper.
        if "hidden_sizes" in ac_kwargs.keys():
            ac_kwargs["hidden_sizes"] = tuple(list(ac_kwargs["hidden_sizes"])+[ac_kwargs["hidden_sizes"][-1]])
        else:
            ac_kwargs["hidden_sizes"] = tuple(list(ac.hidden_sizes)+[ac.hidden_sizes[-1]])
        dr = discor_critic(env.observation_space, env.action_space, **ac_kwargs)
        dr_targ = deepcopy(dr)
        dr.q1 = dr.q1.to(device)
        dr.q2 = dr.q2.to(device)
        dr_targ.q1 = dr_targ.q1.to(device)
        dr_targ.q2 = dr_targ.q2.to(device)

        dc = discor_critic(env.observation_space, env.action_space, **ac_kwargs)
        dc_targ = deepcopy(dr)
        dc.q1 = dc.q1.to(device)
        dc.q2 = dc.q2.to(device)
        dc_targ.q1 = dc_targ.q1.to(device)
        dc_targ.q2 = dc_targ.q2.to(device)

        tao = torch.tensor(10.0)


    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False
    for p in cc_targ.parameters():
        p.requires_grad = False

    # List of parameters for both Q-networks (save this for convenience)
    if discor_critic is None:
        q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters(), cc.q1.parameters(), cc.q2.parameters())
    else:
        q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters(), cc.q1.parameters(), cc.q2.parameters(),
                                   dr.q1.parameters(), dr.q2.parameters(), dc.q1.parameters(), dc.q2.parameters())

    soft_lambda_base = torch.tensor(-10000.0, requires_grad=True)
    softplus = torch.nn.Softplus().to(device)

    # Experience buffer
    replay_buffer = ReplayBuffer(env=env, obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Count variables (protip: try to get a feel for how different size networks behave!)
    if discor_critic is None:
        var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q1, ac.q2,cc.q1,cc.q2])
        logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d, \t qc1: %d, \t qc2: %d\n'%var_counts)
    else:
        var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q1, ac.q2,cc.q1,cc.q2,dr.q1,dr.q2,dc.q1,dc.q2])
        logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d, \t qc1: %d,\t qc2: %d,\t dr1: %d,\t dr2: %d,\t dc1: %d, \t dc2: %d\n'%var_counts)

    # Set up function for computing TD3 Q-losses
    def compute_loss_q(data):
        o, a, r, c, o2, d = data['obs'], data['act'], data['rew'], data['cost'], data['obs2'], data['done']

        q1 = ac.q1(o.to(device),a.to(device))
        q2 = ac.q2(o.to(device),a.to(device))
        qc1 = cc.q1(o.to(device),a.to(device))
        qc2 = cc.q2(o.to(device),a.to(device))

        # Bellman backup for Q functions
        with torch.no_grad():

            soft_lambda = soft_lambda_base.to(device)
            lambda_var = softplus(soft_lambda)

            pi_targ = ac_targ.pi(o2.to(device))

            # Target policy smoothing
            epsilon = torch.randn_like(pi_targ) * target_noise
            epsilon = torch.clamp(epsilon, -noise_clip, noise_clip)
            a2 = pi_targ + epsilon
            a2 = torch.clamp(a2, act_limit_low, act_limit_high)

            # Target Q-values
            q1_pi_targ = ac_targ.q1(o2.to(device), a2.to(device))
            q2_pi_targ = ac_targ.q2(o2.to(device), a2.to(device))

            qc1_pi_targ = cc_targ.q1(o2.to(device), a2.to(device))
            qc2_pi_targ = cc_targ.q2(o2.to(device), a2.to(device))

            #Minimize linear combination at current tradeoff!
            select_q1 = q1_pi_targ-lambda_var*qc1_pi_targ<q2_pi_targ-lambda_var*qc2_pi_targ
            select_q2 = torch.logical_not(select_q1)
            qc_pi_targ = qc1_pi_targ*select_q1+qc2_pi_targ*select_q2
            q_pi_targ = q1_pi_targ*select_q1+q2_pi_targ*select_q2

            backup = r.to(device) + gamma * (1 - d.to(device)) * q_pi_targ
            backup_c = c.to(device) + gamma * (1 - d.to(device)) * qc_pi_targ

            if discor_critic is not None:
                nexterror_r1 = gamma * dr_targ.q1(o2.to(device), a2.to(device))
                nexterror_r2 = gamma * dr_targ.q2(o2.to(device), a2.to(device))
                nexterror_c1 = gamma * dc_targ.q1(o2.to(device), a2.to(device))
                nexterror_c2 = gamma * dc_targ.q2(o2.to(device), a2.to(device))

        # MSE loss against Bellman backup
        if discor_critic is not None:
            loss_q1 = (torch.softmax(nexterror_r1/tao,dim=0)*(q1 - backup)**2).sum()
            loss_q2 = (torch.softmax(nexterror_r2/tao,dim=0)*(q2 - backup)**2).sum()

            loss_qc1 = (torch.softmax(nexterror_c1/tao,dim=0)*(qc1 - backup_c)**2).sum()
            loss_qc2 = (torch.softmax(nexterror_c2/tao,dim=0)*(qc2 - backup_c)**2).sum()
        else:
            loss_q1 = ( (q1 - backup) ** 2).mean()
            loss_q2 = ((q2 - backup) ** 2).mean()

            loss_qc1 = ((qc1 - backup_c) ** 2).mean()
            loss_qc2 = ((qc2 - backup_c) ** 2).mean()

        loss_q = loss_q1 + loss_q2
        loss_qc = loss_qc1 + loss_qc2

        loss_q = loss_q +loss_qc

        # Useful info for logging
        loss_info = dict(Q1Vals=q1.detach().cpu().numpy(),
                         Q2Vals=q2.detach().cpu().numpy(),
                         QC1Vals=qc1.detach().cpu().numpy(),
                         QC2Vals=qc2.detach().cpu().numpy())
        return loss_q, loss_info

    def compute_loss_discor(data):
        o, a, r, c, o2, d = data['obs'], data['act'], data['rew'], data['cost'], data['obs2'], data['done']

        if discor_critic is not None:
            dr1 = dr.q1(o.to(device),a.to(device))
            dr2 = dr.q2(o.to(device), a.to(device))
            dc1 = dc.q1(o.to(device),a.to(device))
            dc2 = dc.q2(o.to(device), a.to(device))

        with torch.no_grad():
            q1 = ac.q1(o.to(device), a.to(device))
            q2 = ac.q2(o.to(device), a.to(device))
            qc1 = cc.q1(o.to(device), a.to(device))
            qc2 = cc.q2(o.to(device), a.to(device))

            soft_lambda = soft_lambda_base.to(device)
            lambda_var = softplus(soft_lambda)

            pi_targ = ac_targ.pi(o2.to(device))

            # Target policy smoothing
            epsilon = torch.randn_like(pi_targ) * target_noise
            epsilon = torch.clamp(epsilon, -noise_clip, noise_clip)
            a2 = pi_targ + epsilon
            a2 = torch.clamp(a2, act_limit_low, act_limit_high)

            # Target Q-values
            q1_pi_targ = ac_targ.q1(o2.to(device), a2.to(device))
            q2_pi_targ = ac_targ.q2(o2.to(device), a2.to(device))

            qc1_pi_targ = cc_targ.q1(o2.to(device), a2.to(device))
            qc2_pi_targ = cc_targ.q2(o2.to(device), a2.to(device))

            # Minimize linear combination at current tradeoff!
            select_q1 = q1_pi_targ - lambda_var * qc1_pi_targ < q2_pi_targ - lambda_var * qc2_pi_targ
            select_q2 = torch.logical_not(select_q1)
            qc_pi_targ = qc1_pi_targ * select_q1 + qc2_pi_targ * select_q2
            q_pi_targ = q1_pi_targ * select_q1 + q2_pi_targ * select_q2

            backup = r.to(device) + gamma * (1 - d.to(device)) * q_pi_targ
            backup_c = c.to(device) + gamma * (1 - d.to(device)) * qc_pi_targ

            # MSE loss against Bellman backup
            backup_dr1 = torch.abs(q1 - backup) + (1 - d.to(device)) * gamma * dr_targ.q1(o2.to(device), a2.to(device))
            backup_dr2 = torch.abs(q2 - backup) + (1 - d.to(device)) * gamma * dr_targ.q2(o2.to(device), a2.to(device))
            backup_dc1 = torch.abs(qc1 - backup_c) + (1 - d.to(device)) * gamma * dc_targ.q1(o2.to(device), a2.to(device))
            backup_dc2 = torch.abs(qc2 - backup_c) + (1 - d.to(device)) * gamma * dc_targ.q2(o2.to(device), a2.to(device))


        loss_dr1 = ((dr1 - backup_dr1)**2).mean()
        loss_dr2 = ((dr2 - backup_dr2) ** 2).mean()
        loss_dc1 = ((dc1 - backup_dc1) ** 2).mean()
        loss_dc2 = ((dc2 - backup_dc2) ** 2).mean()
        loss_d_q = loss_dr1+loss_dr2+loss_dc1+loss_dc2

        mean_error = 0.25*dr1.mean()+0.25*dr2.mean()+0.25*dc1.mean()+0.25*dc2.mean()

        return loss_d_q, mean_error

    # Set up function for computing TD3 pi loss
    def compute_loss_pi(data):
        o = data['obs']
        soft_lambda = soft_lambda_base.to(device)
        lambda_var = softplus(soft_lambda)
        q1_pi = ac.q1(o.to(device), ac.pi(o.to(device)))-lambda_var*cc.q1(o.to(device), ac.pi(o.to(device)))
        return -q1_pi.mean()/(1+lambda_var)

    def compute_loss_lambda(data):
        cost_limit=25
        ep_len = 1000
        o = data['obs']
        soft_lambda = soft_lambda_base.to(device)
        lambda_var = softplus(soft_lambda)
        pi = ac_targ.pi(o.to(device))
        qc = cc_targ.q1(o.to(device),pi)
        qc_constraint = cost_limit/(ep_len*(1-gamma))
        lambda_loss = (lambda_var * (qc_constraint - qc)).mean()
        lambda_info = dict(Lambda=lambda_var.detach().cpu().numpy())
        return lambda_loss, lambda_info

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    q_optimizer = Adam(q_params, lr=q_lr)
    lambda_optimizer = Adam([soft_lambda_base],lr=0.0)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update(data, timer):
        # First run one gradient descent step for Q1 and Q2

        q_optimizer.zero_grad()
        loss_q, loss_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        if discor_critic:
            loss_d_q,mean_error = compute_loss_discor(data)
            loss_d_q.backward()
            q_optimizer.step()
            tao.data = polyak*tao.data+(1-polyak)*mean_error
            logger.store(Tao=tao)

        # Record things
        logger.store(LossQ=loss_q.item(), **loss_info)

        # Possibly update pi and target networks
        if timer % policy_delay == 0:

            # Freeze Q-networks so you don't waste computational effort
            # computing gradients for them during the policy learning step.
            for p in q_params:
                p.requires_grad = False

            # Next run one gradient descent step for pi.
            pi_optimizer.zero_grad()
            loss_pi = compute_loss_pi(data)
            loss_pi.backward()
            pi_optimizer.step()


            # Unfreeze Q-networks so you can optimize it at next DDPG step.
            for p in q_params:
                p.requires_grad = True

            # Record things
            logger.store(LossPi=loss_pi.item())

            # Finally, update target networks by polyak averaging.
            with torch.no_grad():
                for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                    # NB: We use an in-place operations "mul_", "add_" to update target
                    # params, as opposed to "mul" and "add", which would make new tensors.
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)
                for p, p_targ in zip(cc.parameters(), cc_targ.parameters()):
                    # NB: We use an in-place operations "mul_", "add_" to update target
                    # params, as opposed to "mul" and "add", which would make new tensors.
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)
        if timer % lambda_delay == 0:
            for p in q_params:
                p.requires_grad = False
            lambda_optimizer.zero_grad()
            loss_lambda,lambda_info=compute_loss_lambda(data)
            loss_lambda.backward()
            lambda_optimizer.step()
            logger.store(**lambda_info)
            for p in q_params:
                p.requires_grad = True

    def get_action(o, noise_scale,use_oac=False):
        #Idea: flip coin to decide whether to only improve cost or only improve reward?
        a = ac.act(torch.as_tensor(o, dtype=torch.float32).to(device))
        if not use_oac:
            a += noise_scale * np.random.randn(act_dim)
            return np.clip(a, act_limit_low, act_limit_high)
        else:
            a=torch.tensor(a).to(device)
            a.requires_grad = True
            soft_lambda = soft_lambda_base.to(device)
            lambda_var = softplus(soft_lambda)

            q1 = ac.q1(torch.as_tensor(o, dtype=torch.float32).to(device),a)-lambda_var*cc.q1(torch.as_tensor(o, dtype=torch.float32).to(device),a)
            q2 = ac.q2(torch.as_tensor(o, dtype=torch.float32).to(device),a)-lambda_var*cc.q2(torch.as_tensor(o, dtype=torch.float32).to(device),a)
            q_mean = 0.5*(q1+q2)
            sdq = 0.5*(torch.abs(q1-q2))

            q_up = q_mean + beta_oac*sdq

            ac.q1.zero_grad()
            ac.q2.zero_grad()
            q_up.backward()
            grad_a = a.grad.data

            a_new = a + grad_a*shift_oac*noise_scale/torch.norm(grad_a)/(1+lambda_var)
            a = a_new.detach().cpu().numpy()+noise_scale * np.random.randn(act_dim)
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

        if t > start_steps:
            a = get_action(o, act_noise,use_oac=use_oac)
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
        replay_buffer.store(o, a, r, o2, d, cost)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len,EpCost=ep_cost)
            o, ep_ret, ep_cost, ep_len = env.reset(), 0, 0, 0

        # Update handling
        if t >= update_after and t % update_every == 0:
            for j in range(update_every*n_updates):
                batch = replay_buffer.sample_batch(batch_size)
                update(data=batch, timer=j)

        # End of epoch handling
        if (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs):
                logger.save_state({'env': env}, None)

            # Test the performance of the deterministic version of the agent.
            test_agent()

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('EpCost', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('TestEpCost', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('Q1Vals', with_min_and_max=True)
            logger.log_tabular('Q2Vals', with_min_and_max=True)
            logger.log_tabular('QC1Vals', with_min_and_max=True)
            logger.log_tabular('QC2Vals', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('Lambda', average_only=True)
            if discor_critic:
                logger.log_tabular('Tao', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='td3')
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    td3(lambda : gym.make(args.env), actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),
        gamma=args.gamma, seed=args.seed, epochs=args.epochs,
        logger_kwargs=logger_kwargs)

"""import gym
from gym.spaces import Box
class test_env:
    def __init__(self):
        self.observation_space = Box(low=np.array([0]), high=np.array([1]), dtype=np.float32)
        self.action_space = Box(low=np.array([0]), high=np.array([1]), dtype=np.float32)
        self.t=0
    def step(self,action):
        self.t += 1
        return np.array([0]),action,self.t%1000==-1,{"cost":2*action}
    def reset(self):
        return np.array([0])

td3_lagrange(lambda: test_env())"""


