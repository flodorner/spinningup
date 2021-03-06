from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
from spinup.utils.test_policy import load_policy_and_env
import spinup.algos.pytorch.sac.core as core
from spinup.utils.logx import EpochLogger


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

    def __init__(self, env, obs_dim, act_dim, size, data_aug=False, p_var=10):
        self.env = env
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.cost_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        self.data_aug = data_aug
        self.p_var = p_var
        self.threshold = env.threshold

    def store(self, obs, act, rew, next_obs, done, cost):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.cost_buf[self.ptr] = cost
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        # Note in case we pursue this further after the submission: Get rid of the wrapper and do everything in here!
        # This way, we can also implement adaptive penalties to get around the tuning problem.
        idxs = np.random.randint(0, self.size, size=batch_size)
        obs = np.zeros(self.obs_buf[idxs].shape) + self.obs_buf[idxs]
        obs2 = np.zeros(self.obs2_buf[idxs].shape) + self.obs2_buf[idxs]
        rew = np.zeros(self.rew_buf[idxs].shape) + self.rew_buf[idxs]
        cost = self.cost_buf[idxs]
        # Implement data augmentation for the known cost dynamics
        if self.data_aug:
            buckets = self.env.buckets
            assert buckets is None or buckets==self.threshold+1
            if buckets is None:
                total_cost = obs[:, -1]
            else:
                total_cost = np.sum(obs[:, -buckets:],axis=-1)
            total_cost_2 = np.minimum(total_cost + cost, self.threshold + 1)

            # Perturb the cost to generate new data
            p = np.random.randint(-self.p_var , self.p_var+1)
            p = np.maximum(-1 * np.minimum(total_cost, total_cost_2),p)
            p = np.minimum((self.threshold + 1 - np.maximum(total_cost, total_cost_2)),p)

            if buckets is None:
                obs[:, -1] = np.minimum(total_cost + p, self.threshold + 1)
                obs2[:, -1] =  np.minimum(total_cost_2 + p, self.threshold + 1)
            else:
                obs[:, -buckets:] = bucketize_vec(total_cost + p, buckets, self.threshold)
                obs2[:, -buckets:] = bucketize_vec(total_cost_2 + p, buckets, self.threshold)



            rew -= rew * np.logical_and(total_cost_2 + p > self.threshold,total_cost_2 <= self.threshold) * (1-self.env.mult_penalty)
            #Apply reward multiplier if constraint is broken due to data augmenation

            rew -= self.env.add_penalty*np.logical_and(np.logical_and(total_cost_2 + p > self.threshold,total_cost + p <= self.threshold)
                                                       ,np.logical_not(np.logical_and(total_cost_2 > self.threshold,total_cost <= self.threshold)))
            # Add_penalty only gets added if threshold is broken at this step. Also, we need to check whether it already has been added!

            rew -= self.env.cost_penalty * cost * np.logical_and(total_cost_2 + p > self.threshold,
                                                                 total_cost_2 <= self.threshold)
            # Only add cost penalty when total costs are above threshold but had not been without augmentation


            rew += self.env.add_penalty*np.logical_and(np.logical_and(total_cost_2  > self.threshold,total_cost <= self.threshold)
                                                       ,np.logical_not(np.logical_and(total_cost_2 + p > self.threshold,total_cost + p <= self.threshold)))
            # Similarly, we need to remove the penalty when the augmentation causes the threshold not to be broken at a step.

            rew += self.env.cost_penalty * cost * np.logical_and(total_cost_2  > self.threshold, total_cost_2 + p <= self.threshold)
            # Again, if we were above the threshold but are not anymore with the augmenation, we need to remove the cost penalty.

            rew += np.logical_and(total_cost_2  > self.threshold, total_cost_2 + p <= self.threshold) * rew * ((1/self.env.mult_penalty)-1)

            # To get the back the base-environment reward, we need to reverse the multiplicative penalty when the constraint was originally broken


        batch = dict(obs=obs,
                     obs2=obs2,
                     act=self.act_buf[idxs],
                     rew=rew,
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}



def sac(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0,
        steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99,
        polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, start_steps=10000,
        update_after=1000, update_every=50, num_test_episodes=10, max_ep_len=1000, data_aug=False,
        logger_kwargs=dict(), save_freq=1,entropy_constraint=None,collector_policy=None):

    """
    Soft Actor-Critic (SAC)


    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with an ``act``
            method, a ``pi`` module, a ``q1`` module, and a ``q2`` module.
            The ``act`` method and ``pi`` module should accept batches of
            observations as inputs, and ``q1`` and ``q2`` should accept a batch
            of observations and a batch of actions as inputs. When called,
            ``act``, ``q1``, and ``q2`` should return:

            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each
                                           | observation.
            ``q1``       (batch,)          | Tensor containing one current estimate
                                           | of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ``q2``       (batch,)          | Tensor containing the other current
                                           | estimate of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ===========  ================  ======================================

            Calling ``pi`` should return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Tensor containing actions from policy
                                           | given observations.
            ``logp_pi``  (batch,)          | Tensor containing log probabilities of
                                           | actions in ``a``. Importantly: gradients
                                           | should be able to flow back into ``a``.
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object
            you provided to SAC.

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

        lr (float): Learning rate (used for both policy and value learning).

        alpha (float): Entropy regularization coefficient. (Equivalent to
            inverse of reward scale in the original SAC paper.)

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

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

        entropy_constraint (float): Minimum expected entropy as described in https://arxiv.org/pdf/1812.05905.pdf (either this or alpha should be set to 0).
        collector policy (string): directory containing a policy for experience collection (replaces the learnt policy)
    """
    assert alpha==None or entropy_constraint==None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())
    if not collector_policy==None:
        _, collector_policy = load_policy_and_env(collector_policy)



    torch.manual_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Create actor-critic module and target networks
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    ac_targ = deepcopy(ac)
    ac.q1 = ac.q1.to(device)
    ac.q2 = ac.q2.to(device)
    ac.pi = ac.pi.to(device)
    ac_targ.q1 = ac_targ.q1.to(device)
    ac_targ.q2 = ac_targ.q2.to(device)
    ac_targ.pi = ac_targ.pi.to(device)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False

    # List of parameters for both Q-networks (save this for convenience)
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())

    # Experience buffer
    replay_buffer = ReplayBuffer(env=env, obs_dim=obs_dim, act_dim=act_dim, size=replay_size, data_aug=data_aug)

    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q1, ac.q2])
    logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n'%var_counts)

    if not entropy_constraint==None:
        soft_alpha_base = torch.tensor(0.0, requires_grad=True)
        softplus = torch.nn.Softplus().to(device)

    # Set up function for computing SAC Q-losses
    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q1 = ac.q1(o.to(device),a.to(device))
        q2 = ac.q2(o.to(device),a.to(device))
        if not entropy_constraint is None:
            soft_alpha = soft_alpha_base.to(device)
            alpha_var = softplus(soft_alpha)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = ac.pi(o2.to(device))

            # Target Q-values
            q1_pi_targ = ac_targ.q1(o2.to(device), a2.to(device))
            q2_pi_targ = ac_targ.q2(o2.to(device), a2.to(device))
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            if not entropy_constraint is None:
                backup = r.to(device) + gamma * (1 - d.to(device)) * (q_pi_targ - alpha_var * logp_a2)
            else:
                backup = r.to(device) + gamma * (1 - d.to(device)) * (q_pi_targ - alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach().cpu().numpy(),
                      Q2Vals=q2.detach().cpu().numpy())

        return loss_q, q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(data):
        o = data['obs']
        if not entropy_constraint is None:
            soft_alpha = soft_alpha_base.to(device)
            alpha_var = softplus(soft_alpha)
        pi, logp_pi = ac.pi(o.to(device))
        q1_pi = ac.q1(o.to(device), pi)
        q2_pi = ac.q2(o.to(device), pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        if not entropy_constraint is None:
            loss_pi = (alpha_var * logp_pi - q_pi).mean()
        else:
            loss_pi = (alpha * logp_pi - q_pi).mean()



        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().cpu().numpy())

        return loss_pi, pi_info

    def compute_loss_alpha(data):
        soft_alpha = soft_alpha_base.to(device)
        alpha = softplus(soft_alpha)
        o = data['obs']
        pi, logp_pi = ac.pi(o.to(device))
        pi_entropy = -logp_pi.mean()
        loss_alpha = -alpha*(entropy_constraint*act_dim-pi_entropy)
        alpha_info = dict(Alpha=alpha.detach().cpu().numpy())
        return loss_alpha, alpha_info


    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(ac.pi.parameters(), lr=lr)
    q_optimizer = Adam(q_params, lr=lr)
    if not entropy_constraint is None:
        alpha_optimizer = Adam([soft_alpha_base], lr=lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update(data):
        # First run one gradient descent step for Q1 and Q2
        q_optimizer.zero_grad()
        loss_q, q_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        # Record things
        logger.store(LossQ=loss_q.item(), **q_info)

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        pi_optimizer.zero_grad()
        loss_pi, pi_info = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()

        if not entropy_constraint is None:
            alpha_optimizer.zero_grad()
            loss_alpha, alpha_info = compute_loss_alpha(data)
            loss_alpha.backward()
            alpha_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in q_params:
            p.requires_grad = True

        # Record things
        logger.store(LossPi=loss_pi.item(), **pi_info)
        if not entropy_constraint is None:
            logger.store(LossAlpha=loss_alpha.item(), **alpha_info)
        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    def get_action(o, deterministic=False):
        return ac.act(torch.as_tensor(o, dtype=torch.float32).to(device),
                      deterministic)

    def test_agent():
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time
                o, r, d, _ = test_env.step(get_action(o, True))
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)
        test_env.reset()
    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    o, ep_ret, ep_len= env.reset(), 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):

        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards,
        # use the learned policy.
        if t > start_steps:
            if collector_policy is None:
                a = get_action(o)
            else:
                a = collector_policy(torch.as_tensor(o, dtype=torch.float32).to(device))
        else:
            a = env.action_space.sample()

        # Step the env
        o2, r, d, info = env.step(a)
        cost = info["cost"]
        ep_ret += r
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
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, ep_ret, ep_len = env.reset(), 0, 0

        # Update handling
        if t >= update_after and t % update_every == 0:
            for j in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                update(data=batch)

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
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('Q1Vals', with_min_and_max=True)
            logger.log_tabular('Q2Vals', with_min_and_max=True)
            logger.log_tabular('LogPi', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ', average_only=True)

            logger.log_tabular('Time', time.time()-start_time)
            if not entropy_constraint is None:
                logger.log_tabular('Alpha', average_only=True)
                logger.log_tabular('LossAlpha', average_only=True)
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
    parser.add_argument('--exp_name', type=str, default='sac')
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    torch.set_num_threads(torch.get_num_threads())

    sac(lambda : gym.make(args.env), actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),
        gamma=args.gamma, seed=args.seed, epochs=args.epochs,
        logger_kwargs=logger_kwargs)
