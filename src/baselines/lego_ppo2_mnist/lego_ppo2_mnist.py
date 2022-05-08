import time
import numpy as np
import tensorflow as tf
import os.path as osp
from baselines import logger
from collections import deque
from baselines.common import explained_variance, set_global_seeds
from baselines.common.models import get_network_builder
try:
    from mpi4py import MPI
except ImportError:
    MPI = None
from baselines.lego_ppo2_mnist.runner import Runner

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

def constfn(val):
    def f(_):
        return val
    return f

import gym
import gym_lego
import os
from geometric_primitives import brick
from geometric_primitives import bricks

from baselines.common.models import register, lego_cnn_model, lego_mnist_cnn_model, lego_mnist_deep_cnn_model, build_impala_cnn

@register("lego_mnist_cnn")
def lego_mnist_cnn(**conv_kwargs):
    def network_fn(input_shape):
        return lego_mnist_cnn_model(input_shape, **conv_kwargs)
    return network_fn

@register("lego_mnist_cnn3d")
def lego_mnist_cnn3d(**conv_kwargs):
    def network_fn(input_shape):
        return lego_mnist_cnn3d_model(input_shape, **conv_kwargs)
    return network_fn

@register("impala_cnn")
def impala_cnn(**conv_kwargs):
    def network_fn(input_shape, units=64):
        return build_impala_cnn(input_shape, output_units=units, **conv_kwargs)
    return network_fn

def learn(*, network, env, total_timesteps, eval_env = None, seed=None, nsteps=2048, ent_coef=0.01, lr=3e-4,
            vf_coef=0.5,  max_grad_norm=0.5, gamma=0.99, lam=0.95,
            log_interval=1, nminibatches=4, noptepochs=1, cliprange=0.2,
            save_interval=0, load_path=None, model_fn=None,
            **network_kwargs):
    '''
    Learn policy using PPO algorithm (https://arxiv.org/abs/1707.06347)

    Parameters:
    ----------

    network:                          policy network architecture. Either string (mlp, lstm, lnlstm, cnn_lstm, cnn, cnn_small, conv_only - see baselines.common/models.py for full list)
                                      specifying the standard network architecture, or a function that takes tensorflow tensor as input and returns
                                      tuple (output_tensor, extra_feed) where output tensor is the last network layer output, extra_feed is None for feed-forward
                                      neural nets, and extra_feed is a dictionary describing how to feed state into the network for recurrent neural nets.
                                      See common/models.py/lstm for more details on using recurrent nets in policies

    env: baselines.common.vec_env.VecEnv     environment. Needs to be vectorized for parallel environment simulation.
                                      The environments produced by gym.make can be wrapped using baselines.common.vec_env.DummyVecEnv class.


    nsteps: int                       number of steps of the vectorized environment per update (i.e. batch size is nsteps * nenv where
                                      nenv is number of environment copies simulated in parallel)

    total_timesteps: int              number of timesteps (i.e. number of actions taken in the environment)

    ent_coef: float                   policy entropy coefficient in the optimization objective

    lr: float or function             learning rate, constant or a schedule function [0,1] -> R+ where 1 is beginning of the
                                      training and 0 is the end of the training.

    vf_coef: float                    value function loss coefficient in the optimization objective

    max_grad_norm: float or None      gradient norm clipping coefficient

    gamma: float                      discounting factor

    lam: float                        advantage estimation discounting factor (lambda in the paper)

    log_interval: int                 number of timesteps between logging events

    nminibatches: int                 number of training minibatches per update. For recurrent policies,
                                      should be smaller or equal than number of environments run in parallel.

    noptepochs: int                   number of training epochs per update

    cliprange: float or function      clipping range, constant or schedule function [0,1] -> R+ where 1 is beginning of the training
                                      and 0 is the end of the training

    save_interval: int                number of timesteps between saving events

    load_path: str                    path to load the model from

    **network_kwargs:                 keyword arguments to the policy / network builder. See baselines.common/policies.py/build_policy and arguments to a particular type of network
                                      For instance, 'mlp' network architecture has arguments num_hidden and num_layers.



    '''

    set_global_seeds(seed)

    if isinstance(lr, float): lr = constfn(lr)
    else: assert callable(lr)
    if isinstance(cliprange, float): cliprange = constfn(cliprange)
    else: assert callable(cliprange)
    total_timesteps = int(total_timesteps)

    # Get the nb of env
    nenvs = env.num_envs

    # Get state_space and action_space
    ob_space = env.observation_space
    ac_space = env.action_space

    hidden_dim = network_kwargs['num_hidden']
    target_hidden_dim = network_kwargs['num_hidden']

    if isinstance(network, str):
        network_type = network
        policy_network_fn = get_network_builder(network_type)(**network_kwargs)
        '''Embedded Node Feature + Embedded Edge Feature'''
        network = policy_network_fn((hidden_dim + hidden_dim + hidden_dim,))

        target_network_fn = impala_cnn()
        target_network = target_network_fn(input_shape=(14, 14, 1), units=target_hidden_dim)

    print(network.summary())
    print(target_network.summary())

    print(network_type)
    print(network_kwargs)

    # Calculate the batch_size
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches
    is_mpi_root = (MPI is None or MPI.COMM_WORLD.Get_rank() == 0)

    model_type = 'model_multignn'
    adap_tinfo = False

    # Instantiate the model object (that creates act_model and train_model)
    if model_fn is None:
        if model_type == 'model':
            print('\n', 'model', '\n')
            from baselines.lego_ppo2_mnist.model import Model
        elif model_type == 'model_multignn':
            print('\n', 'model_multignn', '\n')
            from baselines.lego_ppo2_mnist.model_multignn import Model
        model_fn = Model

    print('\n', 'adap_tinfo = {}'.format(adap_tinfo), '\n')

    model = model_fn(ac_space=ac_space, policy_network=network, target_network=target_network,
                     hidden_dim=hidden_dim, target_hidden_dim=target_hidden_dim,
                     ent_coef=ent_coef, vf_coef=vf_coef, max_grad_norm=max_grad_norm)

    if load_path is not None:
        load_path = osp.expanduser(load_path)
        ckpt = tf.train.Checkpoint(model=model)
        manager = tf.train.CheckpointManager(ckpt, load_path, max_to_keep=None)
        ckpt.restore(manager.latest_checkpoint)

    # Instantiate the runner object
    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)
    if eval_env is not None:
        eval_runner = Runner(env = eval_env, model = model, nsteps = nsteps, gamma = gamma, lam= lam)

    epinfobuf = deque(maxlen=100)
    if eval_env is not None:
        eval_epinfobuf = deque(maxlen=100)

    directories_ = ['train_coords', 'train_target_info', 'test_coords', 'test_target_info']
    seeds_ = ['42']

    if not os.path.exists('../results'):
        os.makedirs('../results')

        os.makedirs('../results/mnist')

    for dir_idx in range(len(directories_)):

        cur_dir = os.path.join('../results/mnist', directories_[dir_idx])

        for seed_idx in range(len(seeds_)):
            cur_dir_seed = os.path.join(cur_dir, seeds_[seed_idx])

            if not os.path.exists(cur_dir_seed):
                os.makedirs(cur_dir_seed)

    # Start total timer
    tfirststart = time.perf_counter()

    nupdates = total_timesteps//nbatch
    for update in range(1, nupdates+1):
        assert nbatch % nminibatches == 0
        # Start timer
        tstart = time.perf_counter()
        frac = 1.0 - (update - 1.0) / nupdates
        # Calculate the learning rate
        lrnow = lr(frac)
        cliprangenow = cliprange(frac)

        if update % log_interval == 0 and is_mpi_root: logger.info('Stepping environment...')

        # Get minibatch
        node_feature, adjacency, pivot, node_mask, target_information, edge_feature, next_node_feature, next_adjacency, \
        returns, masks, actions, values, neglogpacs, pivot_neglogpacs, \
        rewards, ep_rewards, class_rewards, class_nodes, class_masks, class_targets, class_idx, avail, = runner.run(adap_tinfo=adap_tinfo, training=True) #pylint: disable=E0632
        if eval_env is not None:
            eval_node_feature, eval_adjacency, eval_pivot, eval_node_mask, eval_target_information, eval_edge_feature, eval_next_node_feature, eval_next_adjacency, \
            eval_returns, eval_masks, eval_actions, eval_values, eval_neglogpacs, eval_pivot_neglogpacs, \
            eval_rewards, eval_ep_rewards, eval_class_rewards, eval_class_nodes, eval_class_masks, eval_class_targets, eval_class_idx, eval_avail = eval_runner.run(adap_tinfo=adap_tinfo, training=False) #pylint: disable=E0632

        for i in range(len(class_rewards)):
            train_max_reward_indices = class_rewards[i].reshape([-1]).argsort()[-3:][::-1]

            for idx, train_max_reward_idx in enumerate(train_max_reward_indices):
                train_coords = class_nodes[i][train_max_reward_idx][:int(np.sum(class_masks[i][train_max_reward_idx]))]
                train_target_info = class_targets[i][train_max_reward_idx][:,:,0]

                np.save('../results/mnist/train_coords/{}/{}_{}_{}.npy'.format(seed, update, i, idx), train_coords)
                np.save('../results/mnist/train_target_info/{}/{}_{}_{}.npy'.format(seed, update, i, idx), train_target_info)

        for i in range(len(eval_class_rewards)):
            test_max_reward_indices = eval_class_rewards[i].reshape([-1]).argsort()[-5:][::-1]

            for idx, test_max_reward_idx in enumerate(test_max_reward_indices):
                test_coords = eval_class_nodes[i][test_max_reward_idx][:int(np.sum(eval_class_masks[i][test_max_reward_idx]))]
                test_target_info = eval_class_targets[i][test_max_reward_idx][:,:,0]

                np.save('../results/mnist/test_coords/{}/{}_{}_{}.npy'.format(seed, update, i, idx), test_coords)
                np.save('../results/mnist/test_target_info/{}/{}_{}_{}.npy'.format(seed, update, i, idx), test_target_info)

        states = None

        mblossvals = []

        inds = np.arange(nbatch)
        for epoch in range(noptepochs):
            # Randomize the indexes
            np.random.shuffle(inds)
            # 0 to batch_size with batch_train_size step
            for start in range(0, nbatch, nbatch_train):
                end = start + nbatch_train
                mbinds = inds[start:end]
                obs_slices = (tf.constant(arr[mbinds]) for arr in (node_feature, adjacency, pivot, node_mask, target_information, edge_feature))
                next_obs_slices = (tf.constant(arr[mbinds]) for arr in (next_node_feature, next_adjacency))
                misc_slices = (tf.constant(arr[mbinds]) for arr in (returns, masks, actions, values, neglogpacs, pivot_neglogpacs))
                mblossvals.append(model.train(lrnow, cliprangenow, *obs_slices, *next_obs_slices, *misc_slices, tf.constant(avail[mbinds]),
                                              tf.constant(start), tf.constant(epoch), tf.constant(nbatch_train)))
        # Feedforward --> get losses --> update
        lossvals = np.mean(mblossvals, axis=0)
        # End timer
        tnow = time.perf_counter()
        # Calculate the fps (frame per second)
        fps = int(nbatch / (tnow - tstart))
        if update % log_interval == 0 or update == 1:
            # Calculates if value function is a good predicator of the returns (ev > 1)
            # or if it's just worse than predicting nothing (ev =< 0)
            ev = explained_variance(np.squeeze(values), np.squeeze(returns))
            # logger.logkv("misc/serial_timesteps", update*nsteps)
            logger.logkv("misc/nupdates", update)
            logger.logkv("misc/total_timesteps", update*nbatch)
            # logger.logkv("fps", fps)
            logger.logkv('misc/time_elapsed', tnow - tfirststart)
            logger.logkv("misc/explained_variance", float(ev))
            logger.logkv('eprew_stepwisemean', np.sum(rewards) / (np.sum(rewards > 0) + 1e-8))
            for i in range(len(class_rewards)):
                logger.logkv('eprewmean_c{}'.format(class_idx[i][0][0]), np.sum(class_rewards[i]) / (np.sum(class_rewards[i] > 0) + 1e-8) )
                logger.logkv('eprewmax_c{}'.format(class_idx[i][0][0]), np.max(class_rewards[i]))
            # logger.logkv('eprewmax', np.max(ep_rewards))
            logger.logkv('eval_eprew_stepwisemean', np.sum(eval_rewards) / (np.sum(eval_rewards > 0) + 1e-8))
            for j in range(len(eval_class_rewards)):
                logger.logkv('eval_eprewmean_c{}'.format(eval_class_idx[j][0][0]), np.sum(eval_class_rewards[j]) / (np.sum(eval_class_rewards[j] > 0) + 1e-8) )
                logger.logkv('eval_eprewmax_c{}'.format(eval_class_idx[j][0][0]), np.max(eval_class_rewards[j]))
            for (lossval, lossname) in zip(lossvals, model.loss_names):
                logger.logkv('loss/' + lossname, lossval)

            logger.dumpkvs()

    return model
# Avoid division error when calculate the mean (in our case if epinfo is empty returns np.nan, not return an error)
def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)
