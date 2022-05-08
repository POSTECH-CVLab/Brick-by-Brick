import numpy as np
import tensorflow as tf
from baselines.common.runners import AbstractEnvRunner

import copy

class Runner(AbstractEnvRunner):
    """
    We use this object to make a mini batch of experiences
    __init__:
    - Initialize the runner

    run():
    - Make a mini batch
    """
    def __init__(self, *, env, model, nsteps, gamma, lam):
        super().__init__(env=env, model=model, nsteps=nsteps)
        # Lambda used in GAE (General Advantage Estimation)
        self.lam = lam
        # Discount rate
        self.gamma = gamma

    def run(self, adap_tinfo=False, training=True):
        # Here, we init the lists that will contain the mb of experiences
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[]
        epinfos = []

        mb_node_feature, mb_adjacency, mb_pivot, mb_node_mask, mb_target_information, mb_edge_feature = [],[],[],[],[],[]

        mb_next_node_feature, mb_next_adjacency = [],[]

        mb_ep_rewards = []
        mb_pivot_neglogpacs = []

        num_envs = self.obs['available_actions'].shape[0]

        class_rewards, class_nodes, class_masks = [[] for i in range(10)], [[] for i in range(10)], [[] for i in range(10)]
        class_idx = [[] for i in range(10)]
        class_targets = [[] for i in range(10)]

        mb_avail = []

        # For n in range number of steps
        for _ in range(self.nsteps):
            # Given observations, get action value and neglopacs
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            obs = copy.deepcopy(self.obs)

            node_feature, adjacency, node_mask, target_information, edge_feature, target_class, current_pic, available_actions = \
                tuple([obs.get(key) for key in ['node_attributes', 'adjacency', 'node_mask', 'target_information', 'edge_attributes', 'target_class', 'current_pic', 'available_actions']])

            if adap_tinfo:
                target_information = target_information - np.logical_and(target_information, current_pic)
                obs['target_information'] = target_information

            actions, values, self.states, neglogpacs, pivot_neglogpacs, pivot = self.model.step(obs, training)
            actions = actions._numpy()
            mb_obs.append(copy.deepcopy(self.obs))
            mb_actions.append(np.squeeze(actions))
            mb_values.append(values._numpy())
            mb_neglogpacs.append(neglogpacs._numpy())
            mb_dones.append(np.squeeze(self.dones))

            pivot = pivot._numpy()

            mb_pivot_neglogpacs.append(pivot_neglogpacs._numpy())

            # available_actions = available_actions._numpy()
            mb_avail.append(available_actions)

            # node_feature, adjacency, node_mask, target_information, edge_feature, target_class, current_pic = \
            #     tuple([obs.get(key) for key in ['node_attributes', 'adjacency', 'node_mask', 'target_information', 'edge_attributes', 'target_class', 'current_pic']])

            __ = list(map(lambda x, y : x.append(y), (mb_node_feature, mb_adjacency, mb_pivot, mb_node_mask, mb_target_information, mb_edge_feature),
                                                     (node_feature, adjacency, pivot, node_mask, target_information, edge_feature)))

            """pivot : (num_envs, 1), actions : (num_envs, 1)"""

            # self.obs, rewards, self.dones, __ = self.env.step(np.concatenate([pivot, actions], axis=-1))
            self.obs, rewards, self.dones, ep_rewards = self.env.step(np.concatenate([pivot, actions], axis=-1))

            mb_rewards.append(rewards)
            mb_ep_rewards.append(ep_rewards)

            next_obs = copy.deepcopy(self.obs)
            next_node_feature, next_adjacency = tuple([next_obs.get(key) for key in ['node_attributes', 'adjacency']])

            mb_next_node_feature.append(next_node_feature)
            mb_next_adjacency.append(next_adjacency)

            ___ = list(map(lambda target_class, idx : class_rewards[target_class].append(np.expand_dims(np.array(ep_rewards[idx]), axis=-1)[None,...]),
                           target_class[:, 0], np.arange(target_class.shape[0])))

            ___ = list(map(lambda target_class, idx : class_nodes[target_class].append(node_feature[idx][None,...]),
                           target_class[:, 0], np.arange(target_class.shape[0])))

            ___ = list(map(lambda target_class, idx : class_masks[target_class].append(node_mask[idx][None,...]),
                           target_class[:, 0], np.arange(target_class.shape[0])))

            ___ = list(map(lambda target_class, idx : class_targets[target_class].append(target_information[idx][None,...]),
                           target_class[:, 0], np.arange(target_class.shape[0])))

            # import pdb; pdb.set_trace()

            ___ = list(map(lambda tc, idx : class_idx[tc].append(target_class[idx][None,...]),
                           target_class[:, 0], np.arange(target_class.shape[0])))

        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs)._numpy()

        mb_node_feature, mb_adjacency, mb_pivot, mb_node_mask, \
        mb_target_information, mb_edge_feature = list(map(lambda x : np.asarray(x), (mb_node_feature, mb_adjacency, mb_pivot, mb_node_mask,
                                                                                     mb_target_information, mb_edge_feature)))

        mb_next_node_feature = np.asarray(mb_next_node_feature)
        mb_next_adjacency = np.asarray(mb_next_adjacency)

        mb_ep_rewards = np.asarray(mb_ep_rewards, dtype=np.float32)

        mb_pivot_neglogpacs = np.asarray(mb_pivot_neglogpacs, dtype=np.float32)

        mb_class_rewards = tuple(map(lambda x : sf01(np.asarray(x)), [sub_list for sub_list in class_rewards if sub_list != []]))
        mb_class_nodes = tuple(map(lambda x : sf01(np.asarray(x)), [sub_list for sub_list in class_nodes if sub_list != []]))
        mb_class_masks = tuple(map(lambda x : sf01(np.asarray(x)), [sub_list for sub_list in class_masks if sub_list != []]))
        mb_class_targets = tuple(map(lambda x : sf01(np.asarray(x)), [sub_list for sub_list in class_targets if sub_list != []]))

        mb_class_idx = tuple(map(lambda x : sf01(np.asarray(x)), [sub_list for sub_list in class_idx if sub_list != []]))

        mb_avail = np.asarray(mb_avail, dtype=np.float32)

        # discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values

        # print(mb_returns.shape, mb_dones.shape, mb_actions.shape, mb_values.shape, mb_neglogpacs.shape)

        return (*map(sf01, (mb_node_feature, mb_adjacency, mb_pivot, mb_node_mask, mb_target_information, mb_edge_feature,
                            mb_next_node_feature, mb_next_adjacency)),
                *map(lambda x : np.expand_dims(sf01(x), axis=-1), (mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, mb_pivot_neglogpacs,
                                                                   mb_rewards, mb_ep_rewards)),
                mb_class_rewards, mb_class_nodes, mb_class_masks, mb_class_targets, mb_class_idx, sf01(mb_avail))

def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])
