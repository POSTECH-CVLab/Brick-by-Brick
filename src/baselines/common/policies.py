import tensorflow as tf
from baselines.a2c.utils import fc, fc_relu
from baselines.common.distributions import make_pdtype

import gym
import numpy as np

class PolicyWithValue(tf.Module):
    """
    Encapsulates fields and methods for RL policy and value function estimation with shared parameters
    """

    def __init__(self, ac_space, policy_network, value_network=None, estimate_q=False):
        """
        Parameters:
        ----------
        ac_space        action space

        policy_network  keras network for policy

        value_network   keras network for value

        estimate_q      q value or v value

        """

        self.policy_network = policy_network
        self.value_network = value_network or policy_network
        self.estimate_q = estimate_q
        self.initial_state = None

        self.node_feature_extractor_extended  = fc(128, 'f_e', 128, init_scale=np.sqrt(2))

        self.pivot_classifier = fc(256, 'piv', 1, init_scale=np.sqrt(2))

        # Based on the action space, will select what probability distribution type
        self.pdtype = make_pdtype((None, 256), ac_space, init_scale=np.sqrt(2))

        if estimate_q:
            assert isinstance(ac_space, gym.spaces.Discrete)
            self.value_fc = fc(self.value_network.output_shape, 'q', ac_space.n, init_scale=np.sqrt(2))
        else:
            self.value_fc = fc(self.value_network.output_shape, 'vf', 1, init_scale=np.sqrt(2))

    # @tf.function
    def step(self, observation):
        """
        Compute next action(s) given the observation(s)

        Parameters:
        ----------

        observation     batched observation data

        Returns:
        -------
        (action, value estimate, next state, negative log likelihood of the action under current policy parameters) tuple
        """
        node_feature, adjacency, available_actions = tuple([observation.get(key) for key in ['node_attributes', 'adjacency', 'available_actions']])

        node_latent_feature = tf.linalg.matmul(adjacency, self.policy_network(node_feature))

        node_latent_feature = tf.linalg.matmul(adjacency, self.node_feature_extractor_extended(node_latent_feature))

        global_graph_feature = tf.reduce_mean(node_latent_feature, axis = 1)

        for_pivot_node_latent_feature = tf.concat([node_latent_feature, tf.tile(tf.expand_dims(global_graph_feature, axis=1),
                                                                               (1, node_latent_feature.shape[1], 1))], axis=-1)

        for_pivot_node_logits = tf.squeeze(self.pivot_classifier(for_pivot_node_latent_feature), axis=2)

        for_pivot_mask = tf.cast(tf.reduce_sum(available_actions, axis = 2) == 0, dtype=for_pivot_node_logits.dtype) * (-1e8)

        masked_for_pivot_node_logits = for_pivot_node_logits + for_pivot_mask

        pivot_node_index = tf.argmax(masked_for_pivot_node_logits, axis=1)

        pivot_node_index_logits = tf.math.reduce_max(masked_for_pivot_node_logits, axis=1)

        node_latent_representation = []

        for i in range(node_latent_feature.shape[0]):
            # node_latent_representation.append(tf.concat([node_latent_feature[i][pivot_node_index[i]], global_graph_feature[i]], axis=-1))
            node_latent_representation.append(node_latent_feature[i][pivot_node_index[i]])

        node_latent_representation = tf.stack(node_latent_representation)

        per_pivot_available_actions = []

        for i in range(available_actions.shape[0]):
            per_pivot_available_actions.append(available_actions[i][pivot_node_index[i]])

        per_pivot_available_actions = tf.stack(per_pivot_available_actions)

        # node_latent_representation = tf.concat([tf.gather_nd(node_latent_feature, pivot_node_index[:, None], batch_dims=1), global_graph_feature], axis = -1)
        #
        # per_pivot_available_actions = tf.gather_nd(available_actions, pivot_node_index[:, None], batch_dims=1)

        pd, pi = self.pdtype.pdfromlatent(node_latent_representation)

        negative_logits_mask = (tf.ones_like(per_pivot_available_actions) - per_pivot_available_actions) * int(-1e8)

        masked_action_logits = pi + tf.cast(negative_logits_mask, pi.dtype)

        action = tf.random.categorical(masked_action_logits, 1)

        # action = tf.constant(np.array([[10],[10],[10],[10],[10],[10],[10],[10],
        #                                [10],[10],[10],[10],[10],[10],[10],[10]]))

        # action = pd.sample()
        neglogp = pd.neglogp(action)

        value_node_latent_feature = tf.linalg.matmul(adjacency, self.value_network(node_feature))
        value_node_latent_feature = tf.linalg.matmul(adjacency, self.node_feature_extractor_extended(value_node_latent_feature))
        value_global_graph_feature = tf.reduce_mean(value_node_latent_feature, axis = 1)

        vf = tf.squeeze(self.value_fc(value_global_graph_feature), axis=1)
        # return action, vf, None, neglogp, pivot_node_index[:, None]

        return tf.cast(action, dtype=tf.int32), pivot_node_index_logits, None, neglogp, tf.cast(pivot_node_index[:, None], dtype=tf.int32)

    # @tf.function
    def value(self, observation):
        """
        Compute value estimate(s) given the observation(s)

        Parameters:
        ----------

        observation     observation data (either single or a batch)

        Returns:
        -------
        value estimate
        """

        node_feature, adjacency, available_actions = tuple([observation.get(key) for key in ['node_attributes', 'adjacency', 'available_actions']])

        value_node_latent_feature = tf.linalg.matmul(adjacency, self.value_network(node_feature))
        value_node_latent_feature = tf.linalg.matmul(adjacency, self.node_feature_extractor_extended(value_node_latent_feature))
        value_global_graph_feature = tf.reduce_mean(value_node_latent_feature, axis = 1)

        for_pivot_node_latent_feature = tf.concat([value_node_latent_feature, tf.tile(tf.expand_dims(value_global_graph_feature, axis=1),
                                                                              (1, value_node_latent_feature.shape[1], 1))], axis=-1)

        for_pivot_node_logits = tf.squeeze(self.pivot_classifier(for_pivot_node_latent_feature), axis=2)

        for_pivot_mask = tf.cast(tf.reduce_sum(available_actions, axis = 2) == 0, dtype=for_pivot_node_logits.dtype) * (-1e8)

        masked_for_pivot_node_logits = for_pivot_node_logits + for_pivot_mask

        pivot_node_index = tf.argmax(masked_for_pivot_node_logits, axis=1)

        pivot_node_index_logits = tf.math.reduce_max(masked_for_pivot_node_logits, axis=1)

        # result = tf.squeeze(self.value_fc(value_global_graph_feature), axis=1)
        # return result

        return pivot_node_index_logits

class PolicyWithValue_Masked(tf.Module):
    """
    Encapsulates fields and methods for RL policy and value function estimation with shared parameters
    """

    def __init__(self, ac_space, policy_network, value_network=None, estimate_q=False):
        """
        Parameters:
        ----------
        ac_space        action space

        policy_network  keras network for policy

        value_network   keras network for value

        estimate_q      q value or v value

        """

        self.policy_network = policy_network
        self.value_network = value_network or policy_network
        self.estimate_q = estimate_q
        self.initial_state = None

        self.node_feature_extractor_extended  = fc(128, 'f_e', 128, init_scale=np.sqrt(2))

        # self.pivot_classifier = fc(128, 'piv', 1, init_scale=np.sqrt(2))
        # self.pivot_classifier = fc(256, 'piv', 1, init_scale=np.sqrt(2))
        self.pivot_classifier = fc(128 + 15, 'piv', 1, init_scale=np.sqrt(2))

        # Based on the action space, will select what probability distribution type
        # self.pdtype = make_pdtype((None, 256), ac_space, init_scale=np.sqrt(2))
        # self.pdtype = make_pdtype((None, 128), ac_space, init_scale=np.sqrt(2))
        self.pdtype = make_pdtype((None, 128 + 15), ac_space, init_scale=np.sqrt(2))

        if estimate_q:
            assert isinstance(ac_space, gym.spaces.Discrete)
            self.value_fc = fc(self.value_network.output_shape, 'q', ac_space.n, init_scale=np.sqrt(2))
        else:
            self.value_fc = fc(self.value_network.output_shape, 'vf', 1, init_scale=np.sqrt(2))

    @tf.function
    def step(self, observation):
        """
        Compute next action(s) given the observation(s)

        Parameters:
        ----------

        observation     batched observation data

        Returns:
        -------
        (action, value estimate, next state, negative log likelihood of the action under current policy parameters) tuple
        """
        node_feature, adjacency, available_actions, node_mask, target_information = tuple([observation.get(key) for key in ['node_attributes', 'adjacency', 'available_actions',
                                                                                                                          'node_mask', 'target_information']])

        tiled_target_information = tf.tile(tf.expand_dims(target_information, axis=1), (1, node_feature.shape[1], 1))

        # node_latent_feature = tf.linalg.matmul(adjacency, self.policy_network(tf.concat([node_feature, tiled_target_information], axis=-1)))
        node_latent_feature = tf.linalg.matmul(adjacency, self.policy_network(node_feature))

        '''Shape : (num_envs, max_node, hidden_dim)'''
        node_latent_feature = tf.linalg.matmul(adjacency, self.node_feature_extractor_extended(node_latent_feature))

        # global_graph_feature = tf.reduce_mean(node_latent_feature, axis = 1)
        global_graph_feature = tf.math.divide(tf.reduce_sum(tf.multiply(node_latent_feature, node_mask), axis = 1),
                                              tf.reduce_sum(node_mask, axis = 1))


        for_pivot_node_latent_feature = tf.concat([node_latent_feature, tf.tile(tf.expand_dims(global_graph_feature, axis=1),
                                                                               (1, node_latent_feature.shape[1], 1))], axis=-1)

        # for_pivot_node_logits = tf.squeeze(self.pivot_classifier(for_pivot_node_latent_feature), axis=2)
        # for_pivot_node_logits = tf.squeeze(self.pivot_classifier(node_latent_feature), axis=2)
        for_pivot_node_logits = tf.squeeze(self.pivot_classifier(tf.concat([node_latent_feature, tiled_target_information], axis=-1)), axis=2)

        for_pivot_mask = tf.cast(tf.reduce_sum(available_actions, axis = 2) == 0, dtype=for_pivot_node_logits.dtype) * (-1e8)

        masked_for_pivot_node_logits = for_pivot_node_logits + for_pivot_mask

        pivot_node_index = tf.argmax(masked_for_pivot_node_logits, axis=1)

        pivot_node_index_logits = tf.math.reduce_max(masked_for_pivot_node_logits, axis=1)

        node_latent_representation = []

        for i in range(node_latent_feature.shape[0]):
            # node_latent_representation.append(tf.concat([node_latent_feature[i][pivot_node_index[i]], global_graph_feature[i]], axis=-1))
            node_latent_representation.append(node_latent_feature[i][pivot_node_index[i]])

        '''Shape : (num_envs, hidden_dim)'''
        node_latent_representation = tf.stack(node_latent_representation)

        per_pivot_available_actions = []

        for i in range(available_actions.shape[0]):
            per_pivot_available_actions.append(available_actions[i][pivot_node_index[i]])

        per_pivot_available_actions = tf.stack(per_pivot_available_actions)

        # node_latent_representation = tf.concat([tf.gather_nd(node_latent_feature, pivot_node_index[:, None], batch_dims=1), global_graph_feature], axis = -1)
        #
        # per_pivot_available_actions = tf.gather_nd(available_actions, pivot_node_index[:, None], batch_dims=1)

        # pd, pi = self.pdtype.pdfromlatent(node_latent_representation)
        pd, pi = self.pdtype.pdfromlatent(tf.concat([node_latent_representation, target_information], axis=-1))

        negative_logits_mask = (tf.ones_like(per_pivot_available_actions) - per_pivot_available_actions) * int(-1e8)

        masked_action_logits = pi + tf.cast(negative_logits_mask, pi.dtype)

        action = tf.random.categorical(masked_action_logits, 1)

        # action = tf.constant(np.array([[10],[10],[10],[10],[10],[10],[10],[10],
        #                                [10],[10],[10],[10],[10],[10],[10],[10]]))

        # action = pd.sample()
        neglogp = pd.neglogp(action)

        # value_node_latent_feature = tf.linalg.matmul(adjacency, self.value_network(tf.concat([node_feature, tiled_target_information], axis=-1)))
        value_node_latent_feature = tf.linalg.matmul(adjacency, self.value_network(node_feature))
        value_node_latent_feature = tf.linalg.matmul(adjacency, self.node_feature_extractor_extended(value_node_latent_feature))
        value_global_graph_feature = tf.reduce_mean(value_node_latent_feature, axis = 1)

        vf = tf.squeeze(self.value_fc(value_global_graph_feature), axis=1)
        # return action, vf, None, neglogp, pivot_node_index[:, None]

        return tf.cast(action, dtype=tf.int32), pivot_node_index_logits, None, neglogp, tf.cast(pivot_node_index[:, None], dtype=tf.int32)

    @tf.function
    def value(self, observation):
        """
        Compute value estimate(s) given the observation(s)

        Parameters:
        ----------

        observation     observation data (either single or a batch)

        Returns:
        -------
        value estimate
        """

        node_feature, adjacency, available_actions, node_mask, target_information = tuple([observation.get(key) for key in ['node_attributes', 'adjacency', 'available_actions',
                                                                                                                          'node_mask', 'target_information']])

        tiled_target_information = tf.tile(tf.expand_dims(target_information, axis=1), (1, node_feature.shape[1], 1))

        # value_node_latent_feature = tf.linalg.matmul(adjacency, self.value_network(tf.concat([node_feature, tiled_target_information], axis=-1)))
        value_node_latent_feature = tf.linalg.matmul(adjacency, self.value_network(node_feature))
        value_node_latent_feature = tf.linalg.matmul(adjacency, self.node_feature_extractor_extended(value_node_latent_feature))
        # value_global_graph_feature = tf.reduce_mean(value_node_latent_feature, axis = 1)
        value_global_graph_feature = tf.math.divide(tf.reduce_sum(tf.multiply(value_node_latent_feature, node_mask), axis = 1),
                                                    tf.reduce_sum(node_mask, axis = 1))

        for_pivot_node_latent_feature = tf.concat([value_node_latent_feature, tf.tile(tf.expand_dims(value_global_graph_feature, axis=1),
                                                                              (1, value_node_latent_feature.shape[1], 1))], axis=-1)

        # for_pivot_node_logits = tf.squeeze(self.pivot_classifier(for_pivot_node_latent_feature), axis=2)
        # for_pivot_node_logits = tf.squeeze(self.pivot_classifier(value_node_latent_feature), axis=2)
        for_pivot_node_logits = tf.squeeze(self.pivot_classifier(tf.concat([value_node_latent_feature, tiled_target_information], axis=-1)), axis=2)

        for_pivot_mask = tf.cast(tf.reduce_sum(available_actions, axis = 2) == 0, dtype=for_pivot_node_logits.dtype) * (-1e8)

        masked_for_pivot_node_logits = for_pivot_node_logits + for_pivot_mask

        pivot_node_index = tf.argmax(masked_for_pivot_node_logits, axis=1)

        pivot_node_index_logits = tf.math.reduce_max(masked_for_pivot_node_logits, axis=1)

        # result = tf.squeeze(self.value_fc(value_global_graph_feature), axis=1)
        # return result

        return pivot_node_index_logits

class PolicyWithValue_TargetEmbedded(tf.Module):
    """
    Encapsulates fields and methods for RL policy and value function estimation with shared parameters
    """

    def __init__(self, ac_space, policy_network, value_network=None, estimate_q=False):
        """
        Parameters:
        ----------
        ac_space        action space

        policy_network  keras network for policy

        value_network   keras network for value

        estimate_q      q value or v value

        """

        self.policy_network = policy_network
        # self.value_network = value_network or policy_network
        self.value_network = policy_network
        self.estimate_q = estimate_q
        self.initial_state = None

        self.init_fc = fc_relu(4, 'init_fc', 128, init_scale=np.sqrt(2))
        self.init_target_fc = fc_relu(3, 'init_fc', 128, init_scale=np.sqrt(2))

        self.node_feature_extractor_extended  = fc_relu(128 + 128, 'f_e', 128, init_scale=np.sqrt(2))
        self.node_feature_extractor_extended_2 = fc_relu(128 + 128, 'f_e', 128, init_scale=np.sqrt(2))

        self.pivot_classifier = fc(128, 'piv', 1, init_scale=np.sqrt(2))

        # Based on the action space, will select what probability distribution type
        self.pdtype = make_pdtype((None, 128), ac_space, init_scale=np.sqrt(2))

        if estimate_q:
            assert isinstance(ac_space, gym.spaces.Discrete)
            self.value_fc = fc(self.value_network.output_shape, 'q', ac_space.n, init_scale=np.sqrt(2))
        else:
            self.value_fc = fc(self.value_network.output_shape, 'vf', 1, init_scale=np.sqrt(2))

    @tf.function
    def step(self, observation):
        """
        Compute next action(s) given the observation(s)

        Parameters:
        ----------

        observation     batched observation data

        Returns:
        -------
        (action, value estimate, next state, negative log likelihood of the action under current policy parameters) tuple
        """
        node_feature, adjacency, available_actions, node_mask, target_information = tuple([observation.get(key) for key in ['node_attributes', 'adjacency', 'available_actions',
                                                                                                                          'node_mask', 'target_information']])

        node_feature = self.init_fc(node_feature)
        target_information = self.init_target_fc(target_information)

        tiled_target_information = tf.tile(tf.expand_dims(target_information, axis=1), (1, node_feature.shape[1], 1))

        node_latent_feature = tf.linalg.matmul(adjacency, self.policy_network(tf.concat([node_feature, tiled_target_information], axis=-1)))
        # node_latent_feature = tf.linalg.matmul(adjacency, self.policy_network(node_feature))

        '''Shape : (num_envs, max_node, hidden_dim)'''
        node_latent_feature = tf.linalg.matmul(adjacency, self.node_feature_extractor_extended(tf.concat([node_latent_feature, tiled_target_information], axis=-1)))
        # node_latent_feature = tf.linalg.matmul(adjacency, self.node_feature_extractor_extended(node_latent_feature))

        node_latent_feature = tf.linalg.matmul(adjacency, self.node_feature_extractor_extended_2(tf.concat([node_latent_feature, tiled_target_information], axis=-1)))
        # node_latent_feature = tf.linalg.matmul(adjacency, self.node_feature_extractor_extended_2(node_latent_feature))

        # global_graph_feature = tf.reduce_mean(node_latent_feature, axis = 1)
        global_graph_feature = tf.math.divide(tf.reduce_sum(tf.multiply(node_latent_feature, node_mask), axis = 1),
                                              tf.reduce_sum(node_mask, axis = 1))


        # for_pivot_node_latent_feature = tf.concat([node_latent_feature, tf.tile(tf.expand_dims(global_graph_feature, axis=1),
        #                                                                        (1, node_latent_feature.shape[1], 1))], axis=-1)

        # for_pivot_node_logits = tf.squeeze(self.pivot_classifier(for_pivot_node_latent_feature), axis=2)
        for_pivot_node_logits = tf.squeeze(self.pivot_classifier(node_latent_feature), axis=2)
        # for_pivot_node_logits = tf.squeeze(self.pivot_classifier(tf.concat([node_latent_feature, tiled_target_information], axis=-1)), axis=2)

        for_pivot_mask = tf.cast(tf.reduce_sum(available_actions, axis = 2) == 0, dtype=for_pivot_node_logits.dtype) * (-1e8)

        masked_for_pivot_node_logits = for_pivot_node_logits + for_pivot_mask

        pivot_node_index = tf.random.categorical(masked_for_pivot_node_logits, 1)

        pivot_neglogp = tf.nn.softmax_cross_entropy_with_logits(logits=for_pivot_node_logits, labels=tf.one_hot(pivot_node_index, for_pivot_node_logits.shape[-1]))

        # pivot_node_index_logits = tf.math.reduce_max(masked_for_pivot_node_logits, axis=1)

        node_latent_representation = []

        for i in range(node_latent_feature.shape[0]):
            # node_latent_representation.append(tf.concat([node_latent_feature[i][pivot_node_index[i]], global_graph_feature[i]], axis=-1))
            # node_latent_representation.append(node_latent_feature[i][pivot_node_index[i]])
            node_latent_representation.append(node_latent_feature[i][tf.squeeze(pivot_node_index)[i]])

        '''Shape : (num_envs, hidden_dim)'''
        node_latent_representation = tf.stack(node_latent_representation)

        per_pivot_available_actions = []

        for i in range(available_actions.shape[0]):
            # per_pivot_available_actions.append(available_actions[i][pivot_node_index[i]])
            per_pivot_available_actions.append(available_actions[i][tf.squeeze(pivot_node_index)[i]])

        per_pivot_available_actions = tf.stack(per_pivot_available_actions)

        # node_latent_representation = tf.concat([tf.gather_nd(node_latent_feature, pivot_node_index[:, None], batch_dims=1), global_graph_feature], axis = -1)
        #
        # per_pivot_available_actions = tf.gather_nd(available_actions, pivot_node_index[:, None], batch_dims=1)

        pd, pi = self.pdtype.pdfromlatent(node_latent_representation)
        # pd, pi = self.pdtype.pdfromlatent(tf.concat([node_latent_representation, target_information], axis=-1))

        negative_logits_mask = (tf.ones_like(per_pivot_available_actions) - per_pivot_available_actions) * int(-1e8)

        masked_action_logits = pi + tf.cast(negative_logits_mask, pi.dtype)

        action = tf.random.categorical(masked_action_logits, 1)

        # action = tf.constant(np.array([[10],[10],[10],[10],[10],[10],[10],[10],
        #                                [10],[10],[10],[10],[10],[10],[10],[10]]))

        # action = pd.sample()
        neglogp = pd.neglogp(action) + pivot_neglogp
        # neglogp = pd.neglogp(action)

        vf = tf.squeeze(self.value_fc(global_graph_feature), axis=1)
        # return action, vf, None, neglogp, pivot_node_index[:, None]

        # return tf.cast(action, dtype=tf.int32), vf, None, neglogp, tf.cast(pivot_node_index[:, None], dtype=tf.int32)
        return tf.cast(action, dtype=tf.int32), vf, None, neglogp, tf.cast(pivot_node_index, dtype=tf.int32)

    @tf.function
    def value(self, observation):
        """
        Compute value estimate(s) given the observation(s)

        Parameters:
        ----------

        observation     observation data (either single or a batch)

        Returns:
        -------
        value estimate
        """

        node_feature, adjacency, available_actions, node_mask, target_information = tuple([observation.get(key) for key in ['node_attributes', 'adjacency', 'available_actions',
                                                                                                                          'node_mask', 'target_information']])

        node_feature = self.init_fc(node_feature)
        target_information = self.init_target_fc(target_information)

        tiled_target_information = tf.tile(tf.expand_dims(target_information, axis=1), (1, node_feature.shape[1], 1))

        value_node_latent_feature = tf.linalg.matmul(adjacency, self.policy_network(tf.concat([node_feature, tiled_target_information], axis=-1)))
        # value_node_latent_feature = tf.linalg.matmul(adjacency, self.value_network(node_feature))

        value_node_latent_feature = tf.linalg.matmul(adjacency, self.node_feature_extractor_extended(tf.concat([value_node_latent_feature, tiled_target_information], axis=-1)))
        # value_node_latent_feature = tf.linalg.matmul(adjacency, self.node_feature_extractor_extended(value_node_latent_feature))

        value_node_latent_feature = tf.linalg.matmul(adjacency, self.node_feature_extractor_extended_2(tf.concat([value_node_latent_feature, tiled_target_information], axis=-1)))
        # value_node_latent_feature = tf.linalg.matmul(adjacency, self.node_feature_extractor_extended_2(value_node_latent_feature))

        # value_global_graph_feature = tf.reduce_mean(value_node_latent_feature, axis = 1)
        value_global_graph_feature = tf.math.divide(tf.reduce_sum(tf.multiply(value_node_latent_feature, node_mask), axis = 1),
                                                    tf.reduce_sum(node_mask, axis = 1))

        result = tf.squeeze(self.value_fc(value_global_graph_feature), axis=1)

        # return pivot_node_index_logits
        return result
