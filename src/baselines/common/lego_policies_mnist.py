import tensorflow as tf
from baselines.a2c.utils import fc, fc_relu, conv, ortho_init, fc_sigmoid
from baselines.common.distributions import make_pdtype

import gym
import numpy as np

class Init_FC(tf.keras.Model):
    def __init__(self, hidden_dim=64, init_dim = 4, target_dim = 128, layer_type=0, layer_num=0):
        super(Init_FC, self).__init__()

        self.fc0 = fc_relu(init_dim + target_dim, 'init_{}_fc0_{}'.format(layer_type, layer_num), hidden_dim, init_scale=np.sqrt(2))

        self.fc1 = fc_relu(hidden_dim + target_dim, 'init_{}_fc1_{}'.format(layer_type, layer_num), hidden_dim, init_scale=np.sqrt(2))

        self.fc2 = fc_relu(hidden_dim + target_dim, 'init_{}_fc2_{}'.format(layer_type, layer_num), hidden_dim, init_scale=np.sqrt(2))

    def call(self, init_feature, target_feature):
        inputs = init_feature

        x = self.fc0(tf.concat([init_feature, target_feature], axis=-1))
        x = self.fc1(tf.concat([x, target_feature], axis=-1))
        x = self.fc2(tf.concat([x, target_feature], axis=-1))

        return x

class Init_FC_No_Target(tf.keras.Model):
    def __init__(self, hidden_dim=64, init_dim = 4, target_dim = 128, layer_type=0, layer_num=0):
        super(Init_FC_No_Target, self).__init__()

        self.fc0 = fc_relu(init_dim, 'init_{}_fc0_{}'.format(layer_type, layer_num), hidden_dim, init_scale=np.sqrt(2))

        self.fc1 = fc_relu(hidden_dim, 'init_{}_fc1_{}'.format(layer_type, layer_num), hidden_dim, init_scale=np.sqrt(2))

        self.fc2 = fc_relu(hidden_dim, 'init_{}_fc2_{}'.format(layer_type, layer_num), hidden_dim, init_scale=np.sqrt(2))

    def call(self, init_feature):
        x = self.fc0(init_feature)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

class Lego_Message_Passing_Mnist(tf.keras.Model):
    def __init__(self, policy_network=None, hidden_dim=64, node_dim = 4, target_info_dim=4, edge_dim=4, layer_num=0):
        super(Lego_Message_Passing_Mnist, self).__init__()

        # self.feature_fc = fc_relu(hidden_dim + node_dim + target_info_dim, 'feat_fc_{}'.format(int(layer_num)), hidden_dim, init_scale=np.sqrt(2))
        self.feature_fc = fc_relu(hidden_dim + node_dim, 'feat_fc_{}'.format(int(layer_num)), hidden_dim, init_scale=np.sqrt(2))

        # self.edge_fc = fc_relu(node_dim + node_dim + edge_dim + target_info_dim, 'edge_fc_{}'.format(int(layer_num)), hidden_dim, init_scale=np.sqrt(2))
        self.edge_fc = fc_relu(node_dim + node_dim + edge_dim, 'edge_fc_{}'.format(int(layer_num)), hidden_dim, init_scale=np.sqrt(2))

    def call(self, node_feature, adjacency, target_information, edge_feature, node_mask, training=True):
        tiled_target_information = tf.tile(tf.expand_dims(target_information, axis=1), (1, node_feature.shape[1], 1))
        two_tiled_target_information = tf.tile(tf.expand_dims(tiled_target_information, axis=1), (1, node_feature.shape[1], 1, 1))

        tiled_node_feature = tf.tile(tf.expand_dims(node_feature, axis=1), (1, node_feature.shape[1], 1, 1))
        node_node_feature = tf.concat((tf.transpose(tiled_node_feature, (0, 2, 1, 3)), tiled_node_feature), axis=-1)

        # edge_feature = self.edge_fc(tf.concat((node_node_feature, edge_feature, two_tiled_target_information), axis=-1))
        edge_feature = self.edge_fc(tf.concat((node_node_feature, edge_feature), axis=-1))

        # message = tf.reduce_sum(tf.math.multiply(self.message_fc(tf.concat((node_node_feature, edge_feature), axis=-1)), adjacency[..., None]), axis=2)
        message = tf.reduce_sum(tf.math.multiply(edge_feature, adjacency[..., None]), axis=2)

        '''Shape : (num_envs, max_node, hidden_dim)'''
        # ending_feature = self.feature_fc(tf.concat([message, node_feature, tiled_target_information], axis=-1))
        ending_feature = self.feature_fc(tf.concat([message, node_feature], axis=-1))

        # target_feature = self.target_fc(tf.concat([tf.reduce_mean(ending_feature, axis=1), tf.reduce_mean(edge_feature, axis=(1,2)), target_information], axis=-1))
        target_feature = target_information

        return ending_feature, edge_feature, target_feature

class Lego_Message_Passing_Mnist_With_Msg_Fc(tf.keras.Model):
    def __init__(self, policy_network=None, hidden_dim=64, node_dim = 4, target_info_dim=4, edge_dim=4, layer_num=0):
        super(Lego_Message_Passing_Mnist_with_msg_fc, self).__init__()

        self.message_fc = policy_network or fc_relu(hidden_dim + hidden_dim + edge_dim, 'msg_fc_{}'.format(int(layer_num)),
                                                    hidden_dim, init_scale=np.sqrt(2))

        self.feature_fc = fc_relu(hidden_dim + node_dim, 'feat_fc_{}'.format(int(layer_num)), hidden_dim, init_scale=np.sqrt(2))

        self.edge_fc = fc_relu(node_dim + node_dim + edge_dim, 'edge_fc_{}'.format(int(layer_num)), hidden_dim, init_scale=np.sqrt(2))

    def call(self, node_feature, adjacency, target_information, edge_feature, node_mask, training=True):
        tiled_node_feature = tf.tile(tf.expand_dims(node_feature, axis=1), (1, node_feature.shape[1], 1, 1))
        node_node_feature = tf.concat((tf.transpose(tiled_node_feature, (0, 2, 1, 3)), tiled_node_feature), axis=-1)

        edge_feature = self.edge_fc(tf.concat((node_node_feature, edge_feature), axis=-1))

        message = tf.reduce_sum(tf.math.multiply(self.message_fc(tf.concat((node_node_feature, edge_feature), axis=-1)), adjacency[..., None]), axis=2)

        ending_feature = self.feature_fc(tf.concat([message, node_feature], axis=-1))

        target_feature = target_information

        return ending_feature, edge_feature, target_feature

class Lego_Message_Passing_Mnist_No_Edge(tf.keras.Model):
    def __init__(self, policy_network=None, hidden_dim=64, node_dim = 4, target_info_dim=4, edge_dim=4, layer_num=0):
        super(Lego_Message_Passing_Mnist_No_Edge, self).__init__()

        self.feature_fc = policy_network or fc_relu(node_dim, 'feat_fc_{}'.format(int(layer_num)), hidden_dim, init_scale=np.sqrt(2))

    def call(self, node_feature, adjacency, target_information, edge_feature, node_mask, training=True):
        ending_feature = tf.linalg.matmul(adjacency, self.feature_fc(node_feature))

        edge_feature = edge_feature
        target_feature = target_information

        return ending_feature, edge_feature, target_feature

class PolicyWithValue_Lego_Mnist(tf.Module):
    """
    Encapsulates fields and methods for RL policy and value function estimation with shared parameters
    """

    def __init__(self, ac_space, policy_network, target_network, hidden_dim, target_hidden_dim, value_network=None, estimate_q=False):
        """
        Parameters:
        ----------
        ac_space        action space

        policy_network  keras network for policy

        value_network   keras network for value

        estimate_q      q value or v value

        """

        self.estimate_q = estimate_q
        self.initial_state = None

        self.target_network = target_network

        # hidden_dim = 64

        target_init_dim = 3
        # target_hidden_dim = 64

        edge_init_dim = 4

        # self.init_node_fc = Init_FC_No_Target(hidden_dim=128, target_dim=target_hidden_dim, layer_type=0)
        self.init_node_fc = Init_FC(hidden_dim=hidden_dim, target_dim=target_hidden_dim, layer_type=0)

        # self.init_edge_fc = Init_FC_No_Target(hidden_dim=128, target_dim=target_hidden_dim, layer_type=1)
        self.init_edge_fc = Init_FC(hidden_dim=hidden_dim, target_dim=target_hidden_dim, layer_type=1)

        self.gcn_init = Lego_Message_Passing_Mnist(policy_network, node_dim=hidden_dim, hidden_dim=hidden_dim, target_info_dim=target_hidden_dim, edge_dim=hidden_dim, layer_num=1)

        self.gcn_block = []

        for i in range(2):
            temp_gcn = Lego_Message_Passing_Mnist(node_dim=hidden_dim, hidden_dim=hidden_dim, target_info_dim=target_hidden_dim, edge_dim=hidden_dim, layer_num=i+2)
            # temp_gcn = Lego_Message_Passing_Mnist_No_Edge(node_dim=hidden_dim, hidden_dim=hidden_dim, target_info_dim=target_hidden_dim, edge_dim=hidden_dim, layer_num=i+2)

            self.gcn_block.append(temp_gcn)

        self.pivot_classifier = fc(hidden_dim + target_hidden_dim, 'piv', 1, init_scale=np.sqrt(2))

        # Based on the action space, will select what probability distribution type
        self.pdtype = make_pdtype((None, hidden_dim + target_hidden_dim), ac_space, init_scale=np.sqrt(2))

        if estimate_q:
            assert isinstance(ac_space, gym.spaces.Discrete)
            self.value_fc = fc(hidden_dim, 'q', ac_space.n, init_scale=np.sqrt(2))
        else:
            self.value_fc = fc(hidden_dim + target_hidden_dim, 'vf', 1, init_scale=np.sqrt(2))

    @tf.function
    def step(self, observation, training=True):
        """
        Compute next action(s) given the observation(s)

        Parameters:
        ----------

        observation     batched observation data

        Returns:
        -------
        (action, value estimate, next state, negative log likelihood of the action under current policy parameters) tuple
        """
        node_feature, adjacency, available_actions, node_mask, target_information, edge_feature = \
            tuple([observation.get(key) for key in ['node_attributes', 'adjacency', 'available_actions', 'node_mask', 'target_information', 'edge_attributes']])

        target_information = self.target_network(target_information)

        tiled_target_information = tf.tile(tf.expand_dims(target_information, axis=1), (1, node_feature.shape[1], 1))

        node_feature = self.init_node_fc(node_feature, tiled_target_information)
        # node_feature = self.init_node_fc(node_feature)

        two_tiled_target_information = tf.tile(tf.expand_dims(tiled_target_information, axis=1), (1, edge_feature.shape[1], 1, 1))
        edge_feature = self.init_edge_fc(edge_feature, two_tiled_target_information)
        # edge_feature = self.init_edge_fc(edge_feature)

        node_feature, edge_feature, target_information = self.gcn_init(node_feature, adjacency, target_information, edge_feature, node_mask)

        for i in range(len(self.gcn_block)):
            node_feature, edge_feature, target_information = self.gcn_block[i](node_feature, adjacency, target_information, edge_feature, node_mask)

        tiled_target_information = tf.tile(tf.expand_dims(target_information, axis=1), (1, node_feature.shape[1], 1))

        # global_graph_feature = tf.math.divide(tf.reduce_sum(tf.multiply(node_feature, node_mask), axis = 1),
        #                                       tf.reduce_sum(node_mask, axis = 1))
        global_graph_feature = tf.reduce_max(tf.multiply(node_feature, node_mask) +
                                             tf.tile(node_feature[:, :1, :], (1, node_feature.shape[1], 1)) * (tf.ones_like(node_mask) - node_mask), axis = 1)

        for_pivot_node_logits = tf.squeeze(self.pivot_classifier(tf.concat([node_feature, tiled_target_information], axis=-1)), axis=2)

        for_pivot_mask = tf.cast(tf.reduce_sum(available_actions, axis = 2) == 0, dtype=for_pivot_node_logits.dtype) * (-1e8)

        masked_for_pivot_node_logits = for_pivot_node_logits + for_pivot_mask

        pivot_node_index = tf.random.categorical(masked_for_pivot_node_logits, 1)

        pivot_cce = tf.keras.losses.CategoricalCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)

        pivot_probs = tf.nn.softmax(masked_for_pivot_node_logits, axis=-1)

        pivot_neglogp = pivot_cce(y_true=tf.one_hot(pivot_node_index, for_pivot_node_logits.shape[-1]), y_pred=pivot_probs)

        node_latent_representation = []

        for i in range(node_feature.shape[0]):
            node_latent_representation.append(node_feature[i][tf.squeeze(pivot_node_index)[i]])

        '''Shape : (num_envs, hidden_dim)'''
        node_latent_representation = tf.stack(node_latent_representation)

        per_pivot_available_actions = []

        for i in range(available_actions.shape[0]):
            per_pivot_available_actions.append(available_actions[i][tf.squeeze(pivot_node_index)[i]])

        per_pivot_available_actions = tf.stack(per_pivot_available_actions)

        pd, pi = self.pdtype.pdfromlatent(tf.concat([node_latent_representation, target_information], axis=-1))

        negative_logits_mask = (tf.ones_like(per_pivot_available_actions) - per_pivot_available_actions) * int(-1e8)

        masked_action_logits = pi + tf.cast(negative_logits_mask, pi.dtype)

        action = tf.random.categorical(masked_action_logits, 1)

        cce = tf.keras.losses.CategoricalCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)

        probs = tf.nn.softmax(masked_action_logits, axis=-1)

        neglogp = cce(y_true=tf.one_hot(action, pi.shape[-1]), y_pred=probs)

        vf = tf.squeeze(self.value_fc(tf.concat([global_graph_feature, target_information], axis=-1)), axis=1)

        return tf.cast(action, dtype=tf.int32), vf, None, neglogp, pivot_neglogp, tf.cast(pivot_node_index, dtype=tf.int32)

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

        node_feature, adjacency, available_actions, node_mask, target_information, edge_feature = \
            tuple([observation.get(key) for key in ['node_attributes', 'adjacency', 'available_actions', 'node_mask', 'target_information', 'edge_attributes']])

        target_information = self.target_network(target_information)

        tiled_target_information = tf.tile(tf.expand_dims(target_information, axis=1), (1, node_feature.shape[1], 1))

        node_feature = self.init_node_fc(node_feature, tiled_target_information)
        # node_feature = self.init_node_fc(node_feature)

        two_tiled_target_information = tf.tile(tf.expand_dims(tiled_target_information, axis=1), (1, edge_feature.shape[1], 1, 1))
        edge_feature = self.init_edge_fc(edge_feature, two_tiled_target_information)
        # edge_feature = self.init_edge_fc(edge_feature)

        node_feature, edge_feature, target_information = self.gcn_init(node_feature, adjacency, target_information, edge_feature, node_mask)

        for i in range(len(self.gcn_block)):
            node_feature, edge_feature, target_information = self.gcn_block[i](node_feature, adjacency, target_information, edge_feature, node_mask)

        tiled_target_information = tf.tile(tf.expand_dims(target_information, axis=1), (1, node_feature.shape[1], 1))

        # global_graph_feature = tf.math.divide(tf.reduce_sum(tf.multiply(node_feature, node_mask), axis = 1),
        #                                       tf.reduce_sum(node_mask, axis = 1))
        global_graph_feature = tf.reduce_max(tf.multiply(node_feature, node_mask) +
                                             tf.tile(node_feature[:, :1, :], (1, node_feature.shape[1], 1)) * (tf.ones_like(node_mask) - node_mask), axis = 1)

        result = tf.squeeze(self.value_fc(tf.concat([global_graph_feature, target_information], axis=-1)), axis=1)

        return result

class Lego_Message_Passing_Mnist_MultiGNN(tf.keras.Model):
    def __init__(self, policy_network=None, hidden_dim=64, node_dim = 4, target_info_dim=4, edge_dim=4, layer_num=0):
        super(Lego_Message_Passing_Mnist_MultiGNN, self).__init__()

        # self.feature_fc = fc_relu(hidden_dim + node_dim + target_info_dim, 'feat_fc_{}'.format(int(layer_num)), hidden_dim, init_scale=np.sqrt(2))
        self.feature_fc = fc_relu(hidden_dim + node_dim, 'feat_fc_{}'.format(int(layer_num)), hidden_dim, init_scale=np.sqrt(2))

        # self.edge_fc = fc_relu(node_dim + node_dim + edge_dim + target_info_dim, 'edge_fc_{}'.format(int(layer_num)), hidden_dim, init_scale=np.sqrt(2))
        self.edge_fc = fc_relu(node_dim + node_dim + edge_dim, 'edge_fc_{}'.format(int(layer_num)), hidden_dim, init_scale=np.sqrt(2))

    def call(self, node_feature, adjacency, target_information, edge_feature, node_mask, training=True):
        tiled_target_information = tf.tile(tf.expand_dims(target_information, axis=1), (1, node_feature.shape[1], 1))
        two_tiled_target_information = tf.tile(tf.expand_dims(tiled_target_information, axis=1), (1, node_feature.shape[1], 1, 1))

        tiled_node_feature = tf.tile(tf.expand_dims(node_feature, axis=1), (1, node_feature.shape[1], 1, 1))
        node_node_feature = tf.concat((tf.transpose(tiled_node_feature, (0, 2, 1, 3)), tiled_node_feature), axis=-1)

        edge_feature = self.edge_fc(tf.concat((node_node_feature, edge_feature), axis=-1))

        # message = tf.reduce_sum(tf.math.multiply(edge_feature, adjacency[..., None]), axis=2)
        message = tf.math.divide(tf.reduce_sum(tf.math.multiply(edge_feature, adjacency[..., None]), axis=2),
                                 tf.reduce_sum(adjacency, axis=-1)[..., None] + (tf.ones_like(node_mask) - node_mask))

        '''Shape : (num_envs, max_node, hidden_dim)'''
        # ending_feature = self.feature_fc(tf.concat([message, node_feature, tiled_target_information], axis=-1))
        ending_feature = self.feature_fc(tf.concat([message, node_feature], axis=-1))

        return ending_feature, edge_feature

class PolicyWithValue_Lego_Mnist_MultiGNN(tf.Module):
    """
    Encapsulates fields and methods for RL policy and value function estimation with shared parameters
    """

    def __init__(self, ac_space, policy_network, target_network, hidden_dim, target_hidden_dim, value_network=None, estimate_q=False):
        """
        Parameters:
        ----------
        ac_space        action space

        policy_network  keras network for policy

        value_network   keras network for value

        estimate_q      q value or v value

        """

        self.estimate_q = estimate_q
        self.initial_state = None

        self.target_network = target_network

        # hidden_dim = hidden_dim

        target_init_dim = 3
        # target_hidden_dim = target_hidden_dim

        edge_init_dim = 4

        self.init_node_fc = Init_FC(hidden_dim=hidden_dim, target_dim=target_hidden_dim, layer_type=0)

        self.init_edge_fc = Init_FC(hidden_dim=hidden_dim, target_dim=target_hidden_dim, layer_type=1)

        self.gcn_init = Lego_Message_Passing_Mnist_MultiGNN(policy_network, node_dim=hidden_dim, hidden_dim=hidden_dim, target_info_dim=target_hidden_dim, edge_dim=hidden_dim, layer_num=1)

        self.gcn_block = []

        for i in range(2):
            temp_gcn = Lego_Message_Passing_Mnist_MultiGNN(node_dim=hidden_dim, hidden_dim=hidden_dim, target_info_dim=target_hidden_dim, edge_dim=hidden_dim, layer_num=i+2)

            self.gcn_block.append(temp_gcn)

        self.pivot_gcn_init = Lego_Message_Passing_Mnist_MultiGNN(policy_network=None, node_dim=hidden_dim, hidden_dim=hidden_dim, target_info_dim=target_hidden_dim, edge_dim=hidden_dim, layer_num=1)

        self.pivot_gcn_block = []

        for i in range(2):
            temp_gcn = Lego_Message_Passing_Mnist_MultiGNN(node_dim=hidden_dim, hidden_dim=hidden_dim, target_info_dim=target_hidden_dim, edge_dim=hidden_dim, layer_num=i+2)

            self.pivot_gcn_block.append(temp_gcn)

        self.pivot_classifier = fc(hidden_dim + target_hidden_dim, 'piv', 1, init_scale=np.sqrt(2))

        # self.mask_predictor = fc_sigmoid(hidden_dim, 'mask', ac_space.n, init_scale=np.sqrt(2))

        # Based on the action space, will select what probability distribution type
        self.pdtype = make_pdtype((None, hidden_dim + target_hidden_dim), ac_space, init_scale=np.sqrt(2))

        if estimate_q:
            assert isinstance(ac_space, gym.spaces.Discrete)
            self.value_fc = fc(hidden_dim, 'q', ac_space.n, init_scale=np.sqrt(2))
        else:
            self.value_fc = fc(hidden_dim + hidden_dim + target_hidden_dim, 'vf', 1, init_scale=np.sqrt(2))

    @tf.function
    def step(self, observation, training=True):
        """
        Compute next action(s) given the observation(s)

        Parameters:
        ----------

        observation     batched observation data

        Returns:
        -------
        (action, value estimate, next state, negative log likelihood of the action under current policy parameters) tuple
        """
        node_feature, adjacency, available_actions, node_mask, target_information, edge_feature = \
            tuple([observation.get(key) for key in ['node_attributes', 'adjacency', 'available_actions', 'node_mask', 'target_information', 'edge_attributes']])

        target_information = self.target_network(target_information)
        tiled_target_information = tf.tile(tf.expand_dims(target_information, axis=1), (1, node_feature.shape[1], 1))

        node_feature = self.init_node_fc(node_feature, tiled_target_information)

        two_tiled_target_information = tf.tile(tf.expand_dims(tiled_target_information, axis=1), (1, edge_feature.shape[1], 1, 1))
        edge_feature = self.init_edge_fc(edge_feature, two_tiled_target_information)

        action_node_feature, action_edge_feature = self.gcn_init(node_feature, adjacency, target_information, edge_feature, node_mask)

        for i in range(len(self.gcn_block)):
            action_node_feature, action_edge_feature = self.gcn_block[i](action_node_feature, adjacency, target_information, action_edge_feature, node_mask)

        pivot_node_feature, pivot_edge_feature = self.pivot_gcn_init(node_feature, adjacency, target_information, edge_feature, node_mask)

        for i in range(len(self.gcn_block)):
            pivot_node_feature, pivot_edge_feature = self.pivot_gcn_block[i](pivot_node_feature, adjacency, target_information, pivot_edge_feature, node_mask)

        action_graph_feature = tf.math.divide(tf.reduce_sum(tf.multiply(action_node_feature, node_mask), axis = 1),
                                              tf.reduce_sum(node_mask, axis = 1))
        # action_graph_feature = tf.reduce_max(tf.multiply(action_node_feature, node_mask) +
        #                                      tf.tile(action_node_feature[:, :1, :], (1, action_node_feature.shape[1], 1)) * (tf.ones_like(node_mask) - node_mask), axis = 1)

        pivot_graph_feature = tf.math.divide(tf.reduce_sum(tf.multiply(pivot_node_feature, node_mask), axis = 1),
                                             tf.reduce_sum(node_mask, axis = 1))
        # pivot_graph_feature = tf.reduce_max(tf.multiply(pivot_node_feature, node_mask) +
        #                                      tf.tile(pivot_node_feature[:, :1, :], (1, pivot_node_feature.shape[1], 1)) * (tf.ones_like(node_mask) - node_mask), axis = 1)

        for_pivot_node_logits = tf.squeeze(self.pivot_classifier(tf.concat([pivot_node_feature, tiled_target_information], axis=-1)), axis=2)

        for_pivot_mask = tf.cast(tf.reduce_sum(available_actions, axis = 2) == 0, dtype=for_pivot_node_logits.dtype) * float(-1e8)

        masked_for_pivot_node_logits = for_pivot_node_logits + tf.cast(for_pivot_mask, dtype=for_pivot_node_logits.dtype)

        pivot_node_index = tf.random.categorical(masked_for_pivot_node_logits, 1)

        pivot_cce = tf.keras.losses.CategoricalCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)

        pivot_probs = tf.nn.softmax(masked_for_pivot_node_logits, axis=-1)

        pivot_neglogp = pivot_cce(y_true=tf.one_hot(pivot_node_index, for_pivot_node_logits.shape[-1]), y_pred=tf.expand_dims(pivot_probs, axis=1))

        node_latent_representation = []

        for i in range(node_feature.shape[0]):
            node_latent_representation.append(action_node_feature[i][tf.squeeze(pivot_node_index)[i]])

        '''Shape : (num_envs, hidden_dim)'''
        node_latent_representation = tf.stack(node_latent_representation)

        per_pivot_available_actions = []

        for i in range(available_actions.shape[0]):
            per_pivot_available_actions.append(available_actions[i][tf.squeeze(pivot_node_index)[i]])

        per_pivot_available_actions = tf.stack(per_pivot_available_actions)

        pd, pi = self.pdtype.pdfromlatent(tf.concat([node_latent_representation, target_information], axis=-1))

        negative_logits_mask = (tf.ones_like(per_pivot_available_actions) - per_pivot_available_actions) * float(-1e8)

        masked_action_logits = pi + tf.cast(negative_logits_mask, pi.dtype)

        action = tf.random.categorical(masked_action_logits, 1)

        cce = tf.keras.losses.CategoricalCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)

        probs = tf.nn.softmax(masked_action_logits, axis=-1)

        neglogp = cce(y_true=tf.one_hot(action, pi.shape[-1]), y_pred=tf.expand_dims(probs, axis=1))

        vf = tf.squeeze(self.value_fc(tf.concat([action_graph_feature, pivot_graph_feature, target_information], axis=-1)), axis=1)

        return tf.cast(action, dtype=tf.int32), vf, None, neglogp, pivot_neglogp, tf.cast(pivot_node_index, dtype=tf.int32)

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

        node_feature, adjacency, available_actions, node_mask, target_information, edge_feature = \
            tuple([observation.get(key) for key in ['node_attributes', 'adjacency', 'available_actions', 'node_mask', 'target_information', 'edge_attributes']])

        target_information = self.target_network(target_information)
        tiled_target_information = tf.tile(tf.expand_dims(target_information, axis=1), (1, node_feature.shape[1], 1))

        node_feature = self.init_node_fc(node_feature, tiled_target_information)

        two_tiled_target_information = tf.tile(tf.expand_dims(tiled_target_information, axis=1), (1, edge_feature.shape[1], 1, 1))
        edge_feature = self.init_edge_fc(edge_feature, two_tiled_target_information)

        action_node_feature, action_edge_feature = self.gcn_init(node_feature, adjacency, target_information, edge_feature, node_mask)

        for i in range(len(self.gcn_block)):
            action_node_feature, action_edge_feature = self.gcn_block[i](action_node_feature, adjacency, target_information, action_edge_feature, node_mask)

        pivot_node_feature, pivot_edge_feature = self.pivot_gcn_init(node_feature, adjacency, target_information, edge_feature, node_mask)

        for i in range(len(self.gcn_block)):
            pivot_node_feature, pivot_edge_feature = self.pivot_gcn_block[i](pivot_node_feature, adjacency, target_information, pivot_edge_feature, node_mask)

        action_graph_feature = tf.math.divide(tf.reduce_sum(tf.multiply(action_node_feature, node_mask), axis = 1),
                                              tf.reduce_sum(node_mask, axis = 1))
        # action_graph_feature = tf.reduce_max(tf.multiply(action_node_feature, node_mask) +
        #                                      tf.tile(action_node_feature[:, :1, :], (1, action_node_feature.shape[1], 1)) * (tf.ones_like(node_mask) - node_mask), axis = 1)

        pivot_graph_feature = tf.math.divide(tf.reduce_sum(tf.multiply(pivot_node_feature, node_mask), axis = 1),
                                             tf.reduce_sum(node_mask, axis = 1))
        # pivot_graph_feature = tf.reduce_max(tf.multiply(pivot_node_feature, node_mask) +
        #                                      tf.tile(pivot_node_feature[:, :1, :], (1, pivot_node_feature.shape[1], 1)) * (tf.ones_like(node_mask) - node_mask), axis = 1)

        result = tf.squeeze(self.value_fc(tf.concat([action_graph_feature, pivot_graph_feature, target_information], axis=-1)), axis=1)

        return result

class Lego_FC(tf.keras.Model):
    def __init__(self, policy_network=None, hidden_dim=64, node_dim = 4, target_info_dim=4, edge_dim=4, layer_num=0):
        super(Lego_FC, self).__init__()

        self.feature_fc = fc_relu(hidden_dim + hidden_dim, 'feat_fc_{}'.format(int(layer_num)), hidden_dim, init_scale=np.sqrt(2))

    def call(self, node_feature, adjacency, target_information, edge_feature, node_mask, training=True):
        tiled_target_information = tf.tile(tf.expand_dims(target_information, axis=1), (1, node_feature.shape[1], 1))

        ending_feature = self.feature_fc(tf.concat([node_feature, tiled_target_information], axis=-1))

        return ending_feature

class PolicyWithValue_Lego_Mnist_MultiFC(tf.Module):
    """
    Encapsulates fields and methods for RL policy and value function estimation with shared parameters
    """

    def __init__(self, ac_space, policy_network, target_network, hidden_dim, target_hidden_dim, value_network=None, estimate_q=False):
        """
        Parameters:
        ----------
        ac_space        action space

        policy_network  keras network for policy

        value_network   keras network for value

        estimate_q      q value or v value

        """

        self.estimate_q = estimate_q
        self.initial_state = None

        self.target_network = target_network

        # hidden_dim = hidden_dim

        target_init_dim = 3
        # target_hidden_dim = target_hidden_dim

        edge_init_dim = 4

        self.init_node_fc = Init_FC(hidden_dim=hidden_dim, target_dim=target_hidden_dim, layer_type=0)

        self.gcn_init = Lego_FC(policy_network, node_dim=hidden_dim, hidden_dim=hidden_dim, target_info_dim=target_hidden_dim, edge_dim=hidden_dim, layer_num=1)

        self.gcn_block = []

        for i in range(2):
            temp_gcn = Lego_FC(node_dim=hidden_dim, hidden_dim=hidden_dim, target_info_dim=target_hidden_dim, edge_dim=hidden_dim, layer_num=i+2)

            self.gcn_block.append(temp_gcn)

        self.pivot_gcn_init = Lego_FC(policy_network=None, node_dim=hidden_dim, hidden_dim=hidden_dim, target_info_dim=target_hidden_dim, edge_dim=hidden_dim, layer_num=1)

        self.pivot_gcn_block = []

        for i in range(2):
            temp_gcn = Lego_FC(node_dim=hidden_dim, hidden_dim=hidden_dim, target_info_dim=target_hidden_dim, edge_dim=hidden_dim, layer_num=i+2)

            self.pivot_gcn_block.append(temp_gcn)

        self.pivot_classifier = fc(hidden_dim + target_hidden_dim, 'piv', 1, init_scale=np.sqrt(2))

        # Based on the action space, will select what probability distribution type
        self.pdtype = make_pdtype((None, hidden_dim + target_hidden_dim), ac_space, init_scale=np.sqrt(2))

        if estimate_q:
            assert isinstance(ac_space, gym.spaces.Discrete)
            self.value_fc = fc(hidden_dim, 'q', ac_space.n, init_scale=np.sqrt(2))
        else:
            self.value_fc = fc(hidden_dim + hidden_dim + target_hidden_dim, 'vf', 1, init_scale=np.sqrt(2))

    @tf.function
    def step(self, observation, training=True):
        """
        Compute next action(s) given the observation(s)

        Parameters:
        ----------

        observation     batched observation data

        Returns:
        -------
        (action, value estimate, next state, negative log likelihood of the action under current policy parameters) tuple
        """
        node_feature, adjacency, available_actions, node_mask, target_information, edge_feature = \
            tuple([observation.get(key) for key in ['node_attributes', 'adjacency', 'available_actions', 'node_mask', 'target_information', 'edge_attributes']])

        target_information = self.target_network(target_information)
        tiled_target_information = tf.tile(tf.expand_dims(target_information, axis=1), (1, node_feature.shape[1], 1))

        node_feature = self.init_node_fc(node_feature, tiled_target_information)

        action_node_feature = self.gcn_init(node_feature, adjacency, target_information, edge_feature, node_mask)

        for i in range(len(self.gcn_block)):
            action_node_feature = self.gcn_block[i](action_node_feature, adjacency, target_information, edge_feature, node_mask)

        pivot_node_feature = self.pivot_gcn_init(node_feature, adjacency, target_information, edge_feature, node_mask)

        for i in range(len(self.gcn_block)):
            pivot_node_feature = self.pivot_gcn_block[i](pivot_node_feature, adjacency, target_information, edge_feature, node_mask)

        action_graph_feature = tf.math.divide(tf.reduce_sum(tf.multiply(action_node_feature, node_mask), axis = 1),
                                              tf.reduce_sum(node_mask, axis = 1))

        pivot_graph_feature = tf.math.divide(tf.reduce_sum(tf.multiply(pivot_node_feature, node_mask), axis = 1),
                                             tf.reduce_sum(node_mask, axis = 1))

        for_pivot_node_logits = tf.squeeze(self.pivot_classifier(tf.concat([pivot_node_feature, tiled_target_information], axis=-1)), axis=2)

        for_pivot_mask = tf.cast(tf.reduce_sum(available_actions, axis = 2) == 0, dtype=for_pivot_node_logits.dtype) * (-1e8)

        masked_for_pivot_node_logits = for_pivot_node_logits + for_pivot_mask

        pivot_node_index = tf.random.categorical(masked_for_pivot_node_logits, 1)

        pivot_neglogp = tf.nn.softmax_cross_entropy_with_logits(logits=for_pivot_node_logits, labels=tf.one_hot(pivot_node_index, for_pivot_node_logits.shape[-1]))

        node_latent_representation = []

        for i in range(node_feature.shape[0]):
            node_latent_representation.append(action_node_feature[i][tf.squeeze(pivot_node_index)[i]])

        '''Shape : (num_envs, hidden_dim)'''
        node_latent_representation = tf.stack(node_latent_representation)

        per_pivot_available_actions = []

        for i in range(available_actions.shape[0]):
            per_pivot_available_actions.append(available_actions[i][tf.squeeze(pivot_node_index)[i]])

        per_pivot_available_actions = tf.stack(per_pivot_available_actions)

        # pd, pi = self.pdtype.pdfromlatent(self.pdtype_pre_fc(tf.concat([node_latent_representation, target_information], axis=-1)))
        pd, pi = self.pdtype.pdfromlatent(tf.concat([node_latent_representation, target_information], axis=-1))

        negative_logits_mask = (tf.ones_like(per_pivot_available_actions) - per_pivot_available_actions) * int(-1e8)

        masked_action_logits = pi + tf.cast(negative_logits_mask, pi.dtype)

        action = tf.random.categorical(masked_action_logits, 1)

        neglogp = pd.neglogp(action)

        vf = tf.squeeze(self.value_fc(tf.concat([action_graph_feature, pivot_graph_feature, target_information], axis=-1)), axis=1)

        return tf.cast(action, dtype=tf.int32), vf, None, neglogp, pivot_neglogp, tf.cast(pivot_node_index, dtype=tf.int32)

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

        node_feature, adjacency, available_actions, node_mask, target_information, edge_feature = \
            tuple([observation.get(key) for key in ['node_attributes', 'adjacency', 'available_actions', 'node_mask', 'target_information', 'edge_attributes']])

        target_information = self.target_network(target_information)
        tiled_target_information = tf.tile(tf.expand_dims(target_information, axis=1), (1, node_feature.shape[1], 1))

        node_feature = self.init_node_fc(node_feature, tiled_target_information)

        action_node_feature = self.gcn_init(node_feature, adjacency, target_information, edge_feature, node_mask)

        for i in range(len(self.gcn_block)):
            action_node_feature = self.gcn_block[i](action_node_feature, adjacency, target_information, edge_feature, node_mask)

        pivot_node_feature = self.pivot_gcn_init(node_feature, adjacency, target_information, edge_feature, node_mask)

        for i in range(len(self.gcn_block)):
            pivot_node_feature = self.pivot_gcn_block[i](pivot_node_feature, adjacency, target_information, edge_feature, node_mask)

        action_graph_feature = tf.math.divide(tf.reduce_sum(tf.multiply(action_node_feature, node_mask), axis = 1),
                                              tf.reduce_sum(node_mask, axis = 1))

        pivot_graph_feature = tf.math.divide(tf.reduce_sum(tf.multiply(pivot_node_feature, node_mask), axis = 1),
                                             tf.reduce_sum(node_mask, axis = 1))

        result = tf.squeeze(self.value_fc(tf.concat([action_graph_feature, pivot_graph_feature, target_information], axis=-1)), axis=1)

        return result
