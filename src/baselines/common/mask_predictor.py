import tensorflow as tf
from baselines.a2c.utils import fc, fc_relu, conv, ortho_init, fc_sigmoid
from baselines.common.distributions import make_pdtype

import gym
import numpy as np

class Init_FC(tf.keras.Model):
    def __init__(self, hidden_dim=64, init_dim = 4, target_dim = 192, layer_type=0, layer_num=0):
        super(Init_FC, self).__init__()

        self.fc0 = fc_relu(init_dim + target_dim, 'init_{}_fc0_{}'.format(layer_type, layer_num), hidden_dim, init_scale=np.sqrt(2))

        self.fc1 = fc_relu(hidden_dim + target_dim, 'init_{}_fc1_{}'.format(layer_type, layer_num), hidden_dim, init_scale=np.sqrt(2))

        # self.fc2 = fc_relu(hidden_dim + target_dim, 'init_{}_fc2_{}'.format(layer_type, layer_num), hidden_dim, init_scale=np.sqrt(2))
        self.fc2 = fc(hidden_dim + target_dim, 'init_{}_fc2_{}'.format(layer_type, layer_num), hidden_dim, init_scale=np.sqrt(2))

    def call(self, init_feature, target_feature):
        inputs = init_feature

        x = self.fc0(tf.concat([init_feature, target_feature], axis=-1))
        x = self.fc1(tf.concat([x, target_feature], axis=-1))
        x = self.fc2(tf.concat([x, target_feature], axis=-1))

        return x

class Init_FC_No_Target(tf.keras.Model):
    def __init__(self, hidden_dim=64, init_dim = 4, layer_type=0, layer_num=0):
        super(Init_FC_No_Target, self).__init__()

        self.fc0 = fc_relu(init_dim, 'init_{}_fc0_{}'.format(layer_type, layer_num), hidden_dim, init_scale=np.sqrt(2))

        self.fc1 = fc_relu(hidden_dim, 'init_{}_fc1_{}'.format(layer_type, layer_num), hidden_dim, init_scale=np.sqrt(2))

        self.fc2 = fc_relu(hidden_dim, 'init_{}_fc2_{}'.format(layer_type, layer_num), hidden_dim, init_scale=np.sqrt(2))

    def call(self, init_feature):
        x = self.fc0(init_feature)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

class Mask_Predictor_GNN(tf.keras.Model):
    def __init__(self, hidden_dim=64, node_dim = 4, edge_dim=4, layer_num=0):
        super(Mask_Predictor_GNN, self).__init__()

        self.feature_fc = fc_relu(hidden_dim + node_dim, 'feat_fc_{}'.format(int(layer_num)), hidden_dim, init_scale=np.sqrt(2))

        self.edge_fc = fc_relu(node_dim + node_dim + edge_dim, 'edge_fc_{}'.format(int(layer_num)), hidden_dim, init_scale=np.sqrt(2))

    def call(self, node_feature, adjacency, edge_feature, node_mask, training=True):
        tiled_node_feature = tf.tile(tf.expand_dims(node_feature, axis=1), (1, node_feature.shape[1], 1, 1))
        node_node_feature = tf.concat((tf.transpose(tiled_node_feature, (0, 2, 1, 3)), tiled_node_feature), axis=-1)

        edge_feature = self.edge_fc(tf.concat((node_node_feature, edge_feature), axis=-1))

        message = tf.reduce_sum(tf.math.multiply(edge_feature, adjacency[..., None]), axis=2)

        '''Shape : (num_envs, max_node, hidden_dim)'''
        ending_feature = self.feature_fc(tf.concat([message, node_feature], axis=-1))

        return ending_feature, edge_feature

class Mask_Predictor(tf.Module):
    """
    Encapsulates fields and methods for RL policy and value function estimation with shared parameters
    """

    def __init__(self, ac_space, hidden_dim):
        """
        Parameters:
        ----------
        ac_space        action space

        policy_network  keras network for policy

        value_network   keras network for value

        estimate_q      q value or v value

        """

        # hidden_dim = 64
        reshaped_hidden_dim = hidden_dim

        edge_init_dim = 4

        self.pivot_init_node_fc = Init_FC_No_Target(hidden_dim=reshaped_hidden_dim, layer_type=0)

        self.pivot_init_edge_fc = Init_FC_No_Target(hidden_dim=reshaped_hidden_dim, layer_type=1)

        self.pivot_gcn_init = Mask_Predictor_GNN(node_dim=reshaped_hidden_dim, hidden_dim=reshaped_hidden_dim, edge_dim=reshaped_hidden_dim, layer_num=1)

        self.pivot_gcn_block = []

        for i in range(2):
            temp_gcn = Mask_Predictor_GNN(node_dim=reshaped_hidden_dim, hidden_dim=reshaped_hidden_dim, edge_dim=reshaped_hidden_dim, layer_num=i+2)

            self.pivot_gcn_block.append(temp_gcn)

        self.offset_init_node_fc = Init_FC_No_Target(hidden_dim=reshaped_hidden_dim, layer_type=2)

        self.offset_init_edge_fc = Init_FC_No_Target(hidden_dim=reshaped_hidden_dim, layer_type=3)

        self.offset_gcn_init = Mask_Predictor_GNN(node_dim=reshaped_hidden_dim, hidden_dim=reshaped_hidden_dim, edge_dim=reshaped_hidden_dim, layer_num=1)

        self.offset_gcn_block = []

        for i in range(2):
            temp_gcn = Mask_Predictor_GNN(node_dim=reshaped_hidden_dim, hidden_dim=reshaped_hidden_dim, edge_dim=reshaped_hidden_dim, layer_num=i+2)

            self.offset_gcn_block.append(temp_gcn)

        self.pivot_mask_predictor = fc_sigmoid(reshaped_hidden_dim, 'piv_mask', 1, init_scale=np.sqrt(2))

        self.offset_mask_predictor = fc_sigmoid(reshaped_hidden_dim, 'off_mask', ac_space.n, init_scale=np.sqrt(2))

    @tf.function
    def predict_mask(self, observation, inference=False):
        """
        Compute next action(s) given the observation(s)

        Parameters:
        ----------

        observation     batched observation data

        Returns:
        -------
        (action, value estimate, next state, negative log likelihood of the action under current policy parameters) tuple
        """
        node_feature, adjacency, node_mask, edge_feature = \
            tuple([observation.get(key) for key in ['node_attributes', 'adjacency', 'node_mask', 'edge_attributes']])

        pivot_node_feature = self.pivot_init_node_fc(node_feature)

        pivot_edge_feature = self.pivot_init_edge_fc(edge_feature)

        pivot_node_feature, pivot_edge_feature = self.pivot_gcn_init(pivot_node_feature, adjacency, pivot_edge_feature, node_mask)

        for i in range(len(self.pivot_gcn_block)):
            pivot_node_feature, pivot_edge_feature = self.pivot_gcn_block[i](pivot_node_feature, adjacency, pivot_edge_feature, node_mask)

        offset_node_feature = self.offset_init_node_fc(node_feature)

        offset_edge_feature = self.offset_init_edge_fc(edge_feature)

        offset_node_feature, offset_edge_feature = self.offset_gcn_init(offset_node_feature, adjacency, offset_edge_feature, node_mask)

        for i in range(len(self.offset_gcn_block)):
            offset_node_feature, offset_edge_feature = self.offset_gcn_block[i](offset_node_feature, adjacency, offset_edge_feature, node_mask)

        pivot_mask = self.pivot_mask_predictor(pivot_node_feature)

        offset_mask = self.offset_mask_predictor(offset_node_feature)

        if inference:
            return tf.math.round(pivot_mask), tf.math.round(offset_mask)
        else:
            return pivot_mask, offset_mask

class Mask_Predictor_Last_Linear(tf.Module):
    """
    Encapsulates fields and methods for RL policy and value function estimation with shared parameters
    """

    def __init__(self, ac_space, hidden_dim):
        """
        Parameters:
        ----------
        ac_space        action space

        policy_network  keras network for policy

        value_network   keras network for value

        estimate_q      q value or v value

        """

        # hidden_dim = 64
        reshaped_hidden_dim = hidden_dim

        edge_init_dim = 4

        self.pivot_init_node_fc = Init_FC_No_Target(hidden_dim=reshaped_hidden_dim, layer_type=0)

        self.pivot_init_edge_fc = Init_FC_No_Target(hidden_dim=reshaped_hidden_dim, layer_type=1)

        self.pivot_gcn_init = Mask_Predictor_GNN(node_dim=reshaped_hidden_dim, hidden_dim=reshaped_hidden_dim, edge_dim=reshaped_hidden_dim, layer_num=1)

        self.pivot_gcn_block = []

        for i in range(2):
            temp_gcn = Mask_Predictor_GNN(node_dim=reshaped_hidden_dim, hidden_dim=reshaped_hidden_dim, edge_dim=reshaped_hidden_dim, layer_num=i+2)

            self.pivot_gcn_block.append(temp_gcn)

        self.offset_init_node_fc = Init_FC_No_Target(hidden_dim=reshaped_hidden_dim, layer_type=2)

        self.offset_init_edge_fc = Init_FC_No_Target(hidden_dim=reshaped_hidden_dim, layer_type=3)

        self.offset_gcn_init = Mask_Predictor_GNN(node_dim=reshaped_hidden_dim, hidden_dim=reshaped_hidden_dim, edge_dim=reshaped_hidden_dim, layer_num=1)

        self.offset_gcn_block = []

        for i in range(2):
            temp_gcn = Mask_Predictor_GNN(node_dim=reshaped_hidden_dim, hidden_dim=reshaped_hidden_dim, edge_dim=reshaped_hidden_dim, layer_num=i+2)

            self.offset_gcn_block.append(temp_gcn)

        self.pre_pivot_mask_fc = fc(reshaped_hidden_dim, 'pre_piv_mask', reshaped_hidden_dim, init_scale=np.sqrt(2))

        self.pre_offset_mask_fc = fc(reshaped_hidden_dim, 'pre_off_mask', reshaped_hidden_dim, init_scale=np.sqrt(2))

        self.pivot_mask_predictor = fc_sigmoid(reshaped_hidden_dim, 'piv_mask', 1, init_scale=np.sqrt(2))

        self.offset_mask_predictor = fc_sigmoid(reshaped_hidden_dim, 'off_mask', ac_space.n, init_scale=np.sqrt(2))

    @tf.function
    def predict_mask(self, observation, inference=False):
        """
        Compute next action(s) given the observation(s)

        Parameters:
        ----------

        observation     batched observation data

        Returns:
        -------
        (action, value estimate, next state, negative log likelihood of the action under current policy parameters) tuple
        """
        node_feature, adjacency, node_mask, edge_feature = \
            tuple([observation.get(key) for key in ['node_attributes', 'adjacency', 'node_mask', 'edge_attributes']])

        pivot_node_feature = self.pivot_init_node_fc(node_feature)

        pivot_edge_feature = self.pivot_init_edge_fc(edge_feature)

        pivot_node_feature, pivot_edge_feature = self.pivot_gcn_init(pivot_node_feature, adjacency, pivot_edge_feature, node_mask)

        for i in range(len(self.pivot_gcn_block)):
            pivot_node_feature, pivot_edge_feature = self.pivot_gcn_block[i](pivot_node_feature, adjacency, pivot_edge_feature, node_mask)

        offset_node_feature = self.offset_init_node_fc(node_feature)

        offset_edge_feature = self.offset_init_edge_fc(edge_feature)

        offset_node_feature, offset_edge_feature = self.offset_gcn_init(offset_node_feature, adjacency, offset_edge_feature, node_mask)

        for i in range(len(self.offset_gcn_block)):
            offset_node_feature, offset_edge_feature = self.offset_gcn_block[i](offset_node_feature, adjacency, offset_edge_feature, node_mask)

        pivot_mask = self.pivot_mask_predictor(self.pre_pivot_mask_fc(pivot_node_feature))

        offset_mask = self.offset_mask_predictor(self.pre_offset_mask_fc(offset_node_feature))

        if inference:
            return tf.math.round(pivot_mask), tf.math.round(offset_mask)
        else:
            return pivot_mask, offset_mask

class Mask_Predictor_One_Hop(tf.Module):
    """
    Encapsulates fields and methods for RL policy and value function estimation with shared parameters
    """

    def __init__(self, ac_space, hidden_dim):
        """
        Parameters:
        ----------
        ac_space        action space

        policy_network  keras network for policy

        value_network   keras network for value

        estimate_q      q value or v value

        """

        # hidden_dim = 64
        reshaped_hidden_dim = hidden_dim * 4

        edge_init_dim = 4

        self.pivot_init_node_fc = Init_FC_No_Target(hidden_dim=reshaped_hidden_dim, layer_type=0)

        self.pivot_init_edge_fc = Init_FC_No_Target(hidden_dim=reshaped_hidden_dim, layer_type=1)

        self.pivot_gcn_init = Mask_Predictor_GNN(node_dim=reshaped_hidden_dim, hidden_dim=reshaped_hidden_dim, edge_dim=reshaped_hidden_dim, layer_num=1)

        # self.pivot_gcn_block = []
        #
        # for i in range(2):
        #     temp_gcn = Mask_Predictor_GNN(node_dim=reshaped_hidden_dim, hidden_dim=reshaped_hidden_dim, edge_dim=reshaped_hidden_dim, layer_num=i+2)
        #
        #     self.pivot_gcn_block.append(temp_gcn)

        self.offset_init_node_fc = Init_FC_No_Target(hidden_dim=reshaped_hidden_dim, layer_type=2)

        self.offset_init_edge_fc = Init_FC_No_Target(hidden_dim=reshaped_hidden_dim, layer_type=3)

        self.offset_gcn_init = Mask_Predictor_GNN(node_dim=reshaped_hidden_dim, hidden_dim=reshaped_hidden_dim, edge_dim=reshaped_hidden_dim, layer_num=1)

        # self.offset_gcn_block = []
        #
        # for i in range(2):
        #     temp_gcn = Mask_Predictor_GNN(node_dim=reshaped_hidden_dim, hidden_dim=reshaped_hidden_dim, edge_dim=reshaped_hidden_dim, layer_num=i+2)
        #
        #     self.offset_gcn_block.append(temp_gcn)

        self.pivot_mask_predictor = fc_sigmoid(reshaped_hidden_dim, 'piv_mask', 1, init_scale=np.sqrt(2))

        self.offset_mask_predictor = fc_sigmoid(reshaped_hidden_dim, 'off_mask', ac_space.n, init_scale=np.sqrt(2))

    @tf.function
    def predict_mask(self, observation, inference=False):
        """
        Compute next action(s) given the observation(s)

        Parameters:
        ----------

        observation     batched observation data

        Returns:
        -------
        (action, value estimate, next state, negative log likelihood of the action under current policy parameters) tuple
        """
        node_feature, node_mask, edge_feature = \
            tuple([observation.get(key) for key in ['node_attributes', 'node_mask', 'edge_attributes']])

        adjacency = tf.cast(tf.norm(edge_feature[...,:-1], axis=-1) < 2.5, tf.float32)

        pivot_node_feature = self.pivot_init_node_fc(node_feature)

        pivot_edge_feature = self.pivot_init_edge_fc(edge_feature)

        pivot_node_feature, pivot_edge_feature = self.pivot_gcn_init(pivot_node_feature, adjacency, pivot_edge_feature, node_mask)

        # for i in range(len(self.pivot_gcn_block)):
        #     pivot_node_feature, pivot_edge_feature = self.pivot_gcn_block[i](pivot_node_feature, adjacency, pivot_edge_feature, node_mask)

        offset_node_feature = self.offset_init_node_fc(node_feature)

        offset_edge_feature = self.offset_init_edge_fc(edge_feature)

        offset_node_feature, offset_edge_feature = self.offset_gcn_init(offset_node_feature, adjacency, offset_edge_feature, node_mask)

        # for i in range(len(self.offset_gcn_block)):
        #     offset_node_feature, offset_edge_feature = self.offset_gcn_block[i](offset_node_feature, adjacency, offset_edge_feature, node_mask)

        pivot_mask = self.pivot_mask_predictor(pivot_node_feature)

        offset_mask = self.offset_mask_predictor(offset_node_feature)

        if inference:
            return tf.math.round(pivot_mask), tf.math.round(offset_mask)
        else:
            return pivot_mask, offset_mask
