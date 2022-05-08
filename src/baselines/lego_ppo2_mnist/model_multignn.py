import tensorflow as tf
from baselines.common.lego_policies_mnist import PolicyWithValue_Lego_Mnist_MultiGNN

try:
    from baselines.common.mpi_adam_optimizer import MpiAdamOptimizer
    from mpi4py import MPI
    from baselines.common.mpi_util import sync_from_root
except ImportError:
    MPI = None

import numpy as np
import math

class Model(tf.Module):
    """
    We use this object to :
    __init__:
    - Creates the step_model
    - Creates the train_model

    train():
    - Make the training part (feedforward and retropropagation of gradients)

    save/load():
    - Save load the model
    """
    def __init__(self, *, ac_space, policy_network, target_network, hidden_dim, target_hidden_dim, value_network=None, ent_coef, vf_coef, max_grad_norm):
        super(Model, self).__init__(name='PPO2Model')
        self.train_model = PolicyWithValue_Lego_Mnist_MultiGNN(ac_space, policy_network, target_network, hidden_dim, target_hidden_dim, value_network, estimate_q=False)
        self.var_list = tuple(self.train_model.trainable_variables)
        if MPI is not None:
          self.optimizer = MpiAdamOptimizer(MPI.COMM_WORLD, self.var_list)
        else:
          self.optimizer = tf.keras.optimizers.Adam()
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.step = self.train_model.step
        self.value = self.train_model.value
        self.initial_state = self.train_model.initial_state
        self.supp_coef = 1.0

        # 'mask_precision', 'mask_recall'
        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac', 'pivot_pg_loss', 'pivot_entropy', 'pivot_approxkl',
                           'max_valid_prob', 'max_raw_prob', 'max_invalid_prob', 'num_max_valid', 'num_max_invalid', 'mask_loss', 'invs_mask_loss', 'supp_probs_loss']
        if MPI is not None:
          sync_from_root(self.variables)

    def train(self, lr, cliprange, node_feature, adjacency, pivot, node_mask, target_information, edge_feature,
              next_node_feature, next_adjacency,
              returns, masks, actions, values, neglogpac_old, pivot_neglogpac_old, available_actions, start, epoch, nbatch_train):
        grads, pg_loss, vf_loss, entropy, approxkl, clipfrac, \
        pivot_pg_loss, pivot_entropy, pivot_approxkl, \
        max_valid_prob, max_raw_prob, max_invalid_prob, num_max_valid, num_max_invalid, mask_loss, \
        per_pivot_available_actions, invs_mask_loss, supp_probs_loss = self.get_grad(cliprange, node_feature, adjacency, pivot, node_mask, target_information, edge_feature,
                                                                     next_node_feature, next_adjacency, returns, actions, values,
                                                                     neglogpac_old, pivot_neglogpac_old, available_actions, start, epoch, nbatch_train)

        # mask_precision = tf.keras.metrics.Precision()
        # mask_precision.update_state(per_pivot_available_actions, rounded_predicted_mask)
        #
        # mask_recall = tf.keras.metrics.Recall()
        # mask_recall.update_state(per_pivot_available_actions, rounded_predicted_mask)

        if tf.math.is_nan(entropy) or tf.math.is_nan(pivot_entropy) or tf.math.is_nan(approxkl) or tf.math.is_nan(pivot_approxkl):
            import pdb; pdb.set_trace()

        if MPI is not None:
            self.optimizer.apply_gradients(grads, lr)
        else:
            self.optimizer.learning_rate = lr
            grads_and_vars = zip(grads, self.var_list)
            self.optimizer.apply_gradients(grads_and_vars)

        # mask_precision.result().numpy(), mask_recall.result().numpy()
        return pg_loss, vf_loss, entropy, approxkl, clipfrac, pivot_pg_loss, pivot_entropy, pivot_approxkl, max_valid_prob, max_raw_prob, max_invalid_prob, num_max_valid, num_max_invalid, mask_loss, invs_mask_loss, supp_probs_loss

    @tf.function
    def get_grad(self, cliprange, node_feature, adjacency, pivot, node_mask, target_information, edge_feature,
                 next_node_feature, next_adjacency, returns, actions, values, neglogpac_old, pivot_neglogpac_old, available_actions, start, epoch, nbatch_train):
        # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
        # Returns = R + yV(s')
        advs = returns - values

        # Normalize the advantages
        advs = (advs - tf.reduce_mean(advs)) / (tf.keras.backend.std(advs) + 1e-8)

        temperature = 2.0

        with tf.GradientTape() as tape:
            target_information = self.train_model.target_network(target_information)
            tiled_target_information = tf.tile(tf.expand_dims(target_information, axis=1), (1, node_feature.shape[1], 1))

            node_feature = self.train_model.init_node_fc(node_feature, tiled_target_information)

            two_tiled_target_information = tf.tile(tf.expand_dims(tiled_target_information, axis=1), (1, edge_feature.shape[1], 1, 1))
            edge_feature = self.train_model.init_edge_fc(edge_feature, two_tiled_target_information)

            action_node_feature, action_edge_feature = self.train_model.gcn_init(node_feature, adjacency, target_information, edge_feature, node_mask)

            layer_wise_node_feature = [tf.reduce_mean(action_node_feature)]

            for i in range(len(self.train_model.gcn_block)):
                action_node_feature, action_edge_feature = self.train_model.gcn_block[i](action_node_feature, adjacency, target_information, action_edge_feature, node_mask)

                layer_wise_node_feature.append(tf.reduce_mean(action_node_feature))

            pivot_node_feature, pivot_edge_feature = self.train_model.pivot_gcn_init(node_feature, adjacency, target_information, edge_feature, node_mask)

            for i in range(len(self.train_model.gcn_block)):
                pivot_node_feature, pivot_edge_feature = self.train_model.pivot_gcn_block[i](pivot_node_feature, adjacency, target_information, pivot_edge_feature, node_mask)

            action_graph_feature = tf.math.divide(tf.reduce_sum(tf.multiply(action_node_feature, node_mask), axis = 1),
                                                  tf.reduce_sum(node_mask, axis = 1))

            pivot_graph_feature = tf.math.divide(tf.reduce_sum(tf.multiply(pivot_node_feature, node_mask), axis = 1),
                                                 tf.reduce_sum(node_mask, axis = 1))

            vpred = self.train_model.value_fc(tf.concat([action_graph_feature, pivot_graph_feature, target_information], axis=-1))
            vpredclipped = values + tf.clip_by_value(vpred - values, -cliprange, cliprange)
            vf_losses1 = tf.square(vpred - returns)
            vf_losses2 = tf.square(vpredclipped - returns)
            vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

            for_pivot_node_logits = tf.squeeze(self.train_model.pivot_classifier(tf.concat([pivot_node_feature, tiled_target_information], axis=-1)), axis=2)

            for_pivot_mask = tf.cast(tf.reduce_sum(available_actions, axis = 2) == 0, dtype=for_pivot_node_logits.dtype) * float(-1e8)

            masked_for_pivot_node_logits = for_pivot_node_logits + tf.cast(for_pivot_mask, dtype=for_pivot_node_logits.dtype)

            pivot_cce = tf.keras.losses.CategoricalCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)

            pivot_probs = tf.nn.softmax(masked_for_pivot_node_logits, axis=-1)

            pivot_neglogp = tf.expand_dims(pivot_cce(y_true=tf.one_hot(pivot, for_pivot_node_logits.shape[-1]), y_pred=tf.expand_dims(pivot_probs, axis=1)), axis=-1)

            pivot_a0 = for_pivot_node_logits - tf.reduce_max(for_pivot_node_logits, axis=-1, keepdims=True)
            pivot_ea0 = tf.exp(pivot_a0)
            pivot_z0 = tf.reduce_sum(pivot_ea0, axis=-1, keepdims=True)
            pivot_p0 = pivot_ea0 / (pivot_z0 + 1e-8)

            pivot_entropy = tf.reduce_mean(tf.reduce_sum(pivot_p0 * (tf.math.log(pivot_z0 + 1e-8) - pivot_a0), axis=-1))
            # pivot_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=for_pivot_node_logits, labels=tf.nn.softmax(for_pivot_node_logits, axis=-1)))

            node_latent_representation = tf.gather_nd(action_node_feature, pivot, batch_dims=1)

            per_pivot_available_actions = tf.gather_nd(available_actions, pivot, batch_dims=1)

            pd, pi = self.train_model.pdtype.pdfromlatent(tf.concat([node_latent_representation, target_information], axis=-1))

            # predicted_mask = self.train_model.mask_predictor(node_latent_representation)
            # rounded_predicted_mask = tf.math.round(predicted_mask)
            #
            # bce = tf.keras.losses.BinaryCrossentropy()
            # mask_loss = bce(y_true=per_pivot_available_actions, y_pred=predicted_mask)
            mask_loss = 0

            negative_logits_mask = (tf.ones_like(per_pivot_available_actions) - per_pivot_available_actions) * float(-1e8)

            masked_action_logits = pi + tf.cast(negative_logits_mask, pi.dtype)

            cce = tf.keras.losses.CategoricalCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)

            probs = tf.nn.softmax(masked_action_logits, axis=-1)

            neglogpac = tf.expand_dims(cce(y_true=tf.one_hot(actions, pi.shape[-1]), y_pred=tf.expand_dims(probs, axis=1)), axis=-1)

            # probs_mask_loss = tf.reduce_mean(tf.reduce_sum(tf.nn.softmax(pi, axis=-1) * per_pivot_available_actions, axis=-1))

            # invert_mask = tf.cast((tf.ones_like(per_pivot_available_actions) - per_pivot_available_actions), pi.dtype)
            # invs_mask_loss = tf.reduce_mean(tf.reduce_sum(tf.nn.softmax(pi, axis=-1) * invert_mask, axis=-1))
            invs_mask_loss = 0

            # supp_probs_loss = tf.reduce_mean(tf.reduce_max(tf.nn.softmax(pi, axis=-1) * per_pivot_available_actions, axis=-1))
            supp_probs_loss = 0

            entropy = tf.reduce_mean(pd.entropy())
            # entropy = tf.nn.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pi, labels=tf.nn.softmax(pi, axis=-1)))

            ratio = tf.exp(neglogpac_old - neglogpac)
            pg_losses1 = -advs * ratio
            pg_losses2 = -advs * tf.clip_by_value(ratio, 1-cliprange, 1+cliprange)
            pg_loss = tf.reduce_mean(tf.maximum(pg_losses1, pg_losses2))

            pivot_ratio = tf.exp(pivot_neglogpac_old - pivot_neglogp)
            pivot_pg_losses1 = -advs * pivot_ratio
            pivot_pg_losses2 = -advs * tf.clip_by_value(pivot_ratio, 1-cliprange, 1+cliprange)
            pivot_pg_loss = tf.reduce_mean(tf.maximum(pivot_pg_losses1, pivot_pg_losses2))

            approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - neglogpac_old))
            clipfrac = tf.reduce_mean(tf.cast(tf.greater(tf.abs(ratio - 1.0), cliprange), tf.float32))

            pivot_approxkl = .5 * tf.reduce_mean(tf.square(pivot_neglogp - pivot_neglogpac_old))

            max_valid_prob = tf.reduce_mean(tf.reduce_max((tf.nn.softmax(pi, axis=-1) * per_pivot_available_actions), axis=-1))
            max_raw_prob = tf.reduce_mean(tf.reduce_max((tf.nn.softmax(pi, axis=-1)), axis=-1))
            max_invalid_prob = tf.reduce_mean(tf.reduce_max((tf.nn.softmax(pi, axis=-1) * (tf.ones_like(per_pivot_available_actions)- per_pivot_available_actions)), axis=-1))

            max_prob = tf.cast(tf.one_hot(tf.math.argmax(pi, axis=-1), pi.shape[-1]), tf.float32)

            num_max_valid = tf.reduce_sum(max_prob * per_pivot_available_actions) / pi.shape[0]
            num_max_invalid = tf.reduce_sum(max_prob * (tf.ones_like(per_pivot_available_actions)- per_pivot_available_actions)) / pi.shape[0]

            # loss = pg_loss + pivot_pg_loss - entropy * self.ent_coef + vf_loss * self.vf_coef + mask_loss + invs_mask_loss + supp_probs_loss * self.supp_coef
            loss = pg_loss + pivot_pg_loss - entropy * self.ent_coef + vf_loss * self.vf_coef

        grads = tape.gradient(loss, self.var_list)

        if self.max_grad_norm is not None:
            grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
        if MPI is not None:
            grads = tf.concat([tf.reshape(g, (-1,)) for g in grads], axis=0)
        return grads, pg_loss, vf_loss, entropy, approxkl, clipfrac, pivot_pg_loss, pivot_entropy, pivot_approxkl, \
               max_valid_prob, max_raw_prob, max_invalid_prob, num_max_valid, num_max_invalid, mask_loss, per_pivot_available_actions, \
               invs_mask_loss, supp_probs_loss
