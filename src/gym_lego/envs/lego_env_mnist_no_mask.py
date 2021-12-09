import numpy as np
import math
import rules
import copy
import gym
import os
import tensorflow as tf

from geometric_primitives import brick
from geometric_primitives import bricks
from geometric_primitives import utils_meshes

import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

from rules.rules_mnist import LIST_RULES_2_4
import constants

str_path = constants.str_path_mnist

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

voxel_width = 14
voxel_dim = (14, 8, 14)

class LegoEnv_Mnist_No_Mask(gym.Env):
    def __init__(self, max_bricks = 500, class_to_build = 0, num_bricks_to_build = None,
                 testing = False, target_class_conditioned = False, test_overfitting = True):
        self.num_max_bricks = max_bricks
        self.num_bricks_to_build = num_bricks_to_build
        self.class_to_build = class_to_build

        self.change_coordinate = None

        self.stacked_representation = None

        self.testing = testing
        self.target_class_conditioned = target_class_conditioned
        self.test_overfitting = test_overfitting
        self.target_voxel = None
        self.prev_reward = None

        self.max_node_num = 45

        self.loop_target_index = 0

        self.action_space = gym.spaces.Discrete(len(LIST_RULES_2_4) * 2)

        self.observation_space = gym.spaces.Dict({'adjacency' : gym.spaces.Box(low = 0, high = 1, shape = (self.max_node_num, self.max_node_num), dtype = np.float32),
                                                  'node_attributes' : gym.spaces.Box(low = (-1) * voxel_width, high = voxel_width, shape = (self.max_node_num, 4), dtype = np.float32),
                                                  'node_mask' : gym.spaces.Box(low = 0, high = 1., shape = (self.max_node_num, 1), dtype = np.float32),
                                                  'target_information' : gym.spaces.Box(low = 0., high = 1., shape = (14, 14), dtype = np.float32),
                                                  'edge_attributes' : gym.spaces.Box(low = -20., high = 20., shape = (self.max_node_num, self.max_node_num, 4), dtype = np.float32),
                                                  'target_class' : gym.spaces.Discrete(1),
                                                  'current_pic' : gym.spaces.Box(low = 0., high = 1., shape = (14, 14), dtype = np.float32)})

    def reset(self):
        self.target_voxel, self.target, self.target_num_brick, \
        self.target_class = self.load_target_from_class(class_info=None)

        self.target_embedding = self.target

        btv_index = np.min(np.where(np.sum(self.target_voxel, axis=(0,1)) > 0))
        btv = self.target_voxel[:,:,btv_index]
        bottom_trans_pad = np.array(list(map(lambda i : (list(map(lambda j : np.sum(btv[i-1, j-1:j+2] +
                                                                                    btv[i, j-1:j+2] +
                                                                                    btv[i+1, j-1:j+2]), range(1, btv.shape[1]-1)))),
                                            range(1, btv.shape[0]-1))))

        bottom_trans = np.zeros((btv.shape[0], btv.shape[1])).astype(bottom_trans_pad.dtype)
        bottom_trans[1:-1, 1:-1] = bottom_trans_pad

        x_s, y_s = np.where(bottom_trans == np.max(bottom_trans))

        self.translation = np.array([x_s[0] + 1, y_s[0] + 1, btv_index, 0])

        brick_ = brick.Brick()

        ''' 0 : Vertical / 1 : Horizontal '''
        random_direction = np.random.randint(2)

        random_x = np.random.randint(-3, 4)
        random_y = np.random.randint(-3, 4)

        self.random_translation = np.array([random_x, random_y, 0, 0])

        # brick_.set_position(self.random_translation[:-1])
        # brick_.set_direction(random_direction)
        brick_.set_position([0, 0, 0])
        brick_.set_direction(0)

        self.bricks_ = bricks.Bricks(self.num_max_bricks, '0')
        self.bricks_.add(brick_)

        init_node_coordinates = np.concatenate((brick_.get_position(), [brick_.get_direction()]))

        node_matrix = np.zeros((self.max_node_num, 4), dtype=init_node_coordinates.dtype)
        node_matrix[0] = init_node_coordinates

        # node_matrix = np.ones((self.max_node_num, 4), dtype=init_node_coordinates.dtype)

        node_mask = np.expand_dims(np.eye(self.max_node_num, dtype=np.float32)[0], axis=-1)

        self.brick_voxel = np.zeros(shape = voxel_dim, dtype=np.int32)

        self._occupy(init_node_coordinates, self.brick_voxel)

        self.prev_reward = np.sum(np.logical_and(self.brick_voxel, self.target_voxel)) / np.sum(np.logical_or(self.brick_voxel, self.target_voxel))
        self.prev_abs_reward = np.sum(np.logical_and(self.brick_voxel, self.target_voxel))
        self.last_accum_reward = 0

        X, A, _, _ = self.bricks_.get_graph()

        A += 1 * np.eye(A.shape[0], dtype=A.dtype)

        A = A / np.sum(A, axis = 1)[:, np.newaxis]

        adjacency_matrix = np.zeros((self.max_node_num, self.max_node_num), dtype=A.dtype)
        adjacency_matrix[:A.shape[0],:A.shape[0]] = A

        edge_attributes = np.zeros((self.max_node_num, self.max_node_num, 4)).astype(np.float32)
        displacement = X[np.newaxis, :] - X[:, np.newaxis]
        edge_attributes[:displacement.shape[0], :displacement.shape[1], :] = displacement

        current_pic = self._get_current_pic()

        self.obs = dict(zip(['adjacency', 'node_attributes', 'node_mask', 'target_information', 'edge_attributes', 'target_class', 'current_pic'],
                            (adjacency_matrix.astype(np.float32), node_matrix.astype(np.float32), node_mask.astype(np.float32),
                             self.target_embedding, edge_attributes, self.target_class, current_pic)))

        return self.get_state()

    def step(self, actions):
        pivot_fragment_index, relative_action, pretrain_step = actions

        if pretrain_step == 0:
            done = False
            available_actions = self.get_masks()

            self._update_graph_randomly()

            X, A, _, _ = self.bricks_.get_graph()

            # if np.sum(self.obs['node_mask']) >= self.target_num_brick[0] or A.shape[0] >= self.target_num_brick[0]:
            if np.sum(self.obs['node_mask']) > 30:
                done = True

            return self.get_state(), 0, done, available_actions
        else:
            done = False

            new_brick_coordinate, valid_flag = self._add_node_and_edge(pivot_fragment_index, relative_action)

            if valid_flag == False:
                # episode_reward = self.get_episode_reward()
                episode_reward = self.get_episode_reward_with_intermediate_as_well()

                return self.get_state(), 0., True, episode_reward
            else:
                self._update_graph(new_brick_coordinate)

                reward = self.calculate_reward(valid_flag)
                episode_reward = self.get_episode_reward()

                X, A, _, _ = self.bricks_.get_graph()

                if np.sum(self.obs['node_mask']) >= self.target_num_brick[0] or A.shape[0] >= self.target_num_brick[0]:
                    done = True

                if np.sum(self.obs['node_mask']) == self.max_node_num:
                    done = True
                    print(self.loop_target_index)

                if valid_flag == False:
                    done = True

                return self.get_state(), reward, done, episode_reward

    def render(self):
        visualization.visualize(self.bricks_)

    def get_state(self):
        return copy.deepcopy(self.obs)

    def _add_node_and_edge(self, pivot_fragment_index, relative_action):
        X, A, _, _ = self.bricks_.get_graph()

        if pivot_fragment_index > X.shape[0] - 1:
            return np.zeros(4), False

        pivot_coordinate = X[pivot_fragment_index]

        pivot_position, pivot_direction = pivot_coordinate[:-1], pivot_coordinate[-1]

        rules_index, height_diff = (int(relative_action - len(LIST_RULES_2_4)), 1) if relative_action >= len(LIST_RULES_2_4) else (int(relative_action), 0)

        positional_change =  np.concatenate((np.array(LIST_RULES_2_4[rules_index][1][1]),
                                             np.array([1 - 2 * height_diff])))
        directional_change = np.array([LIST_RULES_2_4[rules_index][1][0]])

        change_coordinate = np.concatenate([positional_change, directional_change]).tolist()

        if pivot_direction == 1:
            change_coordinate = self._reposition(change_coordinate)

        new_brick_coordinate = np.concatenate([pivot_position, [0]]) + change_coordinate

        new_brick_ = brick.Brick()
        new_brick_.set_position(new_brick_coordinate[:-1])
        new_brick_.set_direction(new_brick_coordinate[-1])

        try:
            self.bricks_.validate_brick(new_brick_)
            self.bricks_.add(new_brick_)
        except ValueError:
            return new_brick_coordinate, False

        return new_brick_coordinate, True

    def _update_graph(self, new_brick_coordinate):
        X, A, _, _ = self.bricks_.get_graph()

        A += 1 * np.eye(A.shape[0], dtype=A.dtype)

        A = A / np.sum(A, axis = 1)[:, np.newaxis]

        self._occupy(new_brick_coordinate, self.brick_voxel)

        updated_node_attributes = self.obs['node_attributes']
        updated_node_attributes[(A.shape[0] - 1)] =  new_brick_coordinate
        # updated_node_attributes[(A.shape[0] - 1)] =  new_brick_coordinate + self.random_translation

        # updated_node_attributes = np.ones((self.max_node_num, 4), dtype=np.float32)

        adjacency_matrix = np.zeros((self.max_node_num, self.max_node_num), dtype=A.dtype)
        adjacency_matrix[:A.shape[0],:A.shape[0]] = A

        node_mask = np.expand_dims(np.concatenate([np.ones(A.shape[0], dtype=np.float32),
                                                   np.zeros(self.max_node_num - A.shape[0], dtype=np.float32)], axis=-1), axis=-1)

        edge_attributes = np.zeros((self.max_node_num, self.max_node_num, 4)).astype(np.float32)
        displacement = X[np.newaxis, :] - X[:, np.newaxis]
        displacement[...,3] = np.abs(displacement[...,3])
        edge_attributes[:displacement.shape[0], :displacement.shape[1], :] = displacement

        current_pic = self._get_current_pic()

        self.obs.update(zip(['adjacency', 'node_attributes', 'node_mask', 'edge_attributes', 'current_pic'],
                            (adjacency_matrix.astype(np.float32), updated_node_attributes.astype(np.float32),
                             node_mask.astype(np.float32), edge_attributes, current_pic)))

    def _update_graph_randomly(self):
        next_brick = None
        while next_brick is None:
            next_brick = self.bricks_.sample()[0]

        self.bricks_.add(next_brick)

        X, A, _, _ = self.bricks_.get_graph()

        A += 1 * np.eye(A.shape[0], dtype=A.dtype)

        A = A / np.sum(A, axis = 1)[:, np.newaxis]

        new_brick_coordinate = np.array([*(next_brick.get_position()), next_brick.get_direction()])

        self._occupy(new_brick_coordinate, self.brick_voxel)

        updated_node_attributes = self.obs['node_attributes']
        updated_node_attributes[(A.shape[0] - 1)] = new_brick_coordinate

        # updated_node_attributes = np.zeros((self.max_node_num, 4), dtype=np.float32)

        adjacency_matrix = np.zeros((self.max_node_num, self.max_node_num), dtype=A.dtype)
        adjacency_matrix[:A.shape[0],:A.shape[0]] = A

        node_mask = np.expand_dims(np.concatenate([np.ones(A.shape[0], dtype=np.float32),
                                                   np.zeros(self.max_node_num - A.shape[0], dtype=np.float32)], axis=-1), axis=-1)

        edge_attributes = np.zeros((self.max_node_num, self.max_node_num, 4)).astype(np.float32)
        displacement = X[np.newaxis, :] - X[:, np.newaxis]
        displacement[...,3] = np.abs(displacement[...,3])
        edge_attributes[:displacement.shape[0], :displacement.shape[1], :] = displacement

        current_pic = self._get_current_pic()

        self.obs.update(zip(['adjacency', 'node_attributes', 'node_mask', 'edge_attributes', 'current_pic'],
                            (adjacency_matrix.astype(np.float32), updated_node_attributes.astype(np.float32),
                             node_mask.astype(np.float32), edge_attributes, current_pic)))

    def load_target_from_class(self, class_info):
        '''
        mnist_dataset : 14, 14
        '''
        target_class_idx = 0

        if self.testing is False:
            cur_idx = np.random.randint(400) + 100

            target_bricks = np.load(os.path.join(str_path, 'class_{}/target_voxel_train/{:03d}.npy'.format(target_class_idx, cur_idx)))
            target_information = np.load(os.path.join(str_path, 'class_{}/target_information_train/{:03d}.npy'.format(target_class_idx, cur_idx)))
            num_targets = np.load(os.path.join(str_path, 'class_{}/target_num_brick_train/{:03d}.npy'.format(target_class_idx, cur_idx)))
            target_class = np.load(os.path.join(str_path, 'class_{}/target_class_train/{:03d}.npy'.format(target_class_idx, cur_idx)))
        else:
            cur_idx = np.random.randint(100)

            target_bricks = np.load(os.path.join(str_path, 'class_{}/target_voxel_train/{:03d}.npy'.format(target_class_idx, cur_idx)))
            target_information = np.load(os.path.join(str_path, 'class_{}/target_information_train/{:03d}.npy'.format(target_class_idx, cur_idx)))
            num_targets = np.load(os.path.join(str_path, 'class_{}/target_num_brick_train/{:03d}.npy'.format(target_class_idx, cur_idx)))
            target_class = np.load(os.path.join(str_path, 'class_{}/target_class_train/{:03d}.npy'.format(target_class_idx, cur_idx)))

            # target_bricks = np.load('../mnist_easy_dataset/target_voxel_test/{}.npy'.format(cur_idx))
            # target_information = np.load('../mnist_easy_dataset/target_information_test/{}.npy'.format(cur_idx))
            # num_targets = np.load('../mnist_easy_dataset/target_num_brick_test/{}.npy'.format(cur_idx))
            # target_class = np.load('../mnist_easy_dataset/target_class_test/{}.npy'.format(cur_idx))

        self.loop_target_index += 1

        return target_bricks, target_information[...,None].astype(np.float32), num_targets, target_class

    def get_target_embedding(self):
        return self.target

    def _occupy(self, new_coordinate, voxel):
        translation = voxel_width / 2
        # arranging_coordinate = (translation, 4, 1, 0)
        # arranging_coordinate = (0, 0, 0, 0)
        arranging_coordinate = self.translation

        arranged_x, arranged_y, arranged_z, dir = np.multiply(np.array(new_coordinate, dtype = 'int'), np.array([1,1,1,1])) + \
                                                  np.array(arranging_coordinate, dtype = 'int')

        voxel[(arranged_x - 1 - (dir == 1)):(arranged_x + 1 + (dir == 1)),
              (arranged_y - 1 - (dir == 0)):(arranged_y + 1 + (dir == 0)),
              arranged_z:(arranged_z + 1)] = 1

    def _show_voxel(self, voxel):
        fig = plt.figure()
        ax = fig.gca(projection = '3d')
        ax.voxels(voxel, edgecolor = 'k')
        ax.set(xlabel='X', ylabel='Y', zlabel='Z')

        plt.show()

    def get_target_voxel(self, target_embedding):
        target_brick_voxel = np.zeros(shape = voxel_dim, dtype = 'int')

        for new_coordinate in target_embedding:
            self._occupy(new_coordinate, target_brick_voxel)

        return target_brick_voxel

    def calculate_reward(self, valid_flag):
        if valid_flag is False:
            return 0

        cur_reward = np.sum(np.logical_and(self.brick_voxel, self.target_voxel)) / np.sum(np.logical_or(self.brick_voxel, self.target_voxel))

        reward_change = cur_reward - self.prev_reward

        self.prev_reward = cur_reward

        self.last_accum_reward += reward_change

        cur_abs_reward = np.sum(np.logical_and(self.brick_voxel, self.target_voxel))

        abs_reward_change = cur_abs_reward - self.prev_abs_reward

        self.prev_abs_reward = cur_abs_reward

        if abs_reward_change >= 4:
            return abs_reward_change / 8
        else:
            return 0.

    def get_episode_reward(self):
        last_reward = 0

        if np.sum(self.obs['node_mask']) == self.target_num_brick[0]:
            last_reward = np.sum(np.logical_and(self.brick_voxel, self.target_voxel)) / np.sum(np.logical_or(self.brick_voxel, self.target_voxel))

        return last_reward

    def get_episode_reward_with_intermediate_as_well(self):
        last_reward = np.sum(np.logical_and(self.brick_voxel, self.target_voxel)) / np.sum(np.logical_or(self.brick_voxel, self.target_voxel))

        if np.sum(self.obs['node_mask']) >= self.target_num_brick * 0.8:
            return last_reward
        else:
            return (-1.) * last_reward

    def get_masks(self):
        X, A, _, _ = self.bricks_.get_graph()

        updated_available_offsets = np.vstack([self._get_available_offsets_for_pivot(pivot_candidate) for pivot_candidate in X])

        available_offsets_matrix = np.zeros((self.max_node_num, len(LIST_RULES_2_4) * 2))
        available_offsets_matrix[:X.shape[0]] = updated_available_offsets

        return available_offsets_matrix.astype(np.float32)

    def _get_available_offsets_for_pivot(self, pivot_brick_coordinate):
        if self.bricks_ is not None:
            possible_offsets = np.array([self._check_offset_availability(pivot_brick_coordinate, idx) for idx in range(len(LIST_RULES_2_4) * 2)])

            return possible_offsets.astype(np.int32)[None, :]

    def _check_offset_availability(self, pivot_brick_coordinate, action):
        cur_position = pivot_brick_coordinate[:-1]
        cur_direction = pivot_brick_coordinate[-1]

        zero_normalized_pivot_brick_coordinate = np.concatenate([cur_position, [0]])

        height_gap = int(action >= len(LIST_RULES_2_4))
        index = action - len(LIST_RULES_2_4) * int(action >= len(LIST_RULES_2_4))

        positional_change =  np.concatenate((np.array(LIST_RULES_2_4[index][1][1]),
                                             np.array([1 - 2 * height_gap])))
        directional_change = np.array([LIST_RULES_2_4[index][1][0]])

        change_coordinate = np.concatenate([positional_change, directional_change])

        if cur_direction == 1:
            change_coordinate = self._reposition(change_coordinate)

        new_brick_coordinate = zero_normalized_pivot_brick_coordinate + change_coordinate

        new_brick = np.array([action, new_brick_coordinate])

        new_brick_ = brick.Brick()
        new_brick_.set_position(new_brick_coordinate[:-1])
        new_brick_.set_direction(new_brick_coordinate[-1])

        try:
            self.bricks_.validate_brick(new_brick_)
            return 1
        except ValueError:
            return 0

    def _reposition(self, new_coordinate):
        prev_x, prev_y, z, prev_dir = new_coordinate

        # rad = 3/2 * math.pi
        rad = 1/2 * math.pi

        new_x = math.cos(rad) * prev_x - math.sin(rad) * prev_y
        new_y = math.sin(rad) * prev_x + math.cos(rad) * prev_y

        new_dir = (prev_dir + 1) % 2

        return np.array([round(new_x), round(new_y), z, new_dir], dtype = 'int')

    def _get_current_pic(self):
        return np.rot90((np.sum(self.brick_voxel, axis=1) > 0).astype(np.float32), 1)[..., np.newaxis]

if __name__ == '__main__':
    Benv = LegoEnvMnist_No_Mask()

    obs = Benv.reset()

    # print(obs)
    print('\n')

    Benv.render()

    zero_obs, zero_reward, zero_done, zero_voxel = Benv.step(45)

    # print(zero_obs)
    print(zero_reward)
    print('\n')

    Benv.render()
