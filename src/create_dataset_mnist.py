import os
import numpy as np
import tensorflow as tf
import cv2
#import copy

import constants


str_path = constants.str_path_mnist

directories_ = [
    'target_voxel',
    'target_information',
    'target_num_brick',
    'target_class',
]

classes_ = [
    'class_0',
    'class_1',
    'class_2',
    'class_3',
    'class_4',
    'class_5',
    'class_6',
    'class_7',
    'class_8',
    'class_9',
]

if not os.path.exists(str_path):
    os.makedirs(str_path)


for str_cls in classes_:
    cur_dir = os.path.join(str_path, str_cls)

    if not os.path.exists(cur_dir):
        os.makedirs(cur_dir)

    for dir_idx in range(len(directories_)):
        cur_dir_train = os.path.join(cur_dir, directories_[dir_idx] + '_train')

        if not os.path.exists(cur_dir_train):
            os.makedirs(cur_dir_train)

        cur_dir_test = os.path.join(cur_dir, directories_[dir_idx] + '_test')

        if not os.path.exists(cur_dir_test):
            os.makedirs(cur_dir_test)

(X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()
print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

size_height = X_train.shape[1]
size_width = X_train.shape[2]
num_data_train = X_train.shape[0]
num_data_test = X_test.shape[0]

X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)

X_train = np.expand_dims(X_train, axis=3)
X_test = np.expand_dims(X_test, axis=3)

X_train /= 255.0
X_test /= 255.0
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

def bottom_align(arr):
    non_zero_arr = arr[~np.all(arr == 0., axis=1)]

    aligned_arr = np.zeros((28, 28)).astype(arr.dtype)
    aligned_arr[-non_zero_arr.shape[0]:] = non_zero_arr
    aligned_arr[aligned_arr < .6] = 0
    aligned_arr[aligned_arr > .6] = 1

    return aligned_arr

def threshold_resize_bottom_align(arr):
    non_zero_arr = arr[~np.all(arr == 0., axis=1)]

    aligned_arr = np.zeros((28, 28)).astype(arr.dtype)
    aligned_arr[-non_zero_arr.shape[0]:] = non_zero_arr

    smaller_aligned_arr = cv2.resize(aligned_arr, dsize=(14, 14), interpolation = cv2.INTER_CUBIC)
    smaller_aligned_arr[smaller_aligned_arr < .6] = 0
    smaller_aligned_arr[smaller_aligned_arr > .6] = 1

    return smaller_aligned_arr

def voxelize_bottom_align(arr):
    non_zero_arr = arr[~np.all(arr == 0., axis=1)]

    aligned_arr = np.zeros((28, 28)).astype(arr.dtype)
    aligned_arr[-non_zero_arr.shape[0]:] = non_zero_arr
    aligned_arr[aligned_arr < .6] = 0
    aligned_arr[aligned_arr > .6] = 1

    voxel = np.zeros((28, 8, 28))
    voxel[:,3,:] = np.rot90(aligned_arr, 3)
    voxel[:,4,:] = np.rot90(aligned_arr, 3)
    voxel[:,5,:] = np.rot90(aligned_arr, 3)
    voxel[:,6,:] = np.rot90(aligned_arr, 3)

    return voxel

def voxelize_rescale_bottom_align(arr):
    non_zero_arr = arr[~np.all(arr == 0., axis=1)]

    aligned_arr = np.zeros((28, 28)).astype(arr.dtype)
    aligned_arr[-non_zero_arr.shape[0]:] = non_zero_arr

    smaller_aligned_arr = cv2.resize(aligned_arr, dsize=(14, 14), interpolation  = cv2.INTER_CUBIC)
    smaller_aligned_arr[smaller_aligned_arr < .6] = 0
    smaller_aligned_arr[smaller_aligned_arr > .6] = 1

    voxel = np.zeros((14, 8, 14))
    voxel[:,3,:] = np.rot90(smaller_aligned_arr, 3)
    voxel[:,4,:] = np.rot90(smaller_aligned_arr, 3)
    voxel[:,5,:] = np.rot90(smaller_aligned_arr, 3)
    voxel[:,6,:] = np.rot90(smaller_aligned_arr, 3)

    return voxel

def threshold_resize(arr):
    thresholded_arr = cv2.resize(arr, dsize=(14, 14), interpolation = cv2.INTER_CUBIC)
    thresholded_arr[thresholded_arr < .6] = 0
    thresholded_arr[thresholded_arr > .6] = 1

    return thresholded_arr

def voxelize_rescale(arr):
    thresholded_arr = cv2.resize(arr, dsize=(14, 14), interpolation = cv2.INTER_CUBIC)
    thresholded_arr[thresholded_arr < .6] = 0
    thresholded_arr[thresholded_arr > .6] = 1

    voxel = np.zeros((14, 8, 14))
    voxel[:,3,:] = np.rot90(thresholded_arr, 3)
    voxel[:,4,:] = np.rot90(thresholded_arr, 3)
    voxel[:,5,:] = np.rot90(thresholded_arr, 3)
    voxel[:,6,:] = np.rot90(thresholded_arr, 3)

    return voxel

func_bottom_align = lambda x: bottom_align(x)
func_rescale_and_align = lambda x: threshold_resize_bottom_align(x)

Rescaled_X_train = np.array([func_rescale_and_align(img) for img in X_train[..., 0]])
Voxel_X_train = np.array([voxelize_rescale_bottom_align(img) for img in X_train[..., 0]])

Thresh_X_train = np.array([threshold_resize(img) for img in X_train[..., 0]])
Voxel_Thresh_X_train = np.array([voxelize_rescale(img) for img in X_train[..., 0]])

'''1./12/18 for orig'''
num_target_ratio = 1.1
num_target_min = 14
num_target_max = 20

num_train_data = 500
num_test_data = 100

#Temp_Rescale_X_train = [Rescaled_X_train[np.where(Y_train == i)][:1000, ...] for i in [0,1,2,4,6]]
#Temp_Rescale_X_test = [Rescaled_X_train[np.where(Y_train == i)][:1000, ...] for i in [7,9]]

#Temp_Voxel_X_train = [Voxel_X_train[np.where(Y_train == i)][:1000, ...] for i in [0,1,2,4,6]]
#Temp_Voxel_X_test = [Voxel_X_train[np.where(Y_train == i)][:1000, ...] for i in [7,9]]

Bottom_Aligned_X_train = np.array([func_bottom_align(img) for img in X_train[..., 0]])
num_target = np.array([int(np.sum(x) / 2 * num_target_ratio + 1) for x in Thresh_X_train])
num_target_thesholded = num_target[(num_target_min < num_target) & (num_target < num_target_max)]

train_class = [0]
test_class = [0]
cur_class_idx = 0

thresholded_Y = Y_train[(num_target_min < num_target) & (num_target < num_target_max)]
thresholded_X = Thresh_X_train[(num_target_min < num_target) & (num_target < num_target_max)]
thresholded_X_train = [thresholded_X[np.where(thresholded_Y == i)][:num_train_data, ...] for i in train_class]
thresholded_X_test = [thresholded_X[np.where(thresholded_Y == i)][:num_test_data, ...] for i in test_class]
threshoided_Voxel = Voxel_Thresh_X_train[(num_target_min < num_target) & (num_target < num_target_max)]
thresholded_Voxel_train = [threshoided_Voxel[np.where(thresholded_Y == i)][:num_train_data, ...] for i in train_class]
thresholded_Voxel_test = [threshoided_Voxel[np.where(thresholded_Y == i)][:num_test_data, ...] for i in test_class]
thresholded_num_target_train = [np.expand_dims(num_target_thesholded[np.where(thresholded_Y == i)][:num_train_data, ...], axis=-1) for i in train_class]
thresholded_num_target_test = [np.expand_dims(num_target_thesholded[np.where(thresholded_Y == i)][:num_test_data, ...], axis=-1) for i in test_class]

thresholded_Y_train = [np.expand_dims(thresholded_Y[np.where(thresholded_Y == i)][:num_train_data, ...], axis=-1) for i in train_class]
thresholded_Y_test = [np.expand_dims(thresholded_Y[np.where(thresholded_Y == i)][:num_test_data, ...], axis=-1) for i in test_class]

print(np.max(num_target))

MNIST_3D_Target_Information_Train = np.vstack(thresholded_X_train)
MNIST_3D_Target_Information_Test = np.vstack(thresholded_X_test)
MNIST_3D_Target_Voxel_Train = np.vstack(thresholded_Voxel_train)
MNIST_3D_Target_Voxel_Test = np.vstack(thresholded_Voxel_test)
MNIST_3D_Target_Num_Bricks_Train = np.vstack(thresholded_num_target_train)
MNIST_3D_Target_Num_Bricks_Test = np.vstack(thresholded_num_target_test)

MNIST_3D_Target_Class_Train = np.vstack(thresholded_Y_train)
MNIST_3D_Target_Class_Test = np.vstack(thresholded_Y_test)

print(MNIST_3D_Target_Information_Train.shape, MNIST_3D_Target_Information_Test.shape)

train_inds = np.arange(MNIST_3D_Target_Information_Train.shape[0])
test_inds = np.arange(MNIST_3D_Target_Information_Test.shape[0])

np.random.shuffle(train_inds)
np.random.shuffle(test_inds)

MNIST_3D_Target_Information_Train = MNIST_3D_Target_Information_Train[train_inds].astype(np.int32)
MNIST_3D_Target_Information_Test = MNIST_3D_Target_Information_Test[test_inds].astype(np.int32)
MNIST_3D_Target_Voxel_Train = MNIST_3D_Target_Voxel_Train[train_inds].astype(np.int32)
MNIST_3D_Target_Voxel_Test = MNIST_3D_Target_Voxel_Test[test_inds].astype(np.int32)
MNIST_3D_Target_Num_Bricks_Train = MNIST_3D_Target_Num_Bricks_Train[train_inds].astype(np.int32)
MNIST_3D_Target_Num_Bricks_Test = MNIST_3D_Target_Num_Bricks_Test[test_inds].astype(np.int32)

MNIST_3D_Target_Class_Train = MNIST_3D_Target_Class_Train[train_inds].astype(np.int32)
MNIST_3D_Target_Class_Test = MNIST_3D_Target_Class_Test[test_inds].astype(np.int32)

for i in range(MNIST_3D_Target_Information_Train.shape[0]):
    if i % 200 == 0:
        print('Train', i)
    np.save(os.path.join(str_path, 'class_{}/target_information_train/{:03d}.npy'.format(cur_class_idx, i)), MNIST_3D_Target_Information_Train[i])
    np.save(os.path.join(str_path, 'class_{}/target_voxel_train/{:03d}.npy'.format(cur_class_idx, i)), MNIST_3D_Target_Voxel_Train[i])
    np.save(os.path.join(str_path, 'class_{}/target_num_brick_train/{:03d}.npy'.format(cur_class_idx, i)), MNIST_3D_Target_Num_Bricks_Train[i])
    np.save(os.path.join(str_path, 'class_{}/target_class_train/{:03d}.npy'.format(cur_class_idx, i)), MNIST_3D_Target_Class_Train[i])

for i in range(MNIST_3D_Target_Information_Test.shape[0]):
    if i % 200 == 0:
        print('Test', i)
    np.save(os.path.join(str_path, 'class_{}/target_information_test/{:03d}.npy'.format(cur_class_idx, i)), MNIST_3D_Target_Information_Test[i])
    np.save(os.path.join(str_path, 'class_{}/target_voxel_test/{:03d}.npy'.format(cur_class_idx, i)), MNIST_3D_Target_Voxel_Test[i])
    np.save(os.path.join(str_path, 'class_{}/target_num_brick_test/{:03d}.npy'.format(cur_class_idx, i)), MNIST_3D_Target_Num_Bricks_Test[i])
    np.save(os.path.join(str_path, 'class_{}/target_class_test/{:03d}.npy'.format(cur_class_idx, i)), MNIST_3D_Target_Class_Test[i])
