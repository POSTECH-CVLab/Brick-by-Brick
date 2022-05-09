import numpy as np
import time
from datetime import datetime
import os

import brick
import bricks
import rules


def get_coordinates(bricks, axes):
    coordinates = []

    for brick in bricks.get_bricks():
        vertices = brick.get_vertices()

        new_vertices = []
        for vertex in vertices:
            new_vertices.append(vertex[axes])
        new_vertices = np.array(new_vertices)
        new_vertices = np.unique(new_vertices, axis=0)

        coordinates.append(new_vertices)

    coordinates = np.array(coordinates)
    return coordinates

def shuffle_routine(squares):
    new_squares = []

    for square in squares:
        cur_square = [square[0], square[1], square[3], square[2]]
#        cur_square = [square[0], square[1], square[3], square[2], square[0]]
        new_squares.append(cur_square)
    
    new_squares = np.array(new_squares)
    return new_squares

def check_overlap_1d(min_max_1, min_max_2):
    assert isinstance(min_max_1, tuple)
    assert isinstance(min_max_2, tuple)

    return min_max_1[1] > min_max_2[0] and min_max_2[1] > min_max_1[0]

def check_overlap_2d(min_max_1, min_max_2):
    assert len(min_max_1) == 2
    assert len(min_max_2) == 2

    return check_overlap_1d(min_max_1[0], min_max_2[0]) and check_overlap_1d(min_max_1[1], min_max_2[1])

def check_overlap_3d(min_max_1, min_max_2):
    assert len(min_max_1) == 3
    assert len(min_max_2) == 3

    return check_overlap_1d(min_max_1[0], min_max_2[0]) and check_overlap_1d(min_max_1[1], min_max_2[1]) and check_overlap_1d(min_max_1[2], min_max_2[2])

def get_min_max_3d(vertices):
    min_max = [
        (np.min(vertices[:, 0]), np.max(vertices[:, 0])),
        (np.min(vertices[:, 1]), np.max(vertices[:, 1])),
        (np.min(vertices[:, 2]), np.max(vertices[:, 2]))
    ]

    return min_max

def normalize_points(batched_points):
    assert len(batched_points.shape) == 3

    new_batched_points = []

    for points in batched_points:
        center = np.mean(points, axis=0)

        points = points - center

        norms = np.linalg.norm(points, axis=1)
        max_norms = np.max(norms)

        points = points / max_norms

        new_batched_points.append(points)

    new_batched_points = np.array(new_batched_points)
    return new_batched_points

def save_bricks(bricks_, str_path, str_file=None):
    str_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    if str_file is None:
        str_save = os.path.join(str_path, 'bricks_{}.npy'.format(str_time))
    else:
        str_save = os.path.join(str_path, str_file + '.npy')

    np.save(str_save, bricks_)
