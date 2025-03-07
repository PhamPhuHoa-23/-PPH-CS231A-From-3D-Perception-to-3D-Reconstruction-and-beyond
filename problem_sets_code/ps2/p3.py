import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
import scipy.io as sio
import matplotlib.gridspec as gridspec
from epipolar_utils import *

'''
FACTORIZATION_METHOD The Tomasi and Kanade Factorization Method to determine
the 3D structure of the scene and the motion of the cameras.
Arguments:
    points_im1 - N points in the first image that match with points_im2
    points_im2 - N points in the second image that match with points_im1

    Both points_im1 and points_im2 are from the get_data_from_txt_file() method
Returns:
    structure - the structure matrix
    motion - the motion matrix
'''
def factorization_method(points_im1, points_im2):
    centroid_im1 = points_im1.mean(axis=0)
    centroid_im2 = points_im2.mean(axis=0)

    points_im1_norm = points_im1 - centroid_im1
    points_im2_norm = points_im2 - centroid_im2

    D = np.concatenate((points_im1_norm.T[:2,:], points_im2_norm.T[:2,:]), axis=0)
    print(points_im1_norm.shape)
    print(D.shape)

    U, S, Vt = np.linalg.svd(D)

    print(S)
    U3 = U[:,:3]
    S3 = np.diag(S[:3])
    V3t = Vt[:3]

    M = U3 @ np.sqrt(S3)
    S = np.sqrt(S3) @ V3t

    return S, M

if __name__ == '__main__':
    for im_set in ['data/set1', 'data/set1_subset']:
        # Read in the data
        im1 = imread(im_set+'/image1.jpg')
        im2 = imread(im_set+'/image2.jpg')
        points_im1 = get_data_from_txt_file(im_set + '/pt_2D_1.txt')
        points_im2 = get_data_from_txt_file(im_set + '/pt_2D_2.txt')
        points_3d = get_data_from_txt_file(im_set + '/pt_3D.txt')
        assert (points_im1.shape == points_im2.shape)

        # Run the Factorization Method
        structure, motion = factorization_method(points_im1, points_im2)

        # Plot the structure
        fig = plt.figure()
        ax = fig.add_subplot(121, projection = '3d')
        scatter_3D_axis_equal(structure[0,:], structure[1,:], structure[2,:], ax)
        ax.set_title('Factorization Method')
        ax = fig.add_subplot(122, projection = '3d')
        scatter_3D_axis_equal(points_3d[:,0], points_3d[:,1], points_3d[:,2], ax)
        ax.set_title('Ground Truth')

        plt.show()

        # With 30 points
        # Run the Factorization Method
        k = 30
        structure, motion = factorization_method(points_im1[:k], points_im2[:k])

        # Plot the structure
        fig = plt.figure()
        ax = fig.add_subplot(121, projection='3d')
        scatter_3D_axis_equal(structure[0, :], structure[1, :], structure[2, :], ax)
        ax.set_title(f'Factorization Method With {k}')
        ax = fig.add_subplot(122, projection='3d')
        scatter_3D_axis_equal(points_3d[:k, 0], points_3d[:k, 1], points_3d[:k, 2], ax)
        ax.set_title('Ground Truth')

        plt.show()

        k = 20
        structure, motion = factorization_method(points_im1[:k], points_im2[:k])

        # Plot the structure
        fig = plt.figure()
        ax = fig.add_subplot(121, projection='3d')
        scatter_3D_axis_equal(structure[0, :], structure[1, :], structure[2, :], ax)
        ax.set_title(f'Factorization Method With {k}')
        ax = fig.add_subplot(122, projection='3d')
        scatter_3D_axis_equal(points_3d[:k, 0], points_3d[:k, 1], points_3d[:k, 2], ax)
        ax.set_title('Ground Truth')

        plt.show()

        k = 10
        structure, motion = factorization_method(points_im1[:k], points_im2[:k])

        # Plot the structure
        fig = plt.figure()
        ax = fig.add_subplot(121, projection='3d')
        scatter_3D_axis_equal(structure[0, :], structure[1, :], structure[2, :], ax)
        ax.set_title(f'Factorization Method With {k}')
        ax = fig.add_subplot(122, projection='3d')
        scatter_3D_axis_equal(points_3d[:k, 0], points_3d[:k, 1], points_3d[:k, 2], ax)
        ax.set_title('Ground Truth')

        plt.show()

        k = 16
        structure, motion = factorization_method(points_im1[:k], points_im2[:k])

        # Plot the structure
        fig = plt.figure()
        ax = fig.add_subplot(121, projection='3d')
        scatter_3D_axis_equal(structure[0, :], structure[1, :], structure[2, :], ax)
        ax.set_title(f'Factorization Method With {k}')
        ax = fig.add_subplot(122, projection='3d')
        scatter_3D_axis_equal(points_3d[:k, 0], points_3d[:k, 1], points_3d[:k, 2], ax)
        ax.set_title('Ground Truth')

        plt.show()
