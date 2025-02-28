import numpy as np
import matplotlib.pyplot as plt
from p1 import *
from epipolar_utils import *

'''
COMPUTE_EPIPOLE computes the epipole e in homogenous coordinates
given the fundamental matrix
Arguments:
    F - the Fundamental matrix solved for with normalized_eight_point_alg(points1, points2)

Returns:
    epipole - the homogenous coordinates [x y 1] of the epipole in the image
'''
def compute_epipole(F):
    U, S, Vt = np.linalg.svd(F.T)
    e = Vt[-1]
    e = e / e[-1]
    return e
'''
COMPUTE_H computes a homography to map an epipole to infinity along the horizontal axis 
Arguments:
    e - the epipole
    im2 - the image
Returns:
    H - homography matrix
'''
def compute_H(e, im):
    h, w = im.shape[:2]
    # e = e / e[-1]

    T = np.array([
        [1, 0, -w/2],
        [0, 1, -h/2],
        [0, 0, 1]
    ])

    e_t = T @ e

    e_norm = np.sqrt(e_t[0]**2 + e_t[1]**2)

    # alpha = np.arctan2(e[1], e[0])
    R = np.array([
        [e_t[0] / e_norm, e_t[1] / e_norm, 0],
        [-e_t[1] / e_norm, e_t[0] / e_norm, 0],  # Note the negative signs here
        [0, 0, 1]
    ])

    e_r = R @ e_t
    print(e_t, "Hoi cham")
    print(e_r, "Cuu voi")
    f = e_r[0]
    if abs(f) < 1e-6:
        f = 1e-6 * (1 if f >= 0 else -1)

    G = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [-1/f, 0, 1]
    ])

    return np.linalg.inv(T) @ G @ R @ T
'''
COMPUTE_MATCHING_HOMOGRAPHIES determines homographies H1 and H2 such that they
rectify a pair of images
Arguments:
    e2 - the second epipole
    F - the Fundamental matrix
    im2 - the second image
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1
Returns:
    H1 - the homography associated with the first image
    H2 - the homography associated with the second image
'''
def compute_matching_homographies(e2, F, im2, points1, points2):
    e_x = np.array([
        [0, -e2[2], e2[1]],
        [e2[2], 0, -e2[0]],
        [-e2[1], e2[0], 0]
    ])

    M = e_x @ F + np.reshape(e2, (3,1)) @ np.ones((1,3))

    H2 = compute_H(e2, im2)

    points1_hat = (H2 @ M @ points1.T).T
    points1_hat = points1_hat / points1_hat[:,[-1]]

    points2_hat = (H2 @ points2.T).T
    points2_hat = points2_hat / points2_hat[:,[-1]]

    a = np.linalg.lstsq(points1_hat, points2_hat[:,[0]])[0]

    HA = np.array([
        a.ravel(),
        [0, 1, 0],
        [0, 0, 1]
    ])

    H1 = HA @ H2 @ M

    return H1, H2

if __name__ == '__main__':
    # Read in the data
    im_set = 'data/set1'
    im1 = imread(im_set+'/image1.jpg')
    im2 = imread(im_set+'/image2.jpg')
    points1 = get_data_from_txt_file(im_set+'/pt_2D_1.txt')
    points2 = get_data_from_txt_file(im_set+'/pt_2D_2.txt')
    assert (points1.shape == points2.shape)

    F = normalized_eight_point_alg(points1, points2)
    # F is such that such that (points2)^T * F * points1 = 0, so e1 is e' and e2 is e
    e1 = compute_epipole(F.T)
    e2 = compute_epipole(F)
    print("e1", e1)
    print("e2", e2)

    # Find the homographies needed to rectify the pair of images
    H1, H2 = compute_matching_homographies(e2, F, im2, points1, points2)
    print('')

    # Transforming the images by the homographies
    new_points1 = H1.dot(points1.T)
    new_points2 = H2.dot(points2.T)
    new_points1 /= new_points1[2,:]
    new_points2 /= new_points2[2,:]
    new_points1 = new_points1.T
    new_points2 = new_points2.T
    rectified_im1, offset1 = compute_rectified_image(im1, H1)
    rectified_im2, offset2 = compute_rectified_image(im2, H2)
    new_points1 -= offset1 + (0,)
    new_points2 -= offset2 + (0,)

    # Plotting the image
    F_new = normalized_eight_point_alg(new_points1, new_points2)
    plot_epipolar_lines_on_images(new_points1, new_points2, rectified_im1, rectified_im2, F_new)
    plt.show()
