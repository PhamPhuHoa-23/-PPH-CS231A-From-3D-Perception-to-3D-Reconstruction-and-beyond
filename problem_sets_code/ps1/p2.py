# CS231A Homework 1, Problem 2
import numpy as np

'''
DATA FORMAT

In this problem, we provide and load the data for you. Recall that in the original
problem statement, there exists a grid of black squares on a white background. We
know how these black squares are setup, and thus can determine the locations of
specific points on the grid (namely the corners). We also have images taken of the
grid at a front image (where Z = 0) and a back image (where Z = 150). The data we
load for you consists of three parts: real_XY, front_image, and back_image. For a
corner (0,0), we may see it at the (137, 44) pixel in the front image and the
(148, 22) pixel in the back image. Thus, one row of real_XY will contain the numpy
array [0, 0], corresponding to the real XY location (0, 0). The matching row in
front_image will contain [137, 44] and the matching row in back_image will contain
[148, 22]
'''

'''
COMPUTE_CAMERA_MATRIX
Arguments:
     real_XY - Each row corresponds to an actual point on the 2D plane
     front_image - Each row is the pixel location in the front image where Z=0
     back_image - Each row is the pixel location in the back image where Z=150
Returns:
    camera_matrix - The calibrated camera matrix (3x4 matrix)
'''
def compute_camera_matrix(real_XY, front_image, back_image):
    # Hint: reshape your values such that you have PM=p,
    # and use np.linalg.lstsq or np.linalg.pinv to solve for M.
    # See https://apimirror.com/numpy~1.11/generated/numpy.linalg.pinv
    #
    # Our solution has shapes for M=(8,), P=(48,8), and p=(48,)
    # Alternatively, you can set things up such that M=(4,2), P=(24,4), and p=(24,2)
    # Lastly, reshape and add the (0,0,0,1) row to M to have it be (3,4)

    # BEGIN YOUR CODE HERE
    num_points = real_XY.shape[0]

    P = np.zeros((4*num_points, 8))
    p = np.zeros(4*num_points)

    for i in range(num_points):
        X, Y = real_XY[i]
        x, y = front_image[i]

        P[4*i] = [X, Y, 0, 1, 0, 0, 0, 0]
        p[4*i] = x

        P[4*i+1] = [0, 0, 0, 0, X, Y, 0, 1]
        p[4*i+1] = y

    for i in range(num_points):
        X, Y = real_XY[i]
        x, y = back_image[i]

        P[4*i+2] = [X, Y, 150, 1, 0, 0, 0, 0]
        p[4*i+2] = x

        P[4*i+3] = [0, 0, 0, 0, X, Y, 150, 1]
        p[4*i+3] = y

    M, residual, rank, s = np.linalg.lstsq(P, p)

    camera_matrix = np.zeros((3,4))
    camera_matrix[0] = M[:4]
    camera_matrix[1] = M[4:]
    camera_matrix[2] = [0, 0, 0, 1]

    return np.asarray(camera_matrix)
    # END YOUR CODE HERE

'''
RMS_ERROR
Arguments:
     camera_matrix - The camera matrix of the calibrated camera
     real_XY - Each row corresponds to an actual point on the 2D plane
     front_image - Each row is the pixel location in the front image where Z=0
     back_image - Each row is the pixel location in the back image where Z=150
Returns:
    rms_error - The root mean square error of reprojecting the points back
                into the images
'''
def rms_error(camera_matrix, real_XY, front_image, back_image):
    # BEGIN YOUR CODE HERE
    # (3,4) x (4,xN) = (3,2N)

    front_XY = np.vstack((
        real_XY.T,
        np.zeros((1, real_XY.shape[0])),
        np.ones((1, real_XY.shape[0]))
    ))

    back_XY = np.vstack((
        real_XY.T,
        np.full((1, real_XY.shape[0]), 150),
        np.ones((1, real_XY.shape[0]))
    ))

    threeD_XY = np.concatenate((front_XY, back_XY), axis=1) # (4,2N)
    pixel_calibration = camera_matrix @ threeD_XY
    pixel = np.concatenate((front_image.T, back_image.T), axis=1)

    squared_diff = ((pixel - pixel_calibration[0:2, :]) ** 2)
    total_squared_error = squared_diff.sum()
    N = real_XY.shape[0] * 2
    rms = np.sqrt(total_squared_error / N)

    return rms
    # END YOUR CODE HERE

'''
TEST_P2
Test function. Do not modify.
'''
def test_p2():
    # Loading the example coordinates setup
    real_XY = np.load('real_XY.npy')
    front_image = np.load('front_image.npy')
    back_image = np.load('back_image.npy')

    camera_matrix = compute_camera_matrix(real_XY, front_image, back_image)
    print("Camera Matrix:\n", camera_matrix)
    print()
    print("RMS Error: ", rms_error(camera_matrix, real_XY, front_image, back_image))


if __name__ == '__main__':
    test_p2()