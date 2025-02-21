# CS231A Homework 1, Problem 3
import numpy as np
from utils import mat2euler
import math

'''
COMPUTE_VANISHING_POINTS
Arguments:
    points - a list of all the points where each row is (x, y). 
            It will contain four points: two for each parallel line.
Returns:
    vanishing_point - the pixel location of the vanishing point
'''
def compute_vanishing_point(points):
    # BEGIN YOUR CODE HERE
    p1, p2, p3, p4 = points

    l1 = np.cross(
        (p1[0], p1[1], 1), # homogenous geometry
        (p2[0], p2[1], 1)
    )

    l2 = np.cross(
        (p3[0], p3[1], 1),
        (p4[0], p4[1], 1)
    )

    vanishing_point = np.cross(l1, l2)
    return (vanishing_point / vanishing_point[-1])[:2]

    # x1 = points[0][0]
    # y1 = points[0][1]
    # x2 = points[1][0]
    # y2 = points[1][1]
    # x3 = points[2][0]
    # y3 = points[2][1]
    # x4 = points[3][0]
    # y4 = points[3][1]
    #
    # # slopes
    # m1 = (float)(y2 - y1) / (x2 - x1)
    # m2 = (float)(y4 - y3) / (x4 - x3)
    # # intercepts
    # b1 = y2 - m1 * x2
    # b2 = y4 - m2 * x4
    #
    # # vanishing point coordinates
    # x = (b2 - b1) / (m1 - m2)
    # y = m1 * ((b2 - b1) / (m1 - m2)) + b1
    # vanishing_point = np.array([x, y])
    # return vanishing_point
    # END YOUR CODE HERE

'''
COMPUTE_K_FROM_VANISHING_POINTS
Makes sure to make it so the bottom right element of K is 1 at the end.
Arguments:
    vanishing_points - a list of vanishing points

Returns:
    K - the intrinsic camera matrix (3x3 matrix)
'''
def compute_K_from_vanishing_points(vanishing_points):
    # BEGIN YOUR CODE HERE
    v1, v2, v3 = vanishing_points
    print(vanishing_points)
    v1 = np.append(v1, 1)
    v2 = np.append(v2, 1)
    v3 = np.append(v3, 1)

    A = np.array([
        # Từ v1'wv2 = 0
        [v1[0] * v2[0] + v1[1] * v2[1],  # hệ số w1
         v2[0] + v1[0],  # hệ số w4
         v2[1] + v1[1],  # hệ số w5
         1],  # hệ số w6

        # Từ v2'wv3 = 0
        [v2[0] * v3[0] + v2[1] * v3[1],
         v3[0] + v2[0],
         v3[1] + v2[1],
         1],

        # Từ v3'wv1 = 0
        [v3[0] * v1[0] + v3[1] * v1[1],
         v1[0] + v3[0],
         v1[1] + v3[1],
         1]
    ])

    _, _, Vh = np.linalg.svd(A)
    x = Vh[-1]

    omega = np.array([
        [x[0], 0, x[1]],
        [0, x[0], x[2]],
        [x[1], x[2], x[3]]
    ])

    KT_inv = np.linalg.cholesky(omega)
    K = np.linalg.inv(KT_inv.T)

    K /= K[2, 2]

    return K
    # END YOUR CODE HERE

'''
COMPUTE_ANGLE_BETWEEN_PLANES
Arguments:
    vanishing_pair1 - a list of a pair of vanishing points computed from lines within the same plane
    vanishing_pair2 - a list of another pair of vanishing points from a different plane than vanishing_pair1
    K - the camera matrix used to take both images

Returns:
    angle - the angle in degrees between the planes which the vanishing point pair comes from2
'''
def compute_angle_between_planes(vanishing_pair1, vanishing_pair2, K):
    # BEGIN YOUR CODE HERE
    v1, v2 = vanishing_pair1
    v3, v4 = vanishing_pair2

    v1 = np.append(v1, 1)
    v2 = np.append(v2, 1)
    v3 = np.append(v3, 1)
    v4 = np.append(v4, 1)

    l1_horiz = np.cross(v1, v2)
    l2_horiz = np.cross(v3, v4)

    n1 = K.T @ l1_horiz
    n2 = K.T @ l2_horiz

    cos_angle = np.dot(n1, n2) / (np.linalg.norm(n1) * np.linalg.norm(n2))
    return np.degrees(np.arccos(cos_angle))
    # END YOUR CODE HERE

'''
COMPUTE_ROTATION_MATRIX_BETWEEN_CAMERAS
Arguments:
    vanishing_points1 - a list of vanishing points in image 1
    vanishing_points2 - a list of vanishing points in image 2
    K - the camera matrix used to take both images

Returns:
    R - the rotation matrix between camera 1 and camera 2
'''
def compute_rotation_matrix_between_cameras(vanishing_points1, vanishing_points2, K):
    # BEGIN YOUR CODE HERE
    vanishing_points1 = np.hstack([vanishing_points1, np.ones((vanishing_points1.shape[0], 1))])
    vanishing_points2 = np.hstack([vanishing_points2, np.ones((vanishing_points2.shape[0], 1))])

    K_inv = np.linalg.inv(K)

    d1 = K_inv @ vanishing_points1
    d2 = K_inv @ vanishing_points2

    H = d2 @ d1.T
    U, _, Vt = np.linalg.svd(H)
    R = U @ Vt

    return R
    # END YOUR CODE HERE

'''
TEST_P3
Test function. Do not modify.
'''
def test_p3():
    # Part A: Compute vanishing points
    v1 = compute_vanishing_point(np.array([[1080, 598],[1840, 478],[1094,1340],[1774,1086]]))
    v2 = compute_vanishing_point(np.array([[674,1826],[4, 878],[2456,1060],[1940,866]]))
    v3 = compute_vanishing_point(np.array([[1094,1340],[1080,598],[1774,1086],[1840,478]]))

    v1b = compute_vanishing_point(np.array([[314,1912],[2060,1040],[750,1378],[1438,1094]]))
    v2b = compute_vanishing_point(np.array([[314,1912],[36,1578],[2060,1040],[1598,882]]))
    v3b = compute_vanishing_point(np.array([[750,1378],[714,614],[1438,1094],[1474,494]]))

    # Part B: Compute the camera matrix
    vanishing_points = [v1, v2, v3]
    print("Intrinsic Matrix:\n",compute_K_from_vanishing_points(vanishing_points))

    K_actual = np.array([[2448.0, 0, 1253.0],[0, 2438.0, 986.0],[0,0,1.0]])
    print()
    print("Actual Matrix:\n", K_actual)

    # Part D: Estimate the angle between the box and floor
    floor_vanishing1 = v1
    floor_vanishing2 = v2
    box_vanishing1 = v3
    box_vanishing2 = compute_vanishing_point(np.array([[1094,1340],[1774,1086],[1080,598],[1840,478]]))
    angle = compute_angle_between_planes([floor_vanishing1, floor_vanishing2], [box_vanishing1, box_vanishing2], K_actual)
    print()
    print("Angle between floor and box:", angle)

    # Part E: Compute the rotation matrix between the two cameras
    rotation_matrix = compute_rotation_matrix_between_cameras(np.array([v1, v2, v3]), np.array([v1b, v2b, v3b]), K_actual)
    print("Rotation between two cameras:\n", rotation_matrix)
    z,y,x = mat2euler(rotation_matrix)
    print()
    print("Angle around z-axis (pointing out of camera): %f degrees" % (z * 180 / math.pi))
    print("Angle around y-axis (pointing vertically): %f degrees" % (y * 180 / math.pi))
    print("Angle around x-axis (pointing horizontally): %f degrees" % (x * 180 / math.pi))


if __name__ == '__main__':
    test_p3()
