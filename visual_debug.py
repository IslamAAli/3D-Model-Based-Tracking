import cv2 as cv
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import config

# visualize 3d points in a scatter plot
def visualize_3d_scatter(m_pts):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    ax.scatter3D(m_pts[:,0], m_pts[:,1], m_pts[:,2], s=100, c='red')

    plt.show()

def visualize_3d_lines(m_edges):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    for edge in m_edges:
        line = []
        # extract end points
        edge_start  = edge[0:3]
        edge_end    = edge[3:6]

        line.append(edge_start)
        line.append(edge_end)
        line = np.asarray(line)

        ax.plot3D(line[:, 0], line[:, 1], line[:, 2], 'blue')

    plt.show()

def visualize_3d_lines_pts(m_edges, m_pts_sampled, m_pts_edges):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    for edge in m_edges:
        line = []
        # extract end points
        edge_start = edge[0:3]
        edge_end = edge[3:6]

        line.append(edge_start)
        line.append(edge_end)
        line = np.asarray(line)

        ax.plot3D(line[:, 0], line[:, 1], line[:, 2], 'blue')

    # plot points
    ax.scatter3D(m_pts_sampled[:, 0], m_pts_sampled[:, 1], m_pts_sampled[:, 2], s=100, c='red')
    ax.scatter3D(m_pts_edges[:, 0], m_pts_edges[:, 1], m_pts_edges[:, 2], s=100, c='green')

    plt.show()

def visualize_2d_pts(pts_2d):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.scatter(pts_2d[:, 0], pts_2d[:, 1], marker='o');
    ax.set_xlim([0, 1024])
    ax.set_ylim([0, 768])
    plt.show()


def visualize_3d_pts_img(pts_2d_edge, pts_2d_ctrl):
    # Create a white image
    img = np.ones((768, 1024, 3), np.float)
    img = img * 255

    # img = cv.imread('dataset_eval/0001.png')

    # plot sedge points
    for point in pts_2d_edge:
        cv.circle(img, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)

    # plot control points
    for point in pts_2d_ctrl:
        cv.circle(img, (int(point[0]), int(point[1])), 5, (255, 0, 0), -1)

    # draw edges between edges
    for i in range(8):
        if (i + 1)%4 ==0:
            cv.line(img,    (int(pts_2d_edge[i,0]),int(pts_2d_edge[i,1])),
                            (int(pts_2d_edge[i-3,0]),int(pts_2d_edge[i-3,1])),
                            (0,255,0),2)
        else:
            cv.line(img,    (int(pts_2d_edge[i, 0]), int(pts_2d_edge[i, 1])),
                            (int(pts_2d_edge[i+1, 0]), int(pts_2d_edge[i+1, 1])),
                            (0, 255, 0), 2)

    for i in range(4):
        cv.line(img, (int(pts_2d_edge[i, 0]), int(pts_2d_edge[i, 1])),
                (int(pts_2d_edge[i +4, 0]), int(pts_2d_edge[i +4, 1])),
                (0, 255, 0), 2)


    cv.imshow('Object Projection', img)
    cv.waitKey(0)


def visualize_2d_pts_img(img, points_1, points_2, both=True):

    img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    for point in points_1:
        cv.circle(img, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)

    if both:
        for point in points_2:
            cv.circle(img, (int(point[0]), int(point[1])), 5, (255, 0, 0), -1)

    cv.imshow('Object Projection', img)
    cv.waitKey(0)
