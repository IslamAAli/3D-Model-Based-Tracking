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