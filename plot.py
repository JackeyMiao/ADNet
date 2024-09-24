import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Circle

def display_points(points: np.ndarray) -> None:
    """
    Display a set of 2D points on a scatterplot.

    Args:

    Returns:
        object:
    points: x,y coordinate points.
    """

    y_offset = 0.025
    plt.scatter(points[:, 0], points[:, 1], color='b')
    for i, point in enumerate(points):
        plt.text(point[0], point[1] + y_offset, s=None)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(False)


def display_points_with_pm(points, centers):
    display_points(points)
    N = points.size()[0]
    dist = (points[:, None, :] - points[None, :, :]).norm(p=2, dim=-1)
    dist_p = torch.index_select(dist, 1, centers)
    index = torch.argmin(dist_p, 1)
    for i in range(N):
        p1 = points[i]
        p2 = points[centers[index[i]]]
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color='orange')
    for i in centers:
        plt.scatter(points[i, 0], points[i, 1], color='r')


def display_points_with_pmedian(points, solution, center):
    display_points(points)
    # cs = ["orange", "green", 'steelblue']
    centers = np.argwhere(center == 1).flatten()
    N = points.size()[0]
    for i in centers:
        for j in range(N):
            if solution[i,j] == 1:
                p1 = points[i]
                p2 = points[j]
                plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color='orange')
        plt.scatter(points[i, 0], points[i, 1], color='r')


def display_points_with_mclp(data, ax, solution, radius):
    ax = plt.gca()
    plt.scatter(data[:, 0], data[:, 1], c='red')
    for i in solution:
        plt.scatter(data[i][0], data[i][1], c='blue', marker='+')
        circle = Circle(xy=(data[i][0], data[i][1]), radius=radius, color='b', fill=False, lw=1)
        ax.add_artist(circle)

def display_points_with_p_center(points, solution, center):
    ax = plt.gca()
    plt.scatter(points[:, 0], points[:, 1], c='blue')
    # cs = ["orange", "green", 'steelblue']
    for i in center:
        plt.scatter(points[i][0], points[i][1], c='r', marker='+')
        circle = Circle(xy=(points[i][0], points[i][1]), radius=solution, color='orange', fill=False, lw=1)
        ax.add_artist(circle)


def display_points_with_pc(points, solution, center):
    ax = plt.gca()
    plt.scatter(points[:, 0], points[:, 1], c='blue')
    centers = np.argwhere(center == 1).flatten()
    for i in centers:
        plt.scatter(points[i][0], points[i][1], c='r', marker='+')
        circle = Circle(xy=(points[i][0], points[i][1]), radius=solution, color='orange', fill=False, lw=1)
        ax.add_artist(circle)

