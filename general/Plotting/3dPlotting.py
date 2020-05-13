import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d  # needed to plot 3D


def generate3Dplot(data, title="", axis_labels=("X", "Y", "Z"), axis_lim = ([-10, 15], [-10, 15], [-10, 15])):
    fig = plt.figure(figsize=(13, 9))
    ax = fig.gca(projection='3d')
    # Set x, y, z lim
    ax.set_xlim(axis_lim[0][0], axis_lim[0][1])
    ax.set_ylim(axis_lim[1][0], axis_lim[1][1])
    ax.set_zlim(axis_lim[2][0], axis_lim[2][1])
    ax.view_init(25, -130)
    ax.plot(data[:, 0], data[:, 1], data[:, 2], 'blue')
    ax.set_xlabel(axis_labels[0], rotation=0)
    ax.set_ylabel(axis_labels[1], rotation=0)
    ax.set_zlabel(axis_labels[2], rotation=180)
    ax.set_title(title, loc='center')
    return plt.show()