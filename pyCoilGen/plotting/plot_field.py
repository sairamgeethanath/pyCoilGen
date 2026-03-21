import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_field(Bx, By, Bz, points):
    fig = plt.figure(figsize=(15,5))
    for i, (comp, title) in enumerate(zip([Bx, By, Bz], ["Bx","By","Bz"])):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        sc = ax.scatter(points[:,0], points[:,1], points[:,2], c=comp, cmap='jet')
        ax.set_title(title)
        fig.colorbar(sc, ax=ax, shrink=0.6)
        fig.suptitle("Simulated Gradient Field")
        plt.tight_layout()
    plt.show()