import matplotlib
matplotlib.use("TkAgg")
import numpy as np
import matplotlib.pyplot as plt


def plot_vectors(vecs, cols, alpha=1):
    plt.axvline(x=0, ymin=0, ymax=1)
    plt.axhline(y=0, color='#A9A9A9', zorder=0)
    for i in range(len(vecs)):
        print(i)
        if isinstance(alpha, list):
            alpha_i = alpha[i]
        else:
            alpha_i = alpha


        x = np.concatenate([[0, 0], vecs[i]])
        # print(type([x[0]]), [x[1]], [x[2]], [x[3]])
        plt.quiver([x[0]], [x[1]], [x[2]], [x[3]], angles='xy', scale_units='xy', scale=1, color=cols[i], alpha=alpha_i)


u = np.array([2, 2])
v = np.array([0, 3])
orange = '#FF9A13'
blue = '#1190FF'
plot_vectors([u, v], cols=[orange, blue])
plt.show()


