import matplotlib.pyplot as plt
import numpy as np


def plot_decision_boundary(pred_func, X, y):
    """ Plots contour graph """
    plt.figure(1)
    plt.subplot(212)
    # Set min and max values and give it some padding
    x0_min, x0_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    x1_min, x1_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx0, xx1 = np.meshgrid(np.arange(x0_min, x0_max, h),
                           np.arange(x1_min, x1_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx0.ravel(), xx1.ravel()])
    Z = Z.reshape(xx0.shape)
    # Plot the contour and training examples
    plt.contourf(xx0, xx1, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)


def plot_training_examples(X, y):
    plt.figure(1)
    plt.subplot(211)
    plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)


def plot_show():
    plt.figure(1)
    plt.show()
    print "finished showing"
