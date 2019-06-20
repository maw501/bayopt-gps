import matplotlib.pyplot as plt
from pylab import grid
import numpy as np

def plot_convergence(Xdata, best_Y):
    '''Plots to evaluate the convergence of standard Bayesian optimization algorithms
    Changed from: https://github.com/SheffieldML/GPyOpt/blob/master/GPyOpt/plotting/plots_bo.py
    '''
    n = Xdata.shape[0]
    aux = (Xdata[1:n, :] - Xdata[0:n - 1, :]) ** 2
    distances = np.sqrt(aux.sum(axis=1))

    # Distances between consecutive x's
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(list(range(n - 1)), distances, '-ro')
    plt.xlabel('Iteration')
    plt.ylabel('d(x[n], x[n-1])')
    plt.title('Distance between consecutive X\'s')
    grid(True)

    # Estimated m(x) at the proposed sampling points
    plt.subplot(1, 2, 2)
    plt.plot(list(range(n)), best_Y, '-o')
    plt.title('Value of the best selected sample')
    plt.xlabel('Iteration')
    plt.ylabel('Best y')
    grid(True)

    plt.subplots_adjust(wspace=0.25)
    plt.show()