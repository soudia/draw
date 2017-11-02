""" Stein Variational Gradient Descent
    Reference paper: https://arxiv.org/pdf/1608.04471.pdf
"""


import numpy as np
from scipy.spatial.distance import pdist, squareform


# pylint: disable=W0622,C0325,E1101,C0103
class SVGD(object):
    """ Stein Variational Gradient Descent """
    def __init__(self):
        pass

    def svgd_kernel(self, theta, h=-1):
        """ Stein Variational Gradient Descent Kernel
            Arg:
                theta: mean-centered latent code samples (subtract mean)
                h: Radial Basis Kernel term (h) value (in this case it's
                   the standard deviation)
        """
        sq_dist = pdist(theta)
        pairwise_dists = squareform(sq_dist)**2  # pairwise distance in matrix form
        if h < 0:  # if h < 0, using median trick
            h = np.median(pairwise_dists)
            h = np.sqrt(0.5 * h / np.log(theta.shape[0]+1))

        # Compute the rbf kernel: exp(-||X_i - X_j|| / (2 * h^2)) for all i, j
        Kxy = np.exp(-pairwise_dists / h**2 / 2)

        dxkxy = -np.matmul(Kxy, theta)  # derivative of the kernel Kxy with respect to x
        sumkxy = np.sum(Kxy, axis=1)

        # Check the paper for more context https://arxiv.org/pdf/1608.04471.pdf Delta log(p)
        for i in range(theta.shape[1]):
            dxkxy[:, i] = dxkxy[:, i] + np.multiply(theta[:, i], sumkxy)
        dxkxy = dxkxy / (h**2)
        return (Kxy, dxkxy)

    def update(self, x0, lnprob, n_iter=1000, stepsize=1e-3, bandwidth=-1, alpha=0.9, debug=False):
        # Check input
        if x0 is None or lnprob is None:
            raise ValueError('x0 or lnprob cannot be None!')

        theta = np.copy(x0)

        # TODO: Try Adam here using tensorflow

        # TODO: Mean-center theta, and compute the mean

        # TODO: When updating theta, make sure to re-add the mean. Technically, theta as it is should
        # but we never know.

        # adagrad with momentum
        fudge_factor = 1e-6
        historical_grad = 0
        for iter in range(n_iter):
            if debug and (iter+1) % 1000 == 0:
                print('iter ' + str(iter+1))

            lnpgrad = lnprob(theta)
            # calculating the kernel matrix
            kxy, dxkxy = self.svgd_kernel(theta, h=-1)
            grad_theta = (np.matmul(kxy, lnpgrad) + dxkxy) / x0.shape[0]

            # adagrad (this in fact is adadelta an extension of adagrad)
            if iter == 0:
                historical_grad = historical_grad + grad_theta ** 2
            else:
                historical_grad = alpha * historical_grad + (1 - alpha) * (grad_theta ** 2)
            adj_grad = np.divide(grad_theta, fudge_factor+np.sqrt(historical_grad))
            theta = theta + stepsize * adj_grad
        return theta
