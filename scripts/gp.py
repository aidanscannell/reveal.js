import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import norm

jitter = 1e-8  # small number to ensure numerical stability (eigenvalues of K can decay rapidly)


def kernel(x1, x2, var_f=1.0, l=1.0):
    """Squared exponential kernel"""
    x1 = x1.reshape(-1, 1)
    x2 = x2.reshape(-1, 1)
    return var_f * np.exp((-cdist(x1, x2)**2) / (2 * l**2))
    #return var_f * np.exp((-cdist(x1, x2)**2) / (2*l**2)) + 1e-6*np.eye(x1.shape[0])


def sample(mu, var, num_samples=10, jitter=0.0):
    """Generate N samples from a multivariate Gaussian \mathcal{N}(mu, var)"""
    L = np.linalg.cholesky(
        var + jitter * np.eye(var.shape[0])
    )  # cholesky decomposition (square root) of covariance matrix
    if type(mu) is int:
        mu = np.array(mu)
    mu = mu.reshape(-1, 1)
    f_post = mu + L @ np.random.normal(size=(L.shape[1], num_samples))
    return f_post


def gp_regression(X, y, k, x_star, s_f=1.0, l=1.0):
    # calculate mean
    K = k(X, X, s_f, l)
    Lxx = np.linalg.cholesky(K + jitter * np.eye(*K.shape))
    a = np.linalg.solve(Lxx.T, np.linalg.solve(Lxx, y))
    mu = k(X, x_star, s_f, l).T @ a

    # calculate variance
    v = np.linalg.solve(Lxx, k(X, x_star, s_f, l))
    var = k(x_star, x_star, s_f, l) - v.T @ v

    return mu, var


def gp_regression_noisy(X, y, k, x_star, var_f, var_n, l):
    # calculate mean
    Lxx = np.linalg.cholesky(k(X, X, var_f, l) + var_n * np.eye(X.shape[0]))
    a = np.linalg.solve(Lxx.T, np.linalg.solve(Lxx, y))
    mu = k(X, x_star, var_f, l).T @ a

    # calculate variance
    v = np.linalg.solve(Lxx, k(X, x_star, var_f, l))
    var = k(x_star, x_star, var_f, l) - v.T @ v

    return mu, var


def plot_marginal(ind, mu, std):
    fig = plt.figure(figsize=(18, 6))
    grid = plt.GridSpec(2, 3, wspace=0.4, hspace=0.3)
    main_ax = fig.add_subplot(grid[:, 1:])
    y_ax = fig.add_subplot(grid[:, 0], sharey=main_ax)
    main_ax.plot(x_star, mu, 'c-', lw=3)
    if x_train is not None:
        main_ax.plot(x_train, y_train, 'ko', ms=15)
    y_star = np.linspace(-4., 4., 100)
    main_ax.fill_between(x_star.flatten(),
                         mu.flatten() - 2 * std,
                         mu.flatten() + 2 * std,
                         color="steelblue",
                         alpha=0.3,
                         lw=2,
                         zorder=10)
    xx = -norm(mu, std.reshape(-1, 1)).pdf(y_star) + x_star
    xx = xx.T
    main_ax.plot(np.ones_like(y_star) * x_star[ind], y_star, 'r-')
    y_ax.plot(xx[:, ind], y_star, 'r-', lw=3)
    y_ax.axis('off')
    main_ax.axis('off')


def expected_value_monte_carlo(func, mu, Sigma_x, n=100000000000):
    """
    :param func: a function that expects an 1D np array
    :param mu: the mean of a multivariate normal
    :param Sigma_x: the cov of a multivariate nromal
    :param n: the number of samples to use
    :return: The expected value of func(x) * p_mvnorm(x|mu,Sigma_x)
    """
    from numpy.random import normal
    vfunc = lambda x: list(map(func, x))  #np.vectorize(func)
    exp_val = np.mean(vfunc(normal(mu, Sigma_x, n)))
    return exp_val


def propagate_mean(u, Sigma_x, n):
    func = lambda x: (gp_regression(x_train, y_train, kernel, x))[
        0]  #* mvnorm(x, u, Sigma_x)
    return expected_value_monte_carlo(func, u, Sigma_x, n)


def propagate_GA(u, Sigma_x, n):
    mu = propagate_mean(u, Sigma_x, n)
    func1 = lambda x: (gp_regression(x_train, y_train, kernel, x))[
        1]  #* mvnorm(x, u, Sigma_x)
    func2 = lambda x: (gp_regression(x_train, y_train, kernel, x))[
        0]**2  #* mvnorm(x, u, Sigma_x)
    #variance = integrate(func1, bounds) + integrate(func2, bounds) - mu ** 2
    variance = expected_value_monte_carlo(func1, u, Sigma_x, n) \
               + expected_value_monte_carlo(func2, u, Sigma_x, n) - mu**2
    return mu, variance
