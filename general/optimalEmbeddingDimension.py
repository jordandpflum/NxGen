import numpy as np
import matplotlib.pyplot as plt


def generateOptimalEmbeddingDimension():



    return

def td_embedding(timeseries, emb, tau):
    """
    Time delay embedding of timeseries of scalars

    Parameters:
        timeseries: array of scalars
        emb: (int) embedding dimension
        tau: (int) time delay between values in phase space reconstruction

    Returns:
        array of embedded vectors:
        [x[i], x[i+tau], x[i+2*tau],...,x[i + (m-1)*tau]]
    """
    indexes = np.arange(0, emb, 1) * tau
    return np.array([timeseries[indexes + i] for i in range(len(timeseries) - (emb - 1) * tau)])


def logarithmic_r(min_n,max_n, factor):
    """
    Creates array of values distributed such as log(values) is an array of evenly spaced (space between values =
    log(factor) values between log(min_n) and log (max_n)

    Parameters:
    min_n: minimum value
    max_n: max_n: maximum value ( > arg1 )
    factor: log(factor) is the space between values

    Returns: min_n, min_n * factor, min_n * factor^2, ... min_n * factor^i < max_n
    """
    if max_n <= min_n:
        raise ValueError("arg1 has to be < arg2")
    if factor <= 1:
        raise ValueError("factor(arg3) has to be > 1")
    max_i = int(np.floor(np.log(1.0 * max_n / min_n) / np.log(factor)))
    return np.array([min_n * (factor ** i) for i in range(max_i + 1)])


def grassberg_procaccia(timeseries, emb_dim, time_delay, plot=False):
    """
    Implementation of the Gassberger-Procaccia algorithm to estimate the
    correlation dimension of a set of points in an m-dimensional space.
    This code takes in input a timeseries of scalar values and the embedding dimension + time delay
     necessary to perform a time-delay embedding in phase space to reconstruct the attractor
    Args:
        timeseries: array of scalars
        emb_dim: (int) embedding dimension
        time_delay = (int) time delay between values in phase space reconstruction
    Kwargs:
        plot: if set to True: plots the logarithm of the correlation
        sums against the logarithm of the set of values of r considered in the algorithm
        r is the scaling factor, it tells the threshold distance between points. if we have a plateau
        of local slopes means that we are in a scaling range.
    Returns:
        Correlation dimension (scalar)
    """
    sd_data = np.std(timeseries)
    orbit = td_embedding(timeseries, emb_dim, time_delay)

    n = orbit.shape[0]
    print(n)
    r_vals = logarithmic_r(0.1 * sd_data, 0.7 * sd_data, 1.03)
    distances = np.zeros(shape=(n, n))
    r_matrix_base = np.zeros(shape=(n, n))

    for i in range(n):
        for j in range(i, n):
            distances[i][j] = np.linalg.norm(orbit[i, :] - orbit[j, :])
            r_matrix_base[i][j] = 1

    C_r = []
    for r in r_vals:
        r_matrix = r_matrix_base * r
        heavi_matrix = np.heaviside(r_matrix - distances, 0)
        corr_sum = (2 / float(n * (n - 1))) * np.sum(heavi_matrix)
        C_r.append(corr_sum)

    # strong assumption: the log-log plot is assumed to be a smooth, monotonic function,
    # hence the slope in the scaling region should be the maximum gradient ( in this case
    # is taken as the mean of the last five maximum gradients as they are calculated for every point )

    gradients = np.gradient(np.log(C_r), np.log(r_vals))
    gradients.sort()
    D = np.mean(gradients[-5:])

    if plot:
        fig = plt.figure(figsize=(13, 9))
        ax = fig.add_subplot()
        ax.plot(np.log(r_vals), np.log(C_r), 'blue')
        ax.set_xlabel('log(r)', rotation=0)
        ax.set_ylabel('log(C(r)', rotation=90)
        ax.set_title('Correlation Dimension', loc='center')
        plt.show()

    return D

def grassberg_procaccia_test(system, plot=False):
    """
    Implementation of the Gassberger-Procaccia algorithm to estimate the
    correlation dimension of a set of points in an m-dimensional space.
    This code takes in input a timeseries of scalar values and the embedding dimension + time delay
     necessary to perform a time-delay embedding in phase space to reconstruct the attractor
    Args:
        timeseries: array of scalars
        emb_dim: (int) embedding dimension
        time_delay = (int) time delay between values in phase space reconstruction
    Kwargs:
        plot: if set to True: plots the logarithm of the correlation
        sums against the logarithm of the set of values of r considered in the algorithm
        r is the scaling factor, it tells the threshold distance between points. if we have a plateau
        of local slopes means that we are in a scaling range.
    Returns:
        Correlation dimension (scalar)
    """
    sd_data = np.std(system[:, 0])

    n = system.shape[0]
    r_vals = logarithmic_r(0.1 * sd_data, 0.7 * sd_data, 1.03)
    distances = np.zeros(shape=(n, n))
    r_matrix_base = np.zeros(shape=(n, n))

    for i in range(n):
        for j in range(i, n):
            distances[i][j] = np.linalg.norm(system[i, :] - system[j, :])
            r_matrix_base[i][j] = 1

    C_r = []
    for r in r_vals:
        r_matrix = r_matrix_base * r
        heavi_matrix = np.heaviside(r_matrix - distances, 0)
        corr_sum = (2 / float(n * (n - 1))) * np.sum(heavi_matrix)
        C_r.append(corr_sum)

    # strong assumption: the log-log plot is assumed to be a smooth, monotonic function,
    # hence the slope in the scaling region should be the maximum gradient ( in this case
    # is taken as the mean of the last five maximum gradients as they are calculated for every point )

    gradients = np.gradient(np.log(C_r), np.log(r_vals))
    gradients.sort()
    D = np.mean(gradients[-5:])

    if plot:
        fig = plt.figure(figsize=(13, 9))
        ax = fig.add_subplot()
        ax.plot(np.log(r_vals), np.log(C_r), 'blue')
        ax.set_xlabel('log(r)', rotation=0)
        ax.set_ylabel('log(C(r)', rotation=90)
        ax.set_title('Correlation Dimension', loc='center')
        plt.show()

    return D