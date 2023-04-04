# %%
import numpy as np
from scipy import linalg as LA
from scipy.stats import t

# %%


def get_bandwidth(sample_size, p, method, alpha):
    # p is the number of features
    if method == "tapering":
        return np.floor(sample_size ** (1 / (2 * alpha + 1)))
    elif method == "banding":
        return np.floor((sample_size / np.log(p)) ** (1 / (2 * alpha + 2)))
    elif method == "tapering_undersmoothing":
        return np.floor(sample_size ** (1 / 2))
    else:
        raise ValueError("Method must be either tapering or banding")


def tapering_weights(distance_matrix, bandwidth, method="linear"):
    """
    Construct the tapering weights based on distance d and bandwidth K.

    Methods:
    "linear" :
        1 if |d| <= K/2,
        2 - 2*|d|/K, if K/2 < |d| <= K,
        0 otherwise
    "banding" :
        1 if |d| <= K, 0 otherwise
    """
    if method == "linear":
        return np.clip(2 - 2 * np.abs(distance_matrix) / bandwidth, 0, 1)
    elif method == "banding":
        return np.where(np.abs(distance_matrix) <= bandwidth, 1, 0)
    else:
        raise NotImplementedError("Available methods are 'linear' and 'banding'.")


def cov_tapering(sample_cov, distance_matrix, bandwidth, method="linear", **kwargs):
    return sample_cov * tapering_weights(distance_matrix, bandwidth, method)


def replace_diagonal(matrix, value):
    np.fill_diagonal(matrix, value)
    return matrix


def generate_true_cov_cai2010(distance_matrix, rho, alpha):
    pseudo_distance_matrix = replace_diagonal(
        distance_matrix, 1
    )  # To avoid division by zero
    true_cov = rho * (pseudo_distance_matrix ** (-alpha - 1))
    true_cov = replace_diagonal(true_cov, 1)
    return true_cov


# %%


def symmetrize_using_upper_triangular(matrix, k=1):
    M = np.triu(matrix, k)
    return M + M.T


def generate_poisson_discrete_measurement_error(p, lambd=1, seed=None):
    """
    This function generates a matrix of size n*n with Poisson discrete random variables having a mean of 0.
    :param n: An integer representing the size of the matrix.
    :param lambd: An optional float representing the lambda value of the Poisson distribution. Default is 1.
    :param seed: An optional integer representing the random seed to use for generating the Poisson distribution.
    :return: A matrix (2D array) of size n*n with Poisson discrete random variables having a mean of 0.
    """
    if seed is not None:
        np.random.seed(seed)
    M = np.random.poisson(lam=lambd, size=(p, p)) - lambd * np.ones((p, p))
    return symmetrize_using_upper_triangular(M)


def generate_normal_discrete_measurement_error(p, mu=0, sigma=1, seed=None):
    """
    This function generates a matrix of size n*n with normal discrete random variables having a mean of 0.
    :param n: An integer representing the size of the matrix.
    :param mu: An optional float representing the mean value of the normal distribution. Default is 0.
    :param sigma: An optional float representing the standard deviation of the normal distribution. Default is 1.
    :param seed: An optional integer representing the random seed to use for generating the normal distribution.
    :return: A matrix (2D array) of size n*n with normal discrete random variables having a mean of 0.
    """
    if seed is not None:
        np.random.seed(seed)
    M = np.round(np.random.normal(mu, sigma, size=(p, p))).astype(int)
    return symmetrize_using_upper_triangular(M)


def generate_rounded_t_measurement_error(p, df, seed=None):
    """
    Generate a p by p matrix of t-distributed random variables
    that are rounded to integer values.

    :param p: int, the number of rows and columns of the matrix
    :param df: float, the degrees of freedom parameter for the t-distribution
    :param seed: An optional integer representing the random seed to use for generating the t-distribution.
    :return: numpy.ndarray, the generated matrix of rounded t-distributed random variables
    """
    if seed is not None:
        np.random.seed(seed)
    t_matrix = t.rvs(df=df, size=(p, p))
    rounded_t_matrix = np.round(t_matrix).astype(int)
    return symmetrize_using_upper_triangular(rounded_t_matrix)


def generate_multivariate_t_samples(mean, true_cov, sample_size, df, seed=None):
    """generate random variables of multivariate t distribution
    Parameters
    ----------
    m : array_like
        mean of random variable, length determines dimension of random variable
    S : array_like
        square array of covariance  matrix
    df : int or float
        degrees of freedom
    n : int
        number of observations, return random array will be (n, len(m))
    seed: An optional integer representing the random seed to use for generating the multivariate t distribution.
    Returns
    -------
    rvs : ndarray, (n, len(m))
        each row is an independent draw of a multivariate t distributed
        random variable
    """
    mean = np.asarray(mean)
    d = len(mean)
    if seed is not None:
        np.random.seed(seed)
    if df == np.inf:
        x = np.ones(sample_size)
    else:
        x = np.random.chisquare(df, sample_size) / df
    z = np.random.multivariate_normal(np.zeros(d), true_cov, (sample_size,))
    # same output format as random.multivariate_normal
    return mean + z / np.sqrt(x)[:, None]


def generate_normal_samples(true_cov, n, p, seed=None):
    mean = np.zeros(p)
    if seed is not None:
        np.random.seed(seed)
    samples = np.random.multivariate_normal(mean, true_cov, size=n)
    return samples


def test_if_positive_definite(matrix, tol=1e-8):
    return np.all(np.linalg.eigvalsh(matrix) > -tol)


def compute_smallest_eigenvalue(matrix):
    return np.linalg.eigvalsh(matrix).min()


def test_if_symmetric(matrix):
    return np.allclose(matrix, matrix.T)


def correct_eigenvalues(matrix, tols=1e-8):
    # get the eigenvalues and eigenvectors of the matrix
    eigenvalues, eigenvectors = LA.eigh(matrix)
    # correct the eigenvalues to be positive
    eigenvalues[eigenvalues < 0] = tols
    # reconstruct the matrix
    corrected_matrix = np.dot(
        eigenvectors, np.dot(np.diag(eigenvalues), eigenvectors.T)
    )
    return corrected_matrix


def compute_norm_inverse_difference(matrix1, matrix2, ord=2):
    return LA.norm(LA.inv(matrix1) - LA.inv(matrix2), ord)


def list_to_dict(lst, keys):
    """
    Converts a list of values to a dictionary using a list of keys.

    Parameters:
        lst (list): The list of values to convert to a dictionary.
        keys (list): The list of keys to use for the dictionary. Must have the same length as lst.

    Returns:
        dict: The dictionary with keys from the keys list and values from the lst list.
    """
    if len(lst) != len(keys):
        raise ValueError("The length of the list and keys must be the same.")

    return dict(zip(keys, lst))


# %%
