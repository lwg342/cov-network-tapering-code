# %%
import numpy as np
from scipy import linalg as LA
from scipy.stats import t
from wlpy.covariance import Covariance
from wlpy.gist import heatmap
# %%


def matrix_elementwise_multiplication(mat1, mat2):
    """
    This function performs elementwise multiplication of two matrices after checking their dimensions.
    :param mat1: A matrix (2D array)
    :param mat2: A matrix (2D array)
    :return: A matrix (2D array) which is the result of elementwise multiplication of two matrices.
    """
    if mat1.shape != mat2.shape:
        raise ValueError("Matrices are not of the same dimensions!")
    result_matrix = np.multiply(mat1, mat2)
    return result_matrix


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
        return np.clip(2 - 2*np.abs(distance_matrix)/bandwidth, 0, 1)
    elif method == "banding":
        return np.where(np.abs(distance_matrix) <= bandwidth, 1, 0)
    else:
        raise NotImplementedError(
            "Available methods are 'linear' and 'banding'.")


def cov_tapering(sample_cov, distance_matrix, bandwidth, method="linear"):
    return matrix_elementwise_multiplication(sample_cov, tapering_weights(distance_matrix, bandwidth, method))


# %%
def replace_diagonal(matrix, value):
    np.fill_diagonal(matrix, value)
    return matrix


def generate_true_cov_cai2010(distance_matrix, rho, alpha):
    pseudo_distance_matrix = replace_diagonal(
        distance_matrix, 1)  # To avoid division by zero
    true_cov = rho*(pseudo_distance_matrix**(- alpha - 1))
    true_cov = replace_diagonal(true_cov, 1)
    return true_cov

# %%    

def symmetrize_using_upper_triangular(matrix, k=1):
    M = np.triu(matrix, k)
    return M + M.T

# %% 
def generate_poisson_discrete_measurement_error(p, lambd=1):
    """
    This function generates a matrix of size n*n with Poisson discrete random variables having a mean of 0.
    :param n: An integer representing the size of the matrix.
    :param lambd: An optional float representing the lambda value of the Poisson distribution. Default is 1.
    :return: A matrix (2D array) of size n*n with Poisson discrete random variables having a mean of 0.
    """
    M = np.random.poisson(lam=lambd, size=(p, p)) - lambd*np.ones((p, p))
    return symmetrize_using_upper_triangular(M)


def generate_rounded_t_matrix(p, df):
    """
    Generate a p by p matrix of t-distributed random variables
    that are rounded to integer values.

    :param p: int, the number of rows and columns of the matrix
    :param df: float, the degrees of freedom parameter for the t-distribution
    :return: numpy.ndarray, the generated matrix of rounded t-distributed random variables
    """
    t_matrix = t.rvs(df=df, size=(p, p))
    rounded_t_matrix = np.round(t_matrix).astype(int)
    return symmetrize_using_upper_triangular(rounded_t_matrix)
# %%


def multivariate_t_rvs(mean, true_cov, sample_size, df):
    '''generate random variables of multivariate t distribution
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
    Returns
    -------
    rvs : ndarray, (n, len(m))
        each row is an independent draw of a multivariate t distributed
        random variable
    '''
    mean = np.asarray(mean)
    d = len(mean)
    if df == np.inf:
        x = np.ones(sample_size)
    else:
        x = np.random.chisquare(df, sample_size) / df
    z = np.random.multivariate_normal(
        np.zeros(d), true_cov, (sample_size,))
    # same output format as random.multivariate_normal
    return mean + z/np.sqrt(x)[:, None]


def generate_normal_samples(true_cov, n, p):
    mean = np.zeros(p)
    samples = np.random.multivariate_normal(mean, true_cov, size=n)
    return samples

# %%
def test_if_positive_definite(matrix, tol=1e-8):
    return np.all(np.linalg.eigvalsh(matrix) > -tol)


def compute_smallest_eigenvalue(matrix):
    return np.linalg.eigvalsh(matrix).min()


def test_if_symmetric(matrix):
    return np.allclose(matrix, matrix.T)

# %%

def correct_eigenvalues(matrix, tols = 1e-8):
    # get the eigenvalues and eigenvectors of the matrix
    eigenvalues, eigenvectors = LA.eigh(matrix)
    # correct the eigenvalues to be positive
    eigenvalues[eigenvalues < 0] = tols
    # reconstruct the matrix
    corrected_matrix = np.dot(eigenvectors, np.dot(
        np.diag(eigenvalues), eigenvectors.T))
    return corrected_matrix
