# %% 
import numpy as np
from scipy import linalg as LA
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

class TaperingCovariance(Covariance):
    def __init__(self, sample, observed_distance_matrix=None):
        super().__init__(DF_TxN=sample)

        if observed_distance_matrix == None:
            raise ValueError("Please provide the observed distance matrix")
        elif observed_distance_matrix.shape != (self.N, self.N):
            raise ValueError(
                "The shape of the observed distance matrix is not correct. ")
        else:
            self.distance_matrix = observed_distance_matrix

def linear_tapering(eval, bandwidth):
    return np.clip(2 - 2*np.abs(eval)/bandwidth, 0, 1)

def banding_weights(eval, bandwidth):
    return np.where(np.abs(eval) <= bandwidth, 0, 1)

def construct_tapering_weights(distance_matrix, bandwidth, tapering_function =linear_tapering):
    return tapering_function(distance_matrix, bandwidth)


def cov_tapering(cov, distance_matrix, bandwidth, function=linear_tapering):
    """
    This function performs tapering of the covariance matrix using a distance matrix and a bandwidth.
    :param cov: A covariance matrix (2D array)
    :param distance_matrix: A distance matrix (2D array)
    :param bandwidth: A bandwidth (float)
    :param function: A tapering function (function)
    :return: A covariance matrix (2D array) which is the result of tapering.
    """
    tapering_weights = construct_tapering_weights(
        distance_matrix, bandwidth, function)
    return matrix_elementwise_multiplication(cov, tapering_weights)






# %% 
def replace_diagonal(matrix, value):
    np.fill_diagonal(matrix, value)
    return matrix

def generate_true_cov_cai2010(distance_matrix, rho, alpha):
    pseudo_distance_matrix = replace_diagonal(distance_matrix, 1)
    true_cov = rho*(pseudo_distance_matrix**(- alpha - 1)) 
    true_cov = replace_diagonal(true_cov, 1)
    return true_cov

# %%
def generate_poisson_discrete_random_variables(n, lambd=1):
    """
    This function generates a matrix of size n*n with Poisson discrete random variables having a mean of 0.
    :param n: An integer representing the size of the matrix.
    :param lambd: An optional float representing the lambda value of the Poisson distribution. Default is 1.
    :return: A matrix (2D array) of size n*n with Poisson discrete random variables having a mean of 0.
    """
    return np.random.poisson(lam=lambd, size=(n, n)) - lambd*np.ones((n, n))
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
    """
    This function generates n random samples of normal distributed random variables of dimension p,
    with true_cov as the covariance matrix.
    :param true_cov: A p*p matrix (2D array) representing the true covariance matrix.
    :param n: An integer representing the number of random samples to generate.
    :param p: An integer representing the dimension of the normal distributed random variables.
    :return: A n*p matrix (2D array) representing the generated random samples.
    """
    mean = np.zeros(p)
    samples = np.random.multivariate_normal(mean, true_cov, size=n)
    return samples
