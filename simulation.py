# %% 
import pandas as pd
import numpy as np
from POET.poet import POET
from utils import *
from wlpy.covariance import Covariance
# %% 
# nsim = 1
# sample_size_list = [100, 200, 500, 1000]
# p_list = [100, 200, 500, 1000]
# alpha_list = [0.1, 0.25, 0.5, 0.75, 0.9]
# lambd_list = [0.0, 1e1, 1e2, 1e3, 1e4]
# measurement_error_mean = [0.0, 1e1, 1e2, -1e1, -1e2]
# rho = 0.6
nsim = 1
sample_size_list = [100, 200]
p_list = [100, 200]
alpha_list = [0.1,  0.75]
lambd_list = [0.0, 1e1]
measurement_error_mean = [0.0, 1e1, -1e1]
norm_list = ["fro", 2]
rho = 0.6
# %%


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

def get_covariance_estimators(samples, true_cov, observed_distance_matrix, distance_matrix, tapering_bandwidth, banding_bandwidth, tapering_bandwidth_undersmooth,  n, p, alpha, lambd, bias, norm):
    cov_model = Covariance(samples)
    sample_cov = cov_model.sample_cov()

    taper_cov = cov_tapering(sample_cov, observed_distance_matrix,
                             bandwidth=tapering_bandwidth, method="linear")
    banding_cov = cov_tapering(
        sample_cov, observed_distance_matrix, bandwidth=banding_bandwidth, method="banding")

    estimators = {
        "Sample Covariance": cov_model.sample_cov(),
        "Just Diagonal": np.diag(np.diag(sample_cov)),
        # "Thresholding": POET(samples.T, K=0, C=0.5).SigmaU,
        # "Network Banding": banding_cov,
        # "Linear Shrinkage": cov_model.lw_lin_shrink(),
        # "Network Tapering": taper_cov,
        # "Network Tapering Undersmoothing": cov_tapering(sample_cov, observed_distance_matrix, bandwidth=tapering_bandwidth_undersmooth, method="linear"),
        # "Network Tapering Corrected": correct_eigenvalues(taper_cov),
        # "Oracle Network Banding": cov_tapering(sample_cov, distance_matrix, bandwidth=banding_bandwidth, method="banding"),
        # "Oracle Network Tapering": cov_tapering(sample_cov, distance_matrix, bandwidth=tapering_bandwidth, method="linear"),
        # "Oracle Network Tapering Undersmoothing": cov_tapering(sample_cov, distance_matrix, bandwidth=tapering_bandwidth_undersmooth, method="linear"),
        # "Network Tapering Undersmoothing Alt": cov_tapering(sample_cov, observed_distance_matrix, bandwidth=np.floor((n/np.log(p))**0.5), method="linear"),
    }

 
   
    keys = ["Name", "Norm Type", "Error", "Sample Size", "Feature Dimension", "\\alpha", "\\lambda", "b"]
    
    res_true = list_to_dict(["True Covariance", norm,f"{LA.norm(true_cov, norm): .2f}", n,p, alpha, lambd, bias], keys)
    
    res_estimators = [list_to_dict([name, norm, f"{LA.norm(true_cov - estimator, norm): .2f}", n, p, alpha, lambd, bias], keys)
                        for name, estimator in estimators.items()]
    res_estimators.append(res_true)
    return res_estimators


results = []
for p in p_list:
    index = np.arange(0, p, 1)
    distance_matrix = np.abs(np.subtract.outer(index, index))
    true_cov = generate_true_cov_cai2010(distance_matrix, rho, alpha)
    for n in sample_size_list:
        samples = generate_normal_samples(true_cov, n, p)
        for lambd in lambd_list:
            for bias in measurement_error_mean:
                measurement_error = 1 * \
                    generate_poisson_discrete_measurement_error(
                        p, lambd=lambd) + bias
                observed_distance_matrix = distance_matrix + measurement_error
                for alpha in alpha_list:
                    tapering_bandwidth = get_bandwidth(n, p, "tapering", alpha)
                    tapering_bandwidth_undersmooth = get_bandwidth(n, p, "tapering_undersmoothing", alpha)
                    banding_bandwidth = get_bandwidth(n, p, "banding", alpha)
                    for norm in norm_list: 
                        res = get_covariance_estimators(samples, true_cov, observed_distance_matrix, distance_matrix, tapering_bandwidth, banding_bandwidth, tapering_bandwidth_undersmooth,  n, p, alpha, lambd, bias, norm)
                        results = results+ res
df = pd.DataFrame(results)
# %%


# %%
