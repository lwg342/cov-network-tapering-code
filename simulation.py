# %%
import pandas as pd
import numpy as np
from POET.poet import POET
from utils import *
from wlpy.covariance import Covariance
# %%
nsim = 100
sample_size_list = [100]
p_list = [100]
alpha_list = [0.1, 0.8]
lambd_list = [0.0]
measurement_error_mean = [0.0]
norm_list = [2]
seed_list = [2*j for j in range(nsim)]
rho = 0.6
# %%


def generate_fixed_part(sample_size, p, alpha, lambd, bias, norm_type):

    index = np.arange(0, p, 1)
    distance_matrix = np.abs(np.subtract.outer(index, index))

    true_cov = generate_true_cov_cai2010(distance_matrix, rho, alpha)
    tapering_bandwidth = np.floor(sample_size**(1/(2 * alpha + 1)))
    tapering_bandwidth_undersmooth = np.floor(
        sample_size**(1/(2 * alpha + 1))) + 1
    banding_bandwidth = np.floor(sample_size**(1/(2 * alpha + 1)))
    fixed_part = {"sample_size": sample_size,
                  "p": p,
                  "alpha": alpha,
                  "lambd": lambd,
                  "bias": bias,
                  "true_cov": true_cov,
                  "distance_matrix": distance_matrix,
                  "tapering_bandwidth": tapering_bandwidth, "tapering_bandwidth_undersmooth": tapering_bandwidth_undersmooth,
                  "banding_bandwidth": banding_bandwidth,
                  "norm_type": norm_type,
                  }
    return fixed_part


def generate_random_part(fixed_part, seed=None, **kwargs):
    true_cov = fixed_part["true_cov"]
    sample_size = fixed_part["sample_size"]
    p = fixed_part["p"]
    lambd = fixed_part["lambd"]
    distance_matrix = fixed_part["distance_matrix"]

    samples = generate_normal_samples(true_cov, sample_size, p, seed)
    measurement_error = 1 * \
        generate_poisson_discrete_measurement_error(
            p, lambd=lambd, seed=seed) + bias
    observed_distance_matrix = distance_matrix + measurement_error
    random_part = {"samples": samples,
                   "observed_distance_matrix": observed_distance_matrix}
    return random_part


def compute_average_loss(nsim, random_part_generator, fixed_part, seed_list=None, **kwargs):
    true_cov = fixed_part["true_cov"]
    norm_type = fixed_part["norm_type"]
    rslt = [
        [
            LA.norm(true_cov - cov_model.sample_cov(), norm_type),
            LA.norm(true_cov - cov_tapering(cov_model.sample_cov(), random_part["observed_distance_matrix"],
                                            bandwidth=fixed_part["tapering_bandwidth"],
                                            method="linear", **random_part), norm_type),
            LA.norm(true_cov - cov_model.lw_lin_shrink(), norm_type)
        ]
        for cov_model, random_part in [
            (
                Covariance(random_part_generator(
                    fixed_part, seed=seed_list[i])["samples"]),
                random_part_generator(fixed_part, seed=seed_list[i])
            )
            for i in range(nsim)
        ]
    ]
    return np.mean(rslt, axis=0)


results = []
for p in p_list:
    for alpha in alpha_list:
        for sample_size in sample_size_list:
            for lambd in lambd_list:
                for bias in measurement_error_mean:
                    for norm_type in norm_list:
                        fixed_part = generate_fixed_part(
                            sample_size, p, alpha, lambd, bias, norm_type)
                        loss = compute_average_loss(
                            nsim, generate_random_part, fixed_part, seed_list=seed_list, **fixed_part)
                        res = list_to_dict(
                            loss, ["Sample Covariance", "Network Tapering", "Linear Shrinkage"])
                        res.update(dict((key, fixed_part[key]) for key in [
                            'sample_size', 'p', 'alpha', 'lambd', 'bias', 'norm_type']))
                        results.append(res)
# results = pd.DataFrame(results)
results = pd.DataFrame(results)
results
