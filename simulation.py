# %%
import pickle
import pandas as pd
import numpy as np
from utils import *
from wlpy.covariance import Covariance
import time

error_distribution = "Poisson"
print("Measurement Error Distribution:", error_distribution)
# %%


def generate_fixed_part(
    sample_size, p, alpha, lambd, bias, norm_type, error_distribution
):
    index = np.arange(0, p, 1)
    distance_matrix = np.abs(np.subtract.outer(index, index))

    true_cov = generate_true_cov_cai2010(distance_matrix, rho, alpha)
    tapering_bandwidth = get_bandwidth(sample_size, p, "tapering", alpha)
    banding_bandwidth = get_bandwidth(sample_size, p, "banding", alpha)
    tapering_bandwidth_oversmooth = get_bandwidth(
        sample_size, p, "tapering_oversmoothing", alpha
    )
    fixed_part = {
        "sample_size": sample_size,
        "p": p,
        "alpha": alpha,
        "lambd": lambd,
        "bias": bias,
        "true_cov": true_cov,
        "distance_matrix": distance_matrix,
        "tapering_bandwidth": tapering_bandwidth,
        "tapering_bandwidth_oversmooth": tapering_bandwidth_oversmooth,
        "banding_bandwidth": banding_bandwidth,
        "norm_type": norm_type,
        "error_distribution": error_distribution,
    }
    return fixed_part


def generate_random_part(fixed_part, seed=None, **kwargs):
    true_cov = fixed_part["true_cov"]
    sample_size = fixed_part["sample_size"]
    p = fixed_part["p"]
    lambd = fixed_part["lambd"]
    distance_matrix = fixed_part["distance_matrix"]
    error_distribution = fixed_part["error_distribution"]
    bias = fixed_part["bias"]

    samples = generate_normal_samples(true_cov, sample_size, p, seed)

    if error_distribution == "Poisson":
        measurement_error = (
            1 * generate_poisson_discrete_measurement_error(p, lambd=lambd, seed=seed)
            + bias
        )
    elif error_distribution == "Gaussian":
        measurement_error = (
            1
            * generate_normal_discrete_measurement_error(
                p, sigma=np.sqrt(lambd), seed=seed
            )
            + bias
        )
    else:
        raise NotImplementedError("Only Gaussian and Poisson are implemented")

    observed_distance_matrix = distance_matrix + measurement_error
    cov_model = Covariance(samples)
    random_part = {
        "samples": samples,
        "observed_distance_matrix": observed_distance_matrix,
        "cov_model": cov_model,
        "sample_cov_estimate": cov_model.sample_cov_estimate,
        "linear_shrinkage": cov_model.linear_shrinkage(),
        "threhsold_corr": cov_model.threshold_corr(regularization_constant=0.5),
        "network_tapering_true_distance_matrix": cov_tapering(
            cov_model.sample_cov_estimate,
            fixed_part["distance_matrix"],
            bandwidth=fixed_part["tapering_bandwidth"],
            method="linear",
        ),
        "network_banding_true_distance_matrix": cov_tapering(
            cov_model.sample_cov_estimate,
            fixed_part["distance_matrix"],
            bandwidth=fixed_part["banding_bandwidth"],
            method="banding",
        ),
    }
    return random_part


def compute_average_loss(
    nsim, random_part_generator, fixed_part, seed_list=None, **kwargs
):
    true_cov = fixed_part["true_cov"]
    norm_type = fixed_part["norm_type"]
    rslt = [
        [
            # Sample Covariance
            LA.norm(true_cov - random_part["sample_cov_estimate"], norm_type),
            # Linear Shrinkage
            LA.norm(true_cov - random_part["linear_shrinkage"], norm_type),
            # Threshold Correlation
            LA.norm(
                true_cov - random_part["threhsold_corr"],
                norm_type,
            ),
            LA.norm(
                true_cov
                - cov_tapering(
                    random_part["cov_model"].sample_cov_estimate,
                    random_part["observed_distance_matrix"],
                    bandwidth=fixed_part["tapering_bandwidth"],
                    method="linear",
                    **random_part,
                ),
                norm_type,
            ),
            # Network Tapering with True Distance Matrix
            LA.norm(
                true_cov - random_part["network_tapering_true_distance_matrix"],
                norm_type,
            ),
            # Network Tapering oversmoothing
            LA.norm(
                true_cov
                - cov_tapering(
                    random_part["cov_model"].sample_cov_estimate,
                    random_part["observed_distance_matrix"],
                    bandwidth=fixed_part["tapering_bandwidth_oversmooth"],
                    method="linear",
                    **random_part,
                ),
                norm_type,
            ),
            # Network Banding
            LA.norm(
                true_cov
                - cov_tapering(
                    random_part["cov_model"].sample_cov_estimate,
                    random_part["observed_distance_matrix"],
                    bandwidth=fixed_part["banding_bandwidth"],
                    method="banding",
                    **random_part,
                ),
                norm_type,
            ),
            # Network Banding with True Distance Matrix
            LA.norm(
                true_cov - random_part["network_banding_true_distance_matrix"],
                norm_type,
            ),
        ]
        for random_part in [
            (random_part_generator(fixed_part, seed=seed_list[i])) for i in range(nsim)
        ]
    ]
    return np.mean(rslt, axis=0)


nsim = 100
sample_size_list = [100, 200, 500]
p_list = [100, 200, 500, 1000]
alpha_list = [0.1, 0.25, 0.5, 0.75, 1.0]
lambd_list = [1e1, 1e2, 1e3, 1e4]
measurement_error_mean = [0.0]
norm_list = [2]
seed_list = [2 * j for j in range(nsim)]
rho = 0.6
estimator_list = [
    "Sample Covariance",
    "Linear Shrinkage",
    "Soft Thresholding",
    "Network Tapering",
    "Network Tapering with True Distance Matrix",
    "Network Tapering oversmoothing",
    "Network Banding",
    "Network Banding with True Distance Matrix",
]

# %%


for p in p_list:
    result = []
    for alpha in alpha_list:
        for sample_size in sample_size_list:
            start_time = time.time()
            for lambd in lambd_list:
                for bias in measurement_error_mean:
                    for norm_type in norm_list:
                        fixed_part = generate_fixed_part(
                            sample_size,
                            p,
                            alpha,
                            lambd,
                            bias,
                            norm_type,
                            error_distribution,
                        )
                        loss = compute_average_loss(
                            nsim,
                            generate_random_part,
                            fixed_part,
                            seed_list=seed_list,
                            **fixed_part,
                        )
                        res = {
                            "True Covariance": LA.norm(
                                fixed_part["true_cov"], norm_type
                            )
                        }
                        res.update(list_to_dict(loss, estimator_list))
                        res.update(
                            dict(
                                (key, fixed_part[key])
                                for key in [
                                    "sample_size",
                                    "p",
                                    "alpha",
                                    "lambd",
                                    "bias",
                                    "norm_type",
                                ]
                            )
                        )
                        result.append(res)

            end_time = time.time()
            time_elapsed = round(end_time - start_time)
            print(
                f"p={p}, alpha={alpha}, sample_size={sample_size}, Time elapsed: {time_elapsed} seconds"
            )
    result = pd.DataFrame(result).set_index(
        ["sample_size", "p", "alpha", "lambd", "bias", "norm_type"]
    )
    with open(f"p_{p}_{error_distribution}.pkl", "wb") as file:
        pickle.dump(result, file)

print("Finished")
# %%

p = 500
sample_size = 200
alpha_list = np.linspace(0.1, 1.0, 19)
lambd_list = [1e1, 1e2, 1e3, 1e4]
seed_list = [2 * j for j in range(nsim)]

result_varying_alpha = []
for alpha in alpha_list:
    start_time = time.time()
    for lambd in lambd_list:
        for bias in measurement_error_mean:
            for norm_type in norm_list:
                fixed_part = generate_fixed_part(
                    sample_size, p, alpha, lambd, bias, norm_type, error_distribution
                )
                loss = compute_average_loss(
                    nsim,
                    generate_random_part,
                    fixed_part,
                    seed_list=seed_list,
                    **fixed_part,
                )
                res = {"True Covariance": LA.norm(fixed_part["true_cov"], norm_type)}
                res.update(list_to_dict(loss, estimator_list))
                res.update(
                    dict(
                        (key, fixed_part[key])
                        for key in [
                            "sample_size",
                            "p",
                            "alpha",
                            "lambd",
                            "bias",
                            "norm_type",
                        ]
                    )
                )
                result_varying_alpha.append(res)

        end_time = time.time()
        time_elapsed = round(end_time - start_time)
        print(
            f"p={p}, alpha={alpha}, sample_size={sample_size}, Time elapsed: {time_elapsed} seconds"
        )
result_varying_alpha = pd.DataFrame(result_varying_alpha).set_index(
    ["sample_size", "p", "alpha", "lambd", "bias", "norm_type"]
)
with open(f"varying_alpha_{error_distribution}.pkl", "wb") as file:
    pickle.dump(result_varying_alpha, file)

print("Finished")
