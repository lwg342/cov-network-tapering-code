# %%
from utils import *
import pandas as pd
import matplotlib.pyplot as plt
from wlpy.covariance import Covariance
from wlpy.gist import heatmap

# %% Hyperparameters
p = 1000
n = 500
lambd = 100
rho = 0.6
alpha = 1.0
measurement_error_mean = 0
# %%
index = np.arange(0, p, 1)
distance_matrix = np.abs(np.subtract.outer(index, index))
true_cov = generate_true_cov_cai2010(distance_matrix, rho, alpha)

measurement_error = (
    1 * generate_poisson_discrete_measurement_error(p, lambd=lambd)
    + measurement_error_mean
)
observed_distance_matrix = distance_matrix + measurement_error

# heatmap(observed_distance_matrix)

# %%
samples = generate_normal_samples(true_cov, n, p)
# samples = multivariate_t_rvs(mean=np.zeros(p), true_cov=true_cov, sample_size=n, df=10)

# %%
cov_model = Covariance(samples)
sample_cov = cov_model.sample_cov()


# %%
tapering_bandwidth = get_bandwidth(n, p, "tapering", alpha)
tapering_bandwidth_oversmooth = get_bandwidth(n, p, "tapering_oversmoothing", alpha)
# tapering_bandwidth_oversmooth = 10*np.log(p)
banding_bandwidth = get_bandwidth(n, p, "banding", alpha)

taper_cov = cov_tapering(
    sample_cov, observed_distance_matrix, bandwidth=tapering_bandwidth, method="linear"
)
banding_cov = cov_tapering(
    sample_cov, observed_distance_matrix, bandwidth=banding_bandwidth, method="banding"
)


print(
    f"Tapering Bandwidth: {tapering_bandwidth}\nBanding Bandwidth: {banding_bandwidth}\nTaperingBandwidth oversmooth: {tapering_bandwidth_oversmooth}"
)

estimators = {
    # "True Covariance": true_cov,
    "Sample Covariance": cov_model.sample_cov(),
    "Just Diagonal": np.diag(np.diag(sample_cov)),
    "Thresholding": cov_model.threshold_corr(regularization_constant=0.5),
    "Network Banding": banding_cov,
    "Linear Shrinkage": cov_model.linear_shrinkage(),
    # "Nonlinear Shrinkage": cov_model.nonlin_shrink(),
    "Network Tapering": taper_cov,
    "Network Tapering oversmoothing": cov_tapering(
        sample_cov,
        observed_distance_matrix,
        bandwidth=tapering_bandwidth_oversmooth,
        method="linear",
    ),
    "Network Tapering Corrected": correct_eigenvalues(taper_cov),
    "Network Banding with True Distance": cov_tapering(
        sample_cov, distance_matrix, bandwidth=banding_bandwidth, method="banding"
    ),
    "Network Tapering with True Distance": cov_tapering(
        sample_cov, distance_matrix, bandwidth=tapering_bandwidth, method="linear"
    ),
    "Network Tapering with True Distance oversmoothing": cov_tapering(
        sample_cov,
        distance_matrix,
        bandwidth=tapering_bandwidth_oversmooth,
        method="linear",
    ),
    "Network Tapering oversmoothing Alt": cov_tapering(
        sample_cov,
        observed_distance_matrix,
        bandwidth=np.floor((n / np.log(p)) ** 0.5),
        method="linear",
    ),
}

norm_list = ["fro", 2]
result = {}
for norm in norm_list:
    lst_true_cov = [
        [
            "True Covariance",
            norm,
            f"{LA.norm(true_cov, norm): .2f}",
            test_if_positive_definite(true_cov),
            LA.norm(LA.inv(true_cov), norm),
        ]
    ]
    lst_estimators = [
        [
            name,
            norm,
            f"{LA.norm(true_cov - estimator, norm): .2f}",
            test_if_positive_definite(estimator),
            compute_norm_inverse_difference(true_cov, estimator, norm),
        ]
        for name, estimator in estimators.items()
    ]

    result[norm] = pd.DataFrame(lst_true_cov + lst_estimators)
    result[norm].columns = [
        "Estimator",
        "Norm Type",
        "Norm",
        "Positive Definite",
        "Difference of Inverse",
    ]
    result[norm] = result[norm].set_index("Estimator")

result["fro"]
result[2]
# %% PLOTS
# heatmap(banding_cov, "Banding Covariance")
heatmap(taper_cov - true_cov, "Tapering Covariance")
# heatmap(cov_model.linear_shrinkage())
# %%
observed_distance_matrix
heatmap(observed_distance_matrix, "Observed Distance Matrix")
# %%
plt.hist(measurement_error.reshape(-1))
# %%
compute_smallest_eigenvalue(estimators["Network Tapering"])
# # %%
# Test if changing the bandwidth changes the positive definiteness

# for j in range(50):
#     tapering_bandwidth = np.floor(n**(1/(2 * alpha + 1))) - j
#     # tapering_bandwidth = np.floor(n**(1/2))

#     taper_cov = cov_tapering(sample_cov, observed_distance_matrix,
#                             bandwidth=tapering_bandwidth, method="linear")

#     print(f"{j}: {test_if_positive_definite(taper_cov)}")
# %%
# %%
nsim = 2
sample_size_list = [100]
p_list = [100]
alpha_list = [0.1]
lambd_list = [0.0]
measurement_error_mean = [0.0, -1e1]
norm_list = [2]
seed_list = [2**j for j in range(nsim)]
rho = 0.6


def get_covariance_estimators(
    n,
    p,
    alpha,
    lambd,
    bias,
    norm_type,
    distance_matrix,
    true_cov,
    tapering_bandwidth,
    tapering_bandwidth_oversmooth,
    banding_bandwidth,
    samples,
    observed_distance_matrix,
):
    cov_model = Covariance(samples)
    sample_cov = cov_model.sample_cov()

    taper_cov = cov_tapering(
        sample_cov,
        observed_distance_matrix,
        bandwidth=tapering_bandwidth,
        method="linear",
    )
    banding_cov = cov_tapering(
        sample_cov,
        observed_distance_matrix,
        bandwidth=banding_bandwidth,
        method="banding",
    )

    estimators = {
        "Sample Covariance": cov_model.sample_cov(),
        "Just Diagonal": np.diag(np.diag(sample_cov)),
        "Thresholding": cov_model.threshold_corr(regularization_constant=0.5),
        "Network Banding": banding_cov,
        "Linear Shrinkage": cov_model.linear_shrinkage(),
        "Network Tapering": taper_cov,
        "Network Tapering oversmoothing": cov_tapering(
            sample_cov,
            observed_distance_matrix,
            bandwidth=tapering_bandwidth_oversmooth,
            method="linear",
        ),
        "Oracle Network Banding": cov_tapering(
            sample_cov, distance_matrix, bandwidth=banding_bandwidth, method="banding"
        ),
        "Oracle Network Tapering": cov_tapering(
            sample_cov, distance_matrix, bandwidth=tapering_bandwidth, method="linear"
        ),
        # "Oracle Network Tapering oversmoothing": cov_tapering(sample_cov, distance_matrix, bandwidth=tapering_bandwidth_oversmooth, method="linear"),
    }

    col_name = [
        "Name",
        "Norm Type",
        "Norm",
        "Sample Size",
        "Feature Dimension",
        "\\alpha",
        "\\lambda",
        "b",
    ]

    res_true = list_to_dict(
        [
            "True Covariance",
            norm_type,
            f"{LA.norm(true_cov, norm_type): .2f}",
            n,
            p,
            alpha,
            lambd,
            bias,
        ],
        col_name,
    )

    res_estimators = [
        list_to_dict(
            [
                name,
                norm_type,
                f"{LA.norm(true_cov - estimator, norm_type): .2f}",
                n,
                p,
                alpha,
                lambd,
                bias,
            ],
            col_name,
        )
        for name, estimator in estimators.items()
    ]
    res_estimators.append(res_true)
    return res_estimators


results = []
for p in p_list:
    index = np.arange(0, p, 1)
    distance_matrix = np.abs(np.subtract.outer(index, index))
    for alpha in alpha_list:
        true_cov = generate_true_cov_cai2010(distance_matrix, rho, alpha)
        tapering_bandwidth = get_bandwidth(n, p, "tapering", alpha)
        tapering_bandwidth_oversmooth = get_bandwidth(
            n, p, "tapering_oversmoothing", alpha
        )
        banding_bandwidth = get_bandwidth(n, p, "banding", alpha)

        for n in sample_size_list:
            for lambd in lambd_list:
                for bias in measurement_error_mean:
                    for norm_type in norm_list:
                        samples = generate_normal_samples(true_cov, n, p)
                        measurement_error = (
                            1
                            * generate_poisson_discrete_measurement_error(
                                p, lambd=lambd
                            )
                            + bias
                        )
                        observed_distance_matrix = distance_matrix + measurement_error
                        res = get_covariance_estimators(
                            n,
                            p,
                            alpha,
                            lambd,
                            bias,
                            norm_type,
                            distance_matrix,
                            true_cov,
                            tapering_bandwidth,
                            tapering_bandwidth_oversmooth,
                            banding_bandwidth,
                            samples,
                            observed_distance_matrix,
                        )
                        results = results + res
df = pd.DataFrame(results)
df.sort_values(by=["Norm"], inplace=True)
df
# %%
