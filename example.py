# %%
from utils import *
import pandas as pd
from POET.poet import POET
import matplotlib.pyplot as plt
from wlpy.covariance import Covariance
from wlpy.gist import heatmap
# %% Hyperparameters
p = 500
n = 3000
lambd = 100
rho = 0.6
alpha = 0.5

# %%
index = np.arange(0, p, 1)
distance_matrix = np.abs(np.subtract.outer(index, index))

measurement_error = 1 * \
    generate_poisson_discrete_measurement_error(p, lambd=lambd)


observed_distance_matrix = distance_matrix + measurement_error

true_cov = generate_true_cov_cai2010(distance_matrix, rho, alpha)
heatmap(observed_distance_matrix)

# %%
samples = generate_normal_samples(true_cov, n, p)
# samples = multivariate_t_rvs(mean=np.zeros(p), true_cov=true_cov, sample_size=n, df=10)

# %%
cov_model = Covariance(samples)
sample_cov = cov_model.sample_cov()
tapering_bandwidth = np.floor(n**(1/(2 * alpha + 1)))
# tapering_bandwidth = np.floor(n**(1/2))

taper_cov = cov_tapering(sample_cov, observed_distance_matrix,
                         bandwidth=tapering_bandwidth, method="linear")
banding_bandwidth = np.floor((n/np.log(p))**(1/(2*alpha + 2)))
banding_cov = cov_tapering(sample_cov, observed_distance_matrix,
                           bandwidth=banding_bandwidth, method="banding")


# %%
tapering_bandwidth = get_bandwidth(n, p, "tapering", alpha)
tapering_bandwidth_undersmooth = get_bandwidth(
    n, p, "tapering_undersmoothing", alpha)
banding_bandwidth = get_bandwidth(n, p, "banding", alpha)

print(f"Tapering Bandwidth: {tapering_bandwidth}\nBanding Bandwidth: {banding_bandwidth}\nTaperingBandwidth Undersmooth: {tapering_bandwidth_undersmooth}")

estimators = {
    # "True Covariance": true_cov,
    "Sample Covariance": cov_model.sample_cov(),
    "Just Diagonal": np.diag(np.diag(sample_cov)),
    "Thresholding": POET(samples.T, K=0, C=0.5).SigmaU,
    "Network Banding": banding_cov,
    "Linear Shrinkage": cov_model.lw_lin_shrink(),
    # "Nonlinear Shrinkage": cov_model.nonlin_shrink(),
    "Network Tapering": taper_cov,
    "Network Tapering Undersmoothing": cov_tapering(sample_cov, observed_distance_matrix, bandwidth=tapering_bandwidth_undersmooth, method="linear"),
    "Network Tapering Corrected": correct_eigenvalues(taper_cov),
    "Oracle Network Banding": cov_tapering(sample_cov, distance_matrix, bandwidth=banding_bandwidth, method="banding"),
    "Oracle Network Tapering": cov_tapering(sample_cov, distance_matrix, bandwidth=tapering_bandwidth, method="linear"),
    "Oracle Network Tapering Undersmoothing": cov_tapering(sample_cov, distance_matrix, bandwidth=tapering_bandwidth_undersmooth, method="linear"),
    "Network Tapering Undersmoothing Alt": cov_tapering(sample_cov, observed_distance_matrix, bandwidth=np.floor((n/np.log(p))**0.5), method="linear"),
}

norm_list = ["fro", 2]
result = {}
for norm in norm_list:
    lst_true_cov = [["True Covariance", norm, f"{LA.norm(true_cov, norm): .2f}", test_if_positive_definite(
        true_cov), LA.norm(LA.inv(true_cov), norm)]]
    lst_estimators = [[name, norm, f"{LA.norm(true_cov - estimator, norm): .2f}", test_if_positive_definite(estimator), compute_norm_inverse_difference(true_cov, estimator, norm)]
                       for name, estimator in estimators.items()]
    
    result[norm] = pd.DataFrame(lst_true_cov+lst_estimators)
    result[norm].columns = ["Estimator", "Norm Type",
                            "Norm", "Positive Definite", "Difference of Inverse"]
    result[norm] = result[norm].set_index("Estimator")

result["fro"]
result[2]
# %% PLOTS
# heatmap(banding_cov, "Banding Covariance")
heatmap(taper_cov - true_cov, "Tapering Covariance")
# heatmap(cov_model.lw_lin_shrink())
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
