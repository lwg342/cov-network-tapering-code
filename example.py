# %%
from utils import *
import pandas as pd
from POET.poet import POET
import matplotlib.pyplot as plt
from wlpy.covariance import Covariance
from wlpy.gist import heatmap
# %%
p = 500
n = 100
lambd = 100
rho = 0.6
alpha = 0.1

# %%
index = np.arange(0, p, 1)
distance_matrix = np.abs(np.subtract.outer(index, index))

measurement_error = 5*generate_poisson_discrete_measurement_error(p, lambd=10)


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

print(f"Tapering Bandwidth: {tapering_bandwidth}")
# %%
tapering_bandwidth = get_bandwidth(n,p,"tapering", alpha)
tapering_bandwidth_undersmooth = get_bandwidth(n,p, "tapering_undersmoothing", alpha)
banding_bandwidth = get_bandwidth(n,p,"banding", alpha)

estimators = {"Sample Covariance": cov_model.sample_cov(),
              "Just Diagonal": np.diag(np.diag(sample_cov)),
              "Thresholding": POET(samples.T, K=0, C=0.5).SigmaU,
              "Network Banding": banding_cov,
              "Linear Shrinkage": cov_model.lw_lin_shrink(),
              # "Nonlinear Shrinkage": cov_model.nonlin_shrink(),
              "Network Tapering": taper_cov,
              "Network Tapering Undersmoothing": cov_tapering(sample_cov, observed_distance_matrix, bandwidth=tapering_bandwidth_undersmooth, method="linear"),
              "Network Tapering Corrected": correct_eigenvalues(taper_cov),
              "Banding with True Distance Matrix": cov_tapering(sample_cov, distance_matrix, bandwidth=banding_bandwidth, method="banding"),
              "Tapering with True Distance Matrix": cov_tapering(sample_cov, distance_matrix, bandwidth=tapering_bandwidth, method="linear"),
              "Tapering with True Distance Matrix Undersmoothing": cov_tapering(sample_cov, distance_matrix, bandwidth=tapering_bandwidth_undersmooth, method="linear"),
              }

norm_list = ["fro", 2]
pd.DataFrame([[name, norm, f"{LA.norm(true_cov - estimator, norm): .2f}", test_if_positive_definite(estimator), compute_norm_inverse_difference(true_cov, estimator, norm)]
              for norm in norm_list for name, estimator in estimators.items()]).set_index([0])


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
