# %%
from utils import *
import pandas as pd
# %%
p = 500
n = 100
lambd = 100
rho = 0.6
alpha = 0.1

# %%
index = np.arange(0, p, 1)
distance_matrix = np.abs(np.subtract.outer(index, index))
measurement_error = generate_poisson_discrete_random_variables(p, lambd=lambd)
observed_distance_matrix = replace_diagonal(distance_matrix + measurement_error, 0)

true_cov = generate_true_cov_cai2010(distance_matrix, rho, alpha)

# %%
samples = generate_normal_samples(true_cov, n, p)
# samples = multivariate_t_rvs(mean=np.zeros(p), true_cov=true_cov, sample_size=n, df=10)

# %%
cov_model = Covariance(samples)
sample_cov = cov_model.sample_cov()


tapering_bandwidth = np.floor(n**(1/(2 * alpha + 1)))
taper_cov = cov_tapering(sample_cov, observed_distance_matrix,
                         bandwidth=tapering_bandwidth, method = "linear")

banding_bandwidth = np.floor((n/np.log(p))**(1/(2*alpha + 2)))
banding_cov = cov_tapering(sample_cov, observed_distance_matrix,
                           bandwidth=banding_bandwidth, method="banding")
# %%

compare_dict = {"Sample Covariance": cov_model.sample_cov(),
                "Linear Shrinkage": cov_model.lw_lin_shrink(),
                "Nonlinear Shrinkage": cov_model.nonlin_shrink(),
                "Network Tapering": taper_cov,
                "Network Banding": banding_cov}

norm_list = ["fro", 2]
pd.DataFrame([[key, norm, f"{LA.norm(true_cov - value, norm): .3f}"]
              for norm in norm_list for key, value in compare_dict.items()]).set_index([0, 1])


# %%
heatmap(banding_cov, "Banding Covariance")
heatmap(taper_cov, "Tapering Covariance")
heatmap(cov_model.lw_lin_shrink())
# %%
