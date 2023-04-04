# %%
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

output_folder = "cov-network-tapering-latex"
error_distribution = "Gaussian"

file_list = [
    f"p_100_{error_distribution}.pkl",
    f"p_200_{error_distribution}.pkl",
    f"p_500_{error_distribution}.pkl",
    f"p_1000_{error_distribution}.pkl",
]
load_items = {}
for name in file_list:
    with open(f"{name}", "rb") as f:
        load_items[name] = pickle.load(f)
result = pd.concat([load_items[name].reset_index() for name in file_list], axis=0)
result
# %%
# result["lambd"] = result["lambd"].astype(int)
mapping = {10: "$\\mathbf{10^1}$", 100: "$\\mathbf{10^2}$", 10000: "$\\mathbf{10^4}$"}
result["lambd_sc"] = result["lambd"].map(mapping)

# %%
lambd = 100
table1 = (
    result.loc[(result["lambd"] == lambd)]
    .set_index(["alpha", "p", "sample_size"])
    .sort_index()
    .drop(columns=["lambd", "bias", "norm_type", "lambd_sc"])
)
table1.index.names = ["$\\alpha$", "$p$", "$n$"]
table1.to_latex(
    f"{output_folder}/table_lambda_100_{error_distribution}.tex",
    float_format="%.3f",
    bold_rows=True,
    longtable=True,
    column_format="p{0.7cm}p{0.7cm}p{0.7cm}p{0.8cm}p{0.8cm}p{0.8cm}p{0.8cm}p{0.8cm}p{0.8cm}p{0.8cm}p{0.8cm}p{0.8cm}",
    header=["$\\Sigma$", "SC", "LS", "ST", "NT", "NTD", "NTUS", "NB", "NBD"],
    escape=False,
    caption=f"The average error in terms of spectral norm for different estimators, with the measurement errors generated from a {error_distribution} distribution with parameter $\\lambda = {lambd}$. The first column is the spectral norm of the true covariance matrix $\\Sigma$. The following columns are the spectral norm of sample covariance estimator(SC), the linear shrinkage estimator (LS), the soft thresholding estimator (ST), the network tapering estimator (NT), the network tapering estimator with true distance matrix (NTD), the network tapering estimator with the undersmoothing correction (NTUS), the network banding estimator (NB), and the network banding estimator with the true distance matrix (NBD) ",
    label=f"table:spectral_error_{error_distribution}",
)

table1
# %%
alpha = 0.1

table2 = (
    result.loc[result["alpha"] == alpha]
    .set_index(["p", "sample_size", "lambd_sc"])
    .sort_index()
    .drop(["alpha", "bias", "norm_type", "lambd"], axis=1)
)
table2.index.names = ["$p$", "$n$", "$\\lambda$"]
table2.to_latex(
    f"{output_folder}/table_varying_lambda_alpha_0.1_{error_distribution}.tex",
    float_format="%.3f",
    bold_rows=True,
    longtable=True,
    column_format="p{0.7cm}p{0.7cm}p{0.7cm}p{0.8cm}p{0.8cm}p{0.8cm}p{0.8cm}p{0.8cm}p{0.8cm}p{0.8cm}p{0.8cm}p{0.8cm}",
    header=["$\\Sigma$", "SC", "LS", "ST", "NT", "NTD", "NTUS", "NB", "NBD"],
    escape=False,
    caption=f"The average error in terms of spectral norm for different estimators, with alpha fixed at {alpha}. The measurement errors of the distance are generated from a {error_distribution} distribution with parameter $\\lambda$. The first column is the spectral norm of the true covariance matrix $\\Sigma$. The following columns are the spectral norm of sample covariance estimator(SC), the linear shrinkage estimator (LS), the soft thresholding estimator (ST), the network tapering estimator (NT), the network tapering estimator with true distance matrix (NTD), the network tapering estimator with the undersmoothing correction (NTUS), the network banding estimator (NB), and the network banding estimator with the true distance matrix (NBD) ",
    label=f"table:measurement_error_alpha_{alpha}_{error_distribution}",
)

table2
# %%
alpha = 0.5

table3 = (
    result.loc[result["alpha"] == alpha]
    .set_index(["p", "sample_size", "lambd_sc"])
    .sort_index()
    .drop(["alpha", "bias", "norm_type", "lambd"], axis=1)
)
table3.index.names = ["$p$", "$n$", "$\\lambda$"]
table3.to_latex(
    f"{output_folder}/table_varying_lambda_alpha_{alpha}_{error_distribution}.tex",
    float_format="%.3f",
    bold_rows=True,
    longtable=True,
    column_format="p{0.7cm}p{0.7cm}p{0.7cm}p{0.8cm}p{0.8cm}p{0.8cm}p{0.8cm}p{0.8cm}p{0.8cm}p{0.8cm}p{0.8cm}p{0.8cm}",
    header=["$\\Sigma$", "SC", "LS", "ST", "NT", "NTD", "NTUS", "NB", "NBD"],
    escape=False,
    caption=f"The average error in terms of spectral norm for different estimators, with alpha fixed at {alpha}. The measurement errors of the distance are generated from a {error_distribution} distribution with parameter $\\lambda$. The first column is the spectral norm of the true covariance matrix $\\Sigma$. The following columns are the spectral norm of sample covariance estimator(SC), the linear shrinkage estimator (LS), the soft thresholding estimator (ST), the network tapering estimator (NT), the network tapering estimator with true distance matrix (NTD), the network tapering estimator with the undersmoothing correction (NTUS), the network banding estimator (NB), and the network banding estimator with the true distance matrix (NBD) ",
    label=f"table:measurement_error_alpha_{alpha}_{error_distribution}",
)

table3
# %%
alpha = 1.0

table4 = (
    result.loc[result["alpha"] == alpha]
    .set_index(["p", "sample_size", "lambd_sc"])
    .sort_index()
    .drop(["alpha", "bias", "norm_type", "lambd"], axis=1)
)
table4.index.names = ["$p$", "$n$", "$\\lambda$"]
table4.to_latex(
    f"{output_folder}/table_varying_lambda_alpha_{alpha}_{error_distribution}.tex",
    float_format="%.3f",
    bold_rows=True,
    longtable=True,
    column_format="p{0.7cm}p{0.7cm}p{0.7cm}p{0.8cm}p{0.8cm}p{0.8cm}p{0.8cm}p{0.8cm}p{0.8cm}p{0.8cm}p{0.8cm}p{0.8cm}",
    header=["$\\Sigma$", "SC", "LS", "ST", "NT", "NTD", "NTUS", "NB", "NBD"],
    escape=False,
    caption=f"The average error in terms of spectral norm for different estimators, with alpha fixed at {alpha}. The measurement errors of the distance are generated from a {error_distribution} distribution with parameter $\\lambda$. The first column is the spectral norm of the true covariance matrix $\\Sigma$. The following columns are the spectral norm of sample covariance estimator(SC), the linear shrinkage estimator (LS), the soft thresholding estimator (ST), the network tapering estimator (NT), the network tapering estimator with true distance matrix (NTD), the network tapering estimator with the undersmoothing correction (NTUS), the network banding estimator (NB), and the network banding estimator with the true distance matrix (NBD) ",
    label=f"table:measurement_error_alpha_{alpha}_{error_distribution}",
)

table4
# %%
with open(f"varying_alpha_{error_distribution}.pkl", "rb") as file:
    result_alpha = pickle.load(file)
result_alpha = result_alpha.reset_index()
df_plot = result_alpha.loc[(result_alpha["lambd"] == 100)]


df_plot = df_plot[
    [
        "alpha",
        "True Covariance",
        "Sample Covariance",
        "Network Tapering",
        "Network Tapering with True Distance Matrix",
        "Network Tapering Undersmoothing",
        "Soft Thresholding",
    ]
].set_index("alpha")
col_to_divide_by = "True Covariance"
df_plot.loc[:, df_plot.columns != col_to_divide_by] = df_plot.loc[
    :, df_plot.columns != col_to_divide_by
].div(df_plot[col_to_divide_by], axis=0)
df_plot.drop(col_to_divide_by, axis=1)

font = {"family": "Times", "weight": "normal", "size": 12}
plt.rc("font", **font)
ax = df_plot[
    [
        "Sample Covariance",
        "Network Tapering",
        "Network Tapering with True Distance Matrix",
        "Network Tapering Undersmoothing",
        "Soft Thresholding",
    ]
].plot()
ax.set_xticks(np.linspace(0.1, 1, 10))
ax.legend(["SC", "NT", "NTD", "NTUS", "ST"], loc="center left", bbox_to_anchor=(1, 0.5))
ax.set_xlabel("$\\alpha$")
ax.set_ylabel("Estimation Error")
plt.savefig(f"{output_folder}/varying_alpha_{error_distribution}.pdf", bbox_inches="tight")
