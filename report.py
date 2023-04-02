# %%
import pickle
import pandas as pd

file_list = ["p_100.pkl", "p_200.pkl", "p_500.pkl", "p_1000.pkl"]
load_items = {}
for name in file_list:
    with open(f"{name}", "rb") as f:
        load_items[name] = pickle.load(f)
result = pd.concat([load_items[name].reset_index() for name in file_list], axis=0)
result

# %%
alpha = 0.1
lambd = 100
table1 = (
    result.loc[(result["alpha"] == alpha) & (result["lambd"] == lambd)]
    .set_index(["p", "sample_size"])
    .drop(columns=["alpha", "lambd", "bias", "norm_type"])
)
table1.index.names = ["$p$", "$n$"]
table1.to_latex(
    "table_alpha_0.1.tex",
    float_format="%.2f",
    bold_rows=True,
    longtable=True,
    column_format="p{1cm}p{1cm}p{1cm}p{1cm}p{1cm}p{1cm}p{1cm}p{1cm}p{1cm}p{1cm}p{1cm}",
    header=["$\\Sigma$", "$\\bar{\\Sigma}$", "LS", "ST", "NT", "NTTDM", "NTUS", "NB", "NBTDM"],
    escape=False
)
table1
# %%
lambd = 100
table2 = (
    result.loc[(result["lambd"] == lambd)]
    .set_index(["alpha", "p", "sample_size"])
    .sort_index()
    .drop(columns=["lambd", "bias", "norm_type"])
)
table2.index.names = ["$\\alpha$", "$p$", "$n$"]
table2.to_latex(
    "table_lambda.tex",
    float_format="%.3f",
    bold_rows=True,
    longtable=True,
    column_format="p{1cm}p{1cm}p{1cm}p{1cm}p{1cm}p{1cm}p{1cm}p{1cm}p{1cm}p{1cm}p{1cm}p{1cm}",
    header=["$\\Sigma$", "$\\bar{\\Sigma}$", "LS", "ST", "NT", "NTTDM", "NTUS", "NB", "NBTDM"],
    escape=False
)
table2
# %%
