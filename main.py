import json
import pandas as pd

#%% Load data
#path to file on my laptop
file_path = r"C:\Users\evaha\Desktop\TVs-all-merged.json"
with open(file_path) as f:
    tv_dict = json.load(f)

#Each modelID key in the dictionary contains a list of possibly several TVs, so we gather them all TVs
all_entries = []
for listings in tv_dict.values():
    all_entries.extend(listings)
#df where each TV listing is one row
df_tvs = pd.DataFrame(all_entries)
data = df_tvs

#%% Clean data
from dataclean import data_clean_func
data_cleaned = data.copy() #new df for cleaned data

#clean everything in title
data_cleaned["title"] = data_cleaned["title"].apply(data_clean_func)

#clean everything in featuresmap
def clean_map(m): #function to clean every value within each dictionary in featuresMap col
    return {k: data_clean_func(v) for k, v in m.items()}
data_cleaned["featuresMap"] = data_cleaned["featuresMap"].apply(clean_map)

################################################
#%% Extension:count which words are important
from collections import Counter
from collections import defaultdict

#get all keys from featuresMap column
all_keys = []
for feature_map in data_cleaned["featuresMap"]:
    all_keys.extend(feature_map.keys())

#count frequency of all keys
key_counts = Counter(all_keys)

#show top 20 most common keys
top_20_keys = key_counts.most_common(100)
print("Top 20 most common keys:")
for key, count in top_20_keys:
    print(f"{key}: {count}")

keys_of_interest = [
    "Maximum Resolution",
    "Aspect Ratio",
    "Brand",
    "UPC",
    "V-Chip",
    "Screen Size (Measured Diagonally)",
    "USB Port",
    "TV Type",
    "Vertical Resolution",
    "Screen Size Class",
    "Component Video Inputs",
    "Warranty Terms - Parts",
    "HDMI Inputs",
    "Product Depth (without stand)",
    "Product Depth (with stand)"
]
#inspect the unique values per key for noise
unique_values_per_key = defaultdict(set)

for feature_map in data_cleaned["featuresMap"]:
    for key in keys_of_interest:
        if key in feature_map:
            unique_values_per_key[key].add(feature_map[key])

for key in keys_of_interest:
    print(f"Unieke values for '{key}': {unique_values_per_key[key]}") #per key a set of unique values for that feature


#%% Binary matrix MSMP+ Representation
from mw import extract_mw_title_column, extract_mw_FM_column, extract_model_words_title, extract_model_words_value_dict, binary_matrix_func

#Replication
#Sets with model word of all products
MW_title = extract_mw_title_column(data_cleaned['title']) #Set of model words in title
MW_value = extract_mw_FM_column(data_cleaned['featuresMap']) #Set of model words from featuresMap column

#every row is set of model words of product p
title_mw_list = [extract_model_words_title(t) for t in data_cleaned["title"]]
value_mw_list = [extract_model_words_value_dict(fm) for fm in data_cleaned["featuresMap"]]

#Binary matrix, with rows the elements of all model words, and columns 1 if product p contains the element
binary_matrix = binary_matrix_func(MW_title, MW_value, title_mw_list, value_mw_list)


#%% MSMP++ Representation
from mw_extension import *

#binary matrix with all model words, only 1 if contained in title
binary_matrix_titlevals = binary_matrix_func_titleonly(MW_title, MW_value, title_mw_list)

#Binary matrix for same brands
MW_brand, brand_mw_list = extract_mw_brand_column(data_cleaned["featuresMap"])
brand_matrix, MW_brand_list = brand_binary_matrix(MW_brand, brand_mw_list)

#Binary matrix for same max resolution
MW_mr, mr_mw_list = extract_mw_mr_column(data_cleaned["featuresMap"])
MR_matrix, MW_mr_list = mr_binary_matrix(MW_mr, mr_mw_list)

#Binary matrix for same UPC
MW_upc, upc_mw_list = extract_mw_upc_column(data_cleaned["featuresMap"])
UPC_matrix, MW_upc_list = upc_binary_matrix(MW_upc, upc_mw_list)

#Final binary matrix with all four matrices combined
binary_matrix_ext = np.vstack([
    binary_matrix_titlevals,
    brand_matrix,
    MR_matrix,
    UPC_matrix
])


#%% Create signature matrix with minhashing
from minhash import compute_minhash_signature

signature_matrix = compute_minhash_signature(binary_matrix)
signature_df = pd.DataFrame(signature_matrix)

signature_matrix_ext = compute_minhash_signature(binary_matrix_ext)

#%% LSH with bootstrapping
from bootstrap_lsh import run_lsh_bootstrap
results_df = run_lsh_bootstrap(signature_matrix, data_cleaned, n_bootstraps=5, seed=42)
results_df_ext = run_lsh_bootstrap(signature_matrix_ext, data_cleaned, n_bootstraps=5, seed=42)
#check
print("Replication max fraction:", results_df["fraction_comparisons"].max())
print("Extension max fraction:", results_df_ext["fraction_comparisons"].max())
print("Mean #candidate pairs replication:", results_df["num_candidate_pairs"].mean())
print("Mean #candidate pairs extension:", results_df_ext["num_candidate_pairs"].mean())

#Results for plots
import matplotlib.pyplot as plt
#sort on fractions of comparisons
results_df = results_df.sort_values("fraction_comparisons").reset_index(drop=True)
results_df_ext = results_df_ext.sort_values("fraction_comparisons").reset_index(drop=True)

#Avg over bootstrapsd rep
avg_results = (
    results_df
    .groupby("r")[["PQ", "PC", "F1", "fraction_comparisons"]]
    .mean()
    .reset_index()
    .sort_values("fraction_comparisons")
)

#Avg over bootstrapsd ext
avg_results_ext = (
    results_df_ext
    .groupby("r")[["PQ", "PC", "F1", "fraction_comparisons"]]
    .mean()
    .reset_index()
    .sort_values("fraction_comparisons")
)

### Plots with dots

#PQ
plt.figure()
plt.plot(
    avg_results["fraction_comparisons"],
    avg_results["PQ"],
    marker='o',
    label="MSMP+"
)
plt.plot(
    avg_results_ext["fraction_comparisons"],
    avg_results_ext["PQ"],
    marker='o',
    label="MSMP++"
)
plt.xlabel("Fraction of comparisons")
plt.ylabel("Pair quality")
#plt.title("Pair quality")
plt.grid(True)
plt.legend()
plt.xlim(-0.005, 0.25)
plt.show()

#PC
plt.figure()
plt.plot(
    avg_results["fraction_comparisons"],
    avg_results["PC"],
    marker='o',
    label="MSMP+"
)
plt.plot(
    avg_results_ext["fraction_comparisons"],
    avg_results_ext["PC"],
    marker='o',
    label="MSMP++"
)
plt.xlabel("Fraction of comparisons")
plt.ylabel("Pair completeness")
#plt.title("PC vs Fraction (averaged over bootstraps)")
plt.grid(True)
plt.legend()
plt.show()

#F1*
plt.figure()
plt.plot(
    avg_results["fraction_comparisons"],
    avg_results["F1"],
    marker='o',
    label="MSMP+"
)
plt.plot(
    avg_results_ext["fraction_comparisons"],
    avg_results_ext["F1"],
    marker='o',
    label="MSMP++"
)
plt.xlabel("Fraction of comparisons")
plt.ylabel("F1*")
plt.grid(True)
plt.legend()
plt.show()

#%% Areas under curves
def compute_auc(x, y):
    order = np.argsort(x)
    x_sorted = np.array(x)[order]
    y_sorted = np.array(y)[order]
    return np.trapezoid(y_sorted, x_sorted)

auc_pq_rep = compute_auc(avg_results["fraction_comparisons"], avg_results["PQ"])
auc_pc_rep = compute_auc(avg_results["fraction_comparisons"], avg_results["PC"])
auc_f1_rep = compute_auc(avg_results["fraction_comparisons"], avg_results["F1"])
auc_pq_ext = compute_auc(avg_results_ext["fraction_comparisons"], avg_results_ext["PQ"])
auc_pc_ext = compute_auc(avg_results_ext["fraction_comparisons"], avg_results_ext["PC"])
auc_f1_ext = compute_auc(avg_results_ext["fraction_comparisons"], avg_results_ext["F1"])

print("PQ Area under curve")
print(f"  Replication: {auc_pq_rep:.6f}")
print(f"  Extension:   {auc_pq_ext:.6f}\n")
print("PC Area under curve:")
print(f"  Replication: {auc_pc_rep:.6f}")
print(f"  Extension:   {auc_pc_ext:.6f}\n")
print("F1* Area under curve:")
print(f"  Replication: {auc_f1_rep:.6f}")
print(f"  Extension:   {auc_f1_ext:.6f}\n")

#Areas under curve for common ranges
x_max_common = min(avg_results["fraction_comparisons"].max(),avg_results_ext["fraction_comparisons"].max())
x_grid = np.linspace(0, x_max_common, 200)
interp = lambda xsrc, ysrc: np.interp(x_grid, xsrc, ysrc)
pq_rep_interp = interp(avg_results["fraction_comparisons"].values, avg_results["PQ"].values)
pq_ext_interp = interp(avg_results_ext["fraction_comparisons"].values,avg_results_ext["PQ"].values)
pc_rep_interp = interp(avg_results["fraction_comparisons"].values,avg_results["PC"].values)
pc_ext_interp = interp(avg_results_ext["fraction_comparisons"].values,avg_results_ext["PC"].values)
f1_rep_interp = interp(avg_results["fraction_comparisons"].values,avg_results["F1"].values)
f1_ext_interp = interp(avg_results_ext["fraction_comparisons"].values,avg_results_ext["F1"].values)

auc_pq_rep_common = np.trapezoid(pq_rep_interp, x_grid)
auc_pq_ext_common = np.trapezoid(pq_ext_interp, x_grid)
auc_pc_rep_common = np.trapezoid(pc_rep_interp, x_grid)
auc_pc_ext_common = np.trapezoid(pc_ext_interp, x_grid)
auc_f1_rep_common = np.trapezoid(f1_rep_interp, x_grid)
auc_f1_ext_common = np.trapezoid(f1_ext_interp, x_grid)

print("area under curve for same range")
print(f"PQ   Rep: {auc_pq_rep_common:.6f}   Ext: {auc_pq_ext_common:.6f}")
print(f"PC   Rep: {auc_pc_rep_common:.6f}   Ext: {auc_pc_ext_common:.6f}")
print(f"F1*  Rep: {auc_f1_rep_common:.6f}   Ext: {auc_f1_ext_common:.6f}")


#%% MSM bootstrap
from bootstrap_msm_tune import *

#Tune only epsilon
results_df_msm_tune, timings_df_tune, dissim_matrices_test_tune = run_lsh_msm_bootstrap(
    signature_matrix=signature_matrix,
    data_cleaned=data_cleaned,
    n_bootstraps=5,
    seed=42,
    epsilon_grid=np.arange(0.1, 1.0, 0.1)
)

results_df_msm_ext_tune, timings_df_ext_tune, dissim_matrices_test_ext_tune = run_lsh_msm_bootstrap(
    signature_matrix=signature_matrix_ext,
    data_cleaned=data_cleaned,
    n_bootstraps=5,
    seed=42,
    epsilon_grid=np.arange(0.1, 1.0, 0.1)
)

# F1 plot MSM
import matplotlib.pyplot as plt

# Sort
results_df_msm_tune = results_df_msm_tune.sort_values("fraction_comparisons").reset_index(drop=True)
results_df_msm_ext_tune = results_df_msm_ext_tune.sort_values("fraction_comparisons").reset_index(drop=True)

avg_results_msm_tune = (
    results_df_msm_tune
    .groupby("r")[["F1_msm", "fraction_comparisons"]]
    .mean()
    .reset_index()
    .sort_values("fraction_comparisons")
)


avg_results_msm_ext_tune = (
    results_df_msm_ext_tune
    .groupby("r")[["F1_msm", "fraction_comparisons"]]
    .mean()
    .reset_index()
    .sort_values("fraction_comparisons")
)

plt.figure(figsize=(7, 5))
plt.plot(
    avg_results_msm_tune["fraction_comparisons"],
    avg_results_msm_tune["F1_msm"],
    label="Replication (tuned MSM)",
    linewidth=2
)
plt.plot(
    avg_results_msm_ext_tune["fraction_comparisons"],
    avg_results_msm_ext_tune["F1_msm"],
    label="Extension (tuned MSM)",
    linewidth=2
)

plt.xlabel("Fraction of comparisons")
plt.ylabel("Average F1 (MSM)")
plt.title("MSM – F1 vs Fraction of comparisons (with epsilon tuning, averaged over bootstraps)")
plt.grid(True)
plt.legend()
plt.show()
