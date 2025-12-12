import numpy as np
import pandas as pd

from lsh import lsh_candidatepairs, lsh_performance
from helpers import count_duplicates_for_subset

#Lsh bootstrap for a signature matrix
def run_lsh_bootstrap(signature_matrix, data, n_bootstraps=5, seed=42):
    n_hashes, n_products = signature_matrix.shape

    rng = np.random.default_rng(seed)
    all_results = []

    for bootstrap in range(n_bootstraps):
        print(f"Bootstrap {bootstrap + 1}/{n_bootstraps}")

        #boostrap sampling with replacement
        train_indices = rng.integers(low=0, high=n_products, size=n_products)
        # all indices that are not in train are in test
        train_unique = set(train_indices)
        all_indices = set(range(n_products))
        test_indices = sorted(list(all_indices - train_unique))

        #get the subset signature matrix from the total signature matrix en the original testdata indices
        sig_test = signature_matrix[:, test_indices]
        data_test = data.iloc[test_indices].reset_index(drop=True)

        #calc total possible comparisons and total true duplicates in the testset
        n_test = sig_test.shape[1]
        total_possible_comparisons_test = n_test * (n_test - 1) / 2
        total_duplicates_test = count_duplicates_for_subset(data_test)

        print(f"  Test size = {n_test}, true duplicate pairs = {total_duplicates_test}")

        #Loop over alle r for which n is divisble by r, such that b = nxr
        for r in range(1, n_hashes + 1):
            if n_hashes % r != 0:
                continue

            candidate_pairs = lsh_candidatepairs(sig_test, r)

            metrics = lsh_performance(
                candidate_pairs=candidate_pairs,
                r=r,
                signature_matrix=sig_test,
                total_possible_comparisons=total_possible_comparisons_test,
                total_duplicates=total_duplicates_test,
                data=data_test
            )

            metrics["bootstrap"] = bootstrap
            metrics["r"] = r
            metrics["num_candidate_pairs"] = len(candidate_pairs)

            all_results.append(metrics)

    return pd.DataFrame(all_results) #Returns df with all metrics, one row per combination bootstrapxr
