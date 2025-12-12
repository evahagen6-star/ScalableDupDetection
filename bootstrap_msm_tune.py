import numpy as np
import pandas as pd
import time

from lsh import lsh_candidatepairs, lsh_performance
from helpers import count_duplicates_for_subset
from msm import *

#function for bootsrtapping for lsh and msm and tuning of epsilon
def run_lsh_msm_bootstrap(signature_matrix, data_cleaned, n_bootstraps=5, seed=99, epsilon_grid=None):
    start_time = time.time()
    timings = []
    dissim_matrices_test = {}

    n_hashes, n_products = signature_matrix.shape

    rng = np.random.default_rng(seed)

    all_results = []

    for bootstrap in range(n_bootstraps):
        print(f"Bootstrap {bootstrap + 1}/{n_bootstraps}")

        #boostrap with resampling for the train indices
        train_indices = rng.integers(low=0, high=n_products, size=n_products)

        #test indices
        train_unique = set(train_indices)
        all_indices = set(range(n_products))
        test_indices = sorted(list(all_indices - train_unique))

        #train and test sign matrix and data
        sig_train = signature_matrix[:, train_indices]
        data_train = data_cleaned.iloc[train_indices].reset_index(drop=True)

        sig_test = signature_matrix[:, test_indices]
        data_test = data_cleaned.iloc[test_indices].reset_index(drop=True)

        #calc total possible,comparisons en total duplicates for the train and testset
        n_train = sig_train.shape[1]
        total_possible_comparisons_train = n_train * (n_train - 1) / 2
        total_duplicates_train = count_duplicates_for_subset(data_train)

        n_test = sig_test.shape[1]
        total_possible_comparisons_test = n_test * (n_test - 1) / 2
        total_duplicates_test = count_duplicates_for_subset(data_test)

        #r loop for this bootstrap
        for r in range(1, n_hashes + 1):
            print(f"    r = {r}/{n_hashes}")
            if n_hashes % r != 0:  # n = b * r needs to hold
                continue

            #LSH on trainset
            t0 = time.time()
            candidate_pairs_train = lsh_candidatepairs(sig_train, r)
            t1 = time.time()

            best_epsilon = None
            best_F1_train = -1.0

            if len(candidate_pairs_train) > 0:
                #make dissim matrix once and reuse for all epsilons
                dissim_train = build_dissimilarity_matrix(
                    candidate_pairs=candidate_pairs_train,
                    data=data_train,
                    gamma=0.7,
                    q=3,
                    alpha=0.602,
                    beta=0.0,
                    delta=0.5,
                    mu=0.65,
                    approx_thresh=0.5
                )
                t2 = time.time()

                for eps in epsilon_grid:
                    clusters_train = msm_clustering(dissimilarity_matrix=dissim_train, epsilon=eps)
                    msm_metrics_train = msm_performance(
                        clusters=clusters_train,
                        candidate_pairs=candidate_pairs_train,
                        data=data_train,
                        total_possible_comparisons=total_possible_comparisons_train,
                        total_duplicates=total_duplicates_train
                    )
                    F1_train = msm_metrics_train["F1"]

                    if F1_train > best_F1_train:
                        best_F1_train = F1_train
                        best_epsilon = eps

                print(f"best epsilon for r={r}: {best_epsilon:.2f} (F1={best_F1_train:.3f})")


            else:
                t2 = time.time()
                best_epsilon = 0.5
###########################
            #LSH on testset
            t3 = time.time()
            candidate_pairs_test = lsh_candidatepairs(sig_test, r)
            t4 = time.time()
            print(f"test cand pairs generated: {len(candidate_pairs_test)}")

            metrics = lsh_performance(
                candidate_pairs=candidate_pairs_test,
                r=r,
                signature_matrix=sig_test,
                total_possible_comparisons=total_possible_comparisons_test,
                total_duplicates=total_duplicates_test,
                data=data_test
            )
            print("test LSH performance computed")

            #Now MSM on testset
            print("test dissim MSM")
            if len(candidate_pairs_test) > 0:
                dissim_test = build_dissimilarity_matrix(
                    candidate_pairs=candidate_pairs_test,
                    data=data_test,
                    gamma=0.7,
                    q=3,
                    alpha=0.602,
                    beta=0.0,
                    delta=0.5,
                    mu=0.65,
                    approx_thresh=0.5
                )
                t5 = time.time()

                #save timings to see how long running takes
                timings.append({
                    "bootstrap": bootstrap,
                    "r": r,
                    "time_candpairs_train": t1 - t0,
                    "time_dissim_train": t2 - t1,
                    "time_candpairs_test": t4 - t3,
                    "time_dissim_test": t5 - t4,
                })

                # save dissim matrix for debugging
                ids = data_test.index.astype(str)
                dissim_df = pd.DataFrame(dissim_test, index=ids, columns=ids)
                dissim_matrices_test[(bootstrap, r)] = dissim_df

                print("test MSM clustering")
                clusters_test = msm_clustering(dissimilarity_matrix=dissim_test, epsilon=best_epsilon)
                print("test MSM performance")
                msm_metrics_test = msm_performance(
                    clusters=clusters_test,
                    candidate_pairs=candidate_pairs_test,
                    data=data_test,
                    total_possible_comparisons=total_possible_comparisons_test,
                    total_duplicates=total_duplicates_test
                )

                metrics["PQ_msm"] = msm_metrics_test["PQ"]
                metrics["PC_msm"] = msm_metrics_test["PC"]
                metrics["F1_msm"] = msm_metrics_test["F1"]
            else:
                print("no candidate pairs, MSM skipped")
                metrics["PQ_msm"] = 0.0
                metrics["PC_msm"] = 0.0
                metrics["F1_msm"] = 0.0

            #extra info
            metrics["bootstrap"] = bootstrap
            metrics["r"] = r
            metrics["num_candidate_pairs"] = len(candidate_pairs_test)
            metrics["epsilon"] = best_epsilon

            all_results.append(metrics)

    end_time = time.time()
    print(f"Total elapsed time: {end_time - start_time:.2f} seconds")

    results_df = pd.DataFrame(all_results)
    timings_df = pd.DataFrame(timings)


    return results_df, timings_df, dissim_matrices_test
