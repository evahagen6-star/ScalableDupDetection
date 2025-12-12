import math
import re
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from itertools import combinations

from msm_helpers import msm_similarity


#helpers for same shop and same brand
def brand_name(idx, df):
    fmap = df.iloc[idx]["featuresMap"]
    if not isinstance(fmap, dict):
        return None
    for key in ["Brand", "Brand Name"]:
        if key in fmap and fmap[key] not in (None, ""):
            return str(fmap[key]).strip().lower()
    return None
#True if brands are different
def diff_brand(idx_i, idx_j, df):

    b_i = brand_name(idx_i, df)
    b_j = brand_name(idx_j, df)

    if b_i is None or b_j is None:
        return False #if you dont know one brand, return false

    return b_i != b_j

#shops
def same_shop(idx_i, idx_j, df):
    shop_i = str(df.iloc[idx_i]["shop"]).strip().lower()
    shop_j = str(df.iloc[idx_j]["shop"]).strip().lower()
    return shop_i == shop_j


#Dissimilarity matrix
def build_dissimilarity_matrix(candidate_pairs,
                               data,
                               gamma=0.7,
                               q=3,
                               alpha=0.6,
                               beta=0.0,
                               delta=0.5,
                               mu=0.65,
                               approx_thresh=0.5):

    n = len(data)
    infinity = 1e9

    #initialize with with all distances = infinity
    dissim = np.full((n, n), infinity, dtype=float)
    # distance to self is 0
    np.fill_diagonal(dissim, 0.0)

    for (i, j) in candidate_pairs:
        if i == j:
            continue
        if i > j:
            i, j = j, i #no double pairs

        #for same shop or different brand set distance is infinity
        if same_shop(i, j, data) or diff_brand(i, j, data):
            dist_ij = infinity

        else:
            hSim = msm_similarity(i,
                                  j,
                                  data,
                                  gamma=gamma,
                                  q=q,
                                  alpha=alpha,
                                  beta=beta,
                                  delta=delta,
                                  mu=mu,
                                  approx_thresh=approx_thresh)
            dist_ij = 1.0 - hSim #distance is 1-sim

        dissim[i, j] = dist_ij
        dissim[j, i] = dist_ij

    return dissim


#clustering using the obtained dissim matrix
def msm_clustering(dissimilarity_matrix, epsilon):

    #complete linkage
    clustermodel = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="complete",
        distance_threshold=epsilon
    )
    cluster_labels = clustermodel.fit_predict(dissimilarity_matrix) #assigns all products to a cluster label

    #dictionary with for evey cluster all products in the cluster
    clusters = {}
    for idx, c in enumerate(cluster_labels):
        if c not in clusters:
            clusters[c] = []
        clusters[c].append(idx)

    return clusters



#performance after msm
def msm_performance(clusters,candidate_pairs,data,total_possible_comparisons,total_duplicates):
    #store all pairs of products placed in same cluster in this set
    predicted_pairs = set()

    for idx_list in clusters.values(): #cluster.values gives all lists with indices
        if len(idx_list) < 2: #if only 1 in cluster, no pair
            continue
        for i, j in combinations(idx_list, 2): #all unique combinations of indices in the cluster
            if i > j:
                i, j = j, i
            predicted_pairs.add((i, j)) #save pairs as small index, large index, so no double pairs

    #Count TP and FP based on modelID
    TP = 0
    FP = 0

    for i, j in predicted_pairs:
        if data.loc[i, "modelID"] == data.loc[j, "modelID"]:
            TP += 1
        else:
            FP += 1

    FN = total_duplicates - TP
    if FN < 0:
        FN = 0

    total_non_duplicates = total_possible_comparisons - total_duplicates
    TN = total_non_duplicates - FP
    if TN < 0:
        TN = 0

    comparisons_made = len(candidate_pairs)

    if TP + FP > 0:
        PQ = TP / (TP + FP) #precision
    else:
        PQ = 0.0

    if TP + FN > 0:
        PC = TP / (TP + FN) #recall
    else:
        PC = 0.0

    if PQ + PC > 0:
        F1 = 2 * PQ * PC / (PQ + PC)
    else:
        F1 = 0.0

    # fraction comparisons
    fraction = comparisons_made / total_possible_comparisons

    return {
        "PQ": PQ,
        "PC": PC,
        "F1": F1,
        "fraction_comparisons": fraction,
        "TP": TP,
        "FP": FP,
        "TN": TN,
        "FN": FN
    }

