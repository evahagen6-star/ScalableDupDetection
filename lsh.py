from collections import defaultdict

def lsh_candidatepairs(signature_matrix, r):
    n = signature_matrix.shape[0]  #number of rows of signature matrix/nr of hashfunctions
    if n % r != 0:  # check whether n is divisble by r, because n = b*r has to hold!!
        return None

    b = n // r
    # calc threshold t
    t = (1 / b) ** (1 / r)  # (1/b)^(1/r), not direct used, but for own info to store

    candidate_pairs = set()  #unique pairs

    # Divide into b bands
    for band in range(b):
        #bucket structure for this band, automatically starts a new bucket/list when sees a new hashkey
        buckets = defaultdict(list) #
        start_row = band * r
        end_row = start_row + r

        #hash every column in this band
        for col in range(signature_matrix.shape[1]):
            #the r signature values of the band are the hashkey band_vector
            band_vector = tuple(signature_matrix[start_row:end_row, col])
            #same hashkey goes to same bucket
            buckets[band_vector].append(col)

        #add all pairs from buckets with size > 1
        for bucket in buckets.values():
            if len(bucket) > 1:
                for i in range(len(bucket)):
                    for j in range(i + 1, len(bucket)):
                        candidate_pairs.add((bucket[i], bucket[j]))

    return candidate_pairs



def lsh_performance(candidate_pairs,r,signature_matrix,total_possible_comparisons,total_duplicates,data):

    n = signature_matrix.shape[0]
    b = n // r #needs to hold, already checked that it is feasible
    t = (1 / b) ** (1 / r)

    #TP, FP, TN, FN
    TP = 0
    FP = 0

    for i, j in candidate_pairs:
        if data.loc[i, "modelID"] == data.loc[j, "modelID"]:
            TP += 1
        else:
            FP += 1

    FN = total_duplicates - TP
    if FN < 0:
        FN = 0

    total_non_duplicates = total_possible_comparisons - total_duplicates
    TN = total_non_duplicates - FP #total negatives = TN + FP, so TN = total negatives - FP
    if TN < 0:
        TN = 0

    #nr comparisons LSH does
    comparisons_made = len(candidate_pairs)

    # PQ
    PQ = TP / comparisons_made if comparisons_made > 0 else 0.0

    # PC
    PC = TP / total_duplicates if total_duplicates > 0 else 0.0

    # F1*
    if PQ + PC > 0:
        F1 = 2 * PQ * PC / (PQ + PC)
    else:
        F1 = 0.0

    # fraction comparisons
    fraction = comparisons_made / total_possible_comparisons

    return {
        "b": b,
        "t": t,
        "PQ": PQ,
        "PC": PC,
        "F1": F1,
        "fraction_comparisons": fraction,
        "TP": TP,
        "FP": FP,
        "TN": TN,
        "FN": FN
    }




