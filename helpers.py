def count_possible_comps(n_products_subset):
    if n_products_subset < 2:
        return 0
    return n_products_subset * (n_products_subset - 1) // 2

#Calcs the number of true duplicate pairs within a dataset based on modelID
def count_duplicates_for_subset(data_subset):
    dups_per_modelID = data_subset['modelID'].value_counts()
    total_duplicates = 0
    for group_size in dups_per_modelID: # For each group size = number of products with same modelID
        if group_size > 1:
            duplicate_pairs_in_group = group_size * (group_size - 1) // 2
            total_duplicates += duplicate_pairs_in_group  #add pairs from this group to the total count
    return total_duplicates