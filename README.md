**OVERVIEW**

This project contains a scalable method for detecting duplicate TV products collected from multiple webshops. The code is divided into separate modules, each handling a specific part: data cleaning, constructing binary matrices, MinHashing, LSH, similarity computation, clustering, and bootstrap-based evaluation.
The core idea is to compare a benchmark implementation of MSMP+ with an extended version MSMP++ with a different product representation.
All components are imported and brought together in the main file, where the full method runs from start to finish.

**FUNCTIONS**

- **dataclean.py**
contains a function to clean product titles and feature values for all TVs.

- **mw.py**
extracts model words and builds a binary matrix following the original MSMP+ method (used as a benchmark).
- **mw_extension.py**
modified version of the binary matrix construction: MSMP++

- **minhash.py**
generates a signature matrix from any binary matrix using minhashing.

- **lsh.py**
applies Locality Sensitive Hashing to the signature matrix to generate candidate duplicate pairs.

- **helpers.py**
helper functions for counting true duplicates in subsets of the dataset.

- **bootstrap_lsh.py**
runs 5 bootstraps of LSH, evaluates the generated candidates, and computes Pair Quality (PQ), Pair Completeness (PC), and F1* scores.

- **msm_helpers.py**
contains all individual components of the msm similarity function, and finally the full similarity function

- **msm.py**
uses the similarity function to build a dissimilarity matrix and perform clustering.

- **bootstrap_msm_tune.py**
performs 5 bootstrap runs where	LSH generates candidate pairs to be evaluated by MSM and a train test split is used. The train parts are used for hyperparameter tuning based on F1, and test sets for evaluation and computation of final F1 scores.


**MAIN**

The main script runs the full method step by step:
1.	Data cleaning: titles and feature values are cleaned using dataclean.
2.	Binary matrix construction for the MSMP+ benchmark (via mw) and our MSMP++ extension (via mw_extension)
3.	MinHashing: binary matrices are converted to signature matrices (minhash).
4.	LSH is applied to generate candidate pairs, repeated for 5 bootstraps (bootstrap_lsh).
PC, PQ, and F1* curves are generated.
5.	MSM: for 5 boostraps, clustering is applied to the candidate pairs using bootstrap_msm_tune, and F1 curves are plotted.
