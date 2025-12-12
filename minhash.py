import numpy as np
from sympy import nextprime

#number of pos divisors of n
def num_divisors(n):
    cnt = 0
    d = 1
    while d * d <= n:
        if n % d == 0:
            cnt += 1
            if d * d != n:
                cnt += 1
        d += 1
    return cnt

#want to make sure number of hashes is well divisble, so we get enough datapoints
def choose_n_hashes(og_rows: int,
                    min_divisors: int = 6,
                    max_delta: int = 30):
    #choose n_hashes as approx half the number of og rows, but a nearby number with at least 6 divisors
    target = og_rows // 2

    for delta in range(max_delta + 1):
        for candidate in (target - delta, target + delta):
            if candidate < 2 or candidate > og_rows:
                continue
            if num_divisors(candidate) >= min_divisors:
                return candidate
    #if nothing found, then just use half of og rows
    return max(2, target)


#Signature matrix from binary matrix
def compute_minhash_signature(binary_matrix):
    num_rows, num_cols = binary_matrix.shape

    og_rows = binary_matrix.shape[0]
    #nr_hashes = og_rows // 2  # we choose the nr of hashfunctions to be half og size, // gives int
    nr_hashes = choose_n_hashes(og_rows, min_divisors=6, max_delta=30)
    print(f"Using {nr_hashes} hashes (target half={og_rows//2})")
    # B as first prime number larger than nr rows of input matrix, to make sure no collisions for the permutations
    c = nextprime(og_rows)

    # Create random coefficients for the hashfunctions
    a = np.random.randint(1, c, size=nr_hashes)
    b = np.random.randint(0, c, size=nr_hashes)

    # initialize sign matrix with rows=nr hashfunctions, cols=cols og matrix, and all values initalized at infinity
    signature_matrix = np.full((nr_hashes, num_cols), np.inf)

    def hash_functions(x):
        hash_values = []
        for i in range(nr_hashes):
            hash_value = (a[i] * x + b[i]) % c
            hash_values.append(hash_value)
        return np.array(hash_values)

    for r in range(num_rows):
        h_r = hash_functions(r)  #hashvalue/position in virtual perm for row r
        cols_with_1 = np.where(binary_matrix[r] == 1)[0] #cols/tvs where mw of row r has 1

        #only update the cols with a 1
        for col in cols_with_1:
            for i in range(nr_hashes):
                if h_r[i] < signature_matrix[i, col]:
                    signature_matrix[i, col] = h_r[i]
    return signature_matrix