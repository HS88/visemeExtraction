from numba import jit, prange
import numpy as np

# closing sieve, minima extrema processing
@jit(nopython=True)
def o_sieve(subset_minima: np.ndarray, r: int, x: int) -> float:
    # start bound is starting point of first interval
    start_bound = 0 if x - r < -1 else x - r + 1
    # end bound is *one after* starting point of last interval
    end_bound = 4800 - r + 1 if x + r > 4800 else x + 1

    x_minimum = subset_minima[start_bound]
    for i in range(1, end_bound - start_bound):
        x_minimum = min(x_minimum, subset_minima[start_bound + i])
    return x_minimum

# one iteration of the multiscale analysis
@jit(nopython=True, parallel=True)
def multiscale_step(r: int, initial: np.ndarray) -> np.ndarray:
    # pre-compute ALL subset minima
    end_bound = 4800 - r + 1
    subset_minima = np.empty(end_bound)
    for i in prange(4800 - r + 1):
        subset_minima[i] = np.min(initial[i:i + r])

    # perform sieve filter
    post = np.empty(4800)
    for i in prange(4800):
        post[i] = o_sieve(subset_minima, r, i)
    return post

# full multiscale analysis
@jit(nopython=True)
def multiscale_full(initial: np.ndarray) -> np.ndarray:
    filters = np.empty((61, 4800))
    filters[0] = initial
    for i in range(1, 61):
        filters[i] = multiscale_step(i+1, initial)
    return filters


# if __name__ == "__main__":
#     filters = np.empty(61, 40000)
#     filters[0] = 0  # img.flatten('F')
#     for index in range(1, 61):
#         filters[index] = multiscale_step(index + 1, filters[index - 1])
#     differences = filters[1:] - filters[:-1]
#     sums = np.sum(differences, 1)
