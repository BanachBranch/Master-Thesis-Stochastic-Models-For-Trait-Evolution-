import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.stats import truncnorm

import Individual_Model as im


def truncated_norm(mean, std, low, high, size=1):
    """
    Generates normally distributed values between low and high.
    """

    return truncnorm.rvs(
        (low - mean) / std, (high - mean) / std, loc=mean, scale=std, size=size
    )


def run(sigma_alpha, sigma_mutation, seed):
    # Parameters

    # seed = 15
    K = 1000  # Effective Population size

    sigma_birth = 0.9  # Birth width
    # sigma_alpha = 0.9       # Competition width
    # sigma_mutation = 0.05    # STD of mutation offset h

    mutation_prob = 0.1  # Probability that mutation occurs
    uK = 1  # Mutation scaling

    T_max = 1500  # simulation time
    t_incr = 0.5  # Time between recordings

    identification = (
        f"PPChampExample__K{K}_"
        f"uK{uK}_"
        f"b{sigma_birth}_"
        f"d{0}_"
        f"a{sigma_alpha}_"
        f"mp{mutation_prob}_"
        f"mo{sigma_mutation}_"
        f"Tmax{T_max}"
        f"t_incr{t_incr}"
        f"seed{seed}"
    )

    # Run Individual Model

    model = im.IndividualModel(
        K,
        uK,
        seed,
        identification,
        T_max,
        t_incr,
        birth_rate=lambda x: np.exp(-(x**2) / (2 * (sigma_birth**2))),
        death_rate=lambda x: 0,
        alpha=lambda x, y: np.exp(-((x - y) ** 2) / (2 * (sigma_alpha**2))),
        mutation_prob=lambda x: mutation_prob,
        mutation_offset=lambda x: x
        + truncated_norm(mean=0, std=sigma_mutation, low=-2 + x, high=2 - x),
    )

    model.gillespie(verbose=True)
    model.plot_population()
    model.save_instance()


if __name__ == "__main__":
    start = time.perf_counter()

    # Parameters
    seeds = [30, 31, 30, 31, 30, 31]
    sigma_alphas = [0.7, 0.7, 0.6, 0.6, 1.0, 1.0]
    sigma_mutations = [0.025, 0.01, 0.025, 0.01, 0.025, 0.01]

    # Multiprocessing
    with ProcessPoolExecutor() as ex:
        list(ex.map(run, sigma_alphas, sigma_mutations, seeds))

    # PRINT ====
    end = time.perf_counter()
    minutes, seconds = divmod(end - start, 60)
    print(f"{int(minutes)} minutes and {int(seconds)} seconds.")
