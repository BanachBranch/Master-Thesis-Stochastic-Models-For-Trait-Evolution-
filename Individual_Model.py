# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 16:47:37 2026

@author: Adrian
"""

import os
import time

import dill
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.stats import truncnorm


class IndividualModel:
    def __init__(
        self,
        K,
        uK,
        seed,
        identification,
        T_max,
        t_incr,
        mutation_prob,
        birth_rate,
        death_rate,
        alpha,
        mutation_offset,
    ):
        """
        Parameters
        ----------
        K : Integer
            Effective population size.
        uK : Integer
            Mutation scaling factor.
        seed : Integer
            Seed for RNG.
        identification : String
            String for file names.

        mutation_prob : Function
            Given trait x, returns probability that an offspring is a mutant.
        birth_rate : Function
            Given trait x, returns birth rate.
        death_rate : Function
            Given trait x, returns death rate.
        alpha : Function
            Given traits x and y, returns stress of x in presence of y.
        mutation_offset : Function
            Given a trait x, samples the mutation offset h of a mutant x + h.

        Returns
        -------
        None.

        """

        # PARAMETERS ========================

        self.K = K
        self.uK = uK
        self.seed = seed

        self.birth_rate = birth_rate
        self.death_rate = death_rate
        self.alpha = alpha
        self.mutation_prob = mutation_prob
        self.mutation_offset = mutation_offset

        self.id = identification

        self.T_max = T_max
        self.t_incr = t_incr

    def save_instance(self):
        """
        Saves the instance as a pickle in the folder.
        """

        os.makedirs("02 Instances", exist_ok=True)

        filepath = os.path.join("02 Instances", f"{self.id}.pkl")

        with open(filepath, "wb") as f:
            dill.dump(self, f)

    def competition_rates(self, population):
        """
        Given a population, calculates the stress rates for each individual.
        """

        x = population

        # For each pair of individuals, calculate stress factor (N x N matrix)
        X_i = x[:, None]
        X_j = x[None, :]
        A = self.alpha(X_i, X_j)

        return A.sum(axis=1) / self.K

    def gillespie(self, verbose=True):
        # START PARAMETERS ===========================

        step = 0
        t = 0.0

        times = []
        populations = []

        population = np.full(self.K, -1.0)
        np.random.seed(self.seed)

        t_checkpoint = 0

        # GILLESPIE ALGORITHM ===========================

        while t < self.T_max and len(population) > 0:
            N = len(population)

            # Rates
            birth_rates = self.birth_rate(population)
            death_rates = self.competition_rates(population) + self.death_rate(
                population
            )

            total_birth = np.sum(birth_rates)
            total_death = np.sum(death_rates)
            total_rate = total_birth + total_death

            # Next event time
            dt = np.random.exponential(1 / total_rate)
            t += dt

            # Choose event type
            if np.random.rand() < total_birth / total_rate:  # Birth event
                # Choose reproducing individual
                prob = birth_rates / total_birth
                i = np.random.choice(N, p=prob)
                x = population[i]

                if np.random.rand() < self.mutation_prob(x) * self.uK:
                    # Mutation
                    new_individual = self.mutation_offset(x)
                    population = np.append(population, new_individual)

                else:
                    # No Mutation
                    new_individual = x
                    population = np.append(population, new_individual)

            else:  # Death event
                prob = death_rates / total_death
                i = np.random.choice(N, p=prob)
                population = np.delete(population, i)

            # Record current population.
            if t_checkpoint <= t:
                if verbose:
                    print(f"\nTime {t}")
                    print(f"Individuals: {N}")
                    # print(f"Standard Deviation: {np.std(population)}")

                times.append(t)
                populations.append(population.copy())

                t_checkpoint += self.t_incr

            step += 1

        self.steps = step
        self.times = times
        self.populations = populations

    def plot_population(self):
        # Create folder
        os.makedirs("01 Plots", exist_ok=True)

        # PLOT TRAIT EVOLUTION ===============================
        plt.figure(figsize=(10, 4))

        for t, pop in zip(self.times, self.populations):
            plt.scatter([t] * len(pop), pop, s=1, alpha=0.3, color="blue")

        plt.xlabel("Time $t$")
        plt.ylabel("Trait $x$")
        plt.title("Trait distribution over time")
        plt.grid(True)

        plt.savefig(f"./01 Plots/{self.id}__TraitEvo.png", dpi=1000)

        # PLOT POPULATION SIZE ===============================
        plt.figure(figsize=(10, 4))

        sizes = [len(p) for p in self.populations]
        plt.plot(self.times, sizes)

        plt.xlabel("Time $t$")
        plt.ylabel("Population size")
        plt.title("Population size over time")
        plt.grid(True)

        plt.savefig(f"./01 Plots/{self.id}__PopSize.png", dpi=1000)

        # PLOT HEAT MAP DENSITY ===============================
        plt.figure(figsize=(10, 4))

        # Bins for each time step
        bins = np.linspace(-2, 2, 200)
        density_matrix = []

        for pop in self.populations:
            hist, _ = np.histogram(pop, bins=bins, density=True)
            density_matrix.append(hist)

        density_matrix = np.array(density_matrix)

        plt.imshow(
            density_matrix.T,
            aspect="auto",
            origin="lower",
            extent=[self.times[0], self.times[-1], bins[0], bins[-1]],
            cmap="inferno",
            norm=LogNorm(vmin=1e-4, vmax=density_matrix.max()),
        )

        plt.colorbar()
        plt.xlabel("Time $t$")
        plt.ylabel("Trait $x$")
        plt.title("Trait Density Heatmap")

        plt.savefig(f"./01 Plots/{self.id}__Density.png", dpi=1000)
        plt.show()


def plot_hist(x):
    plt.hist(x, bins=50, range=(-2, 2))
    plt.xlabel("value")
    plt.ylabel("frequency")
    plt.show()


def truncated_norm(mean, std, low, high, size=1):
    """
    Generates normally distributed values between low and high.
    """

    return truncnorm.rvs(
        (low - mean) / std, (high - mean) / std, loc=mean, scale=std, size=size
    )


if __name__ == "__main__":
    start = time.perf_counter()

    # PARAMETERS =========================

    seed = 15

    K = 100  # Effective Population size

    sigma_birth = 0.9  # Birth width
    sigma_alpha = 0.7  # Competition width
    sigma_mutation = 0.025  # STD of mutation offset h

    mutation_prob = 0.1  # Probability that mutation occurs
    uK = 1  # Mutation scaling

    T_max = 50  # simulation time
    t_incr = 0.5  # Time between recordings

    identification = (
        f"ChampExample__K{K}_"
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

    # EXAMPLES =========================

    model = IndividualModel(
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

    # PRINT ====

    end = time.perf_counter()
    minutes, seconds = divmod(end - start, 60)
    print(f"{int(minutes)} minutes and {int(seconds)} seconds.")
