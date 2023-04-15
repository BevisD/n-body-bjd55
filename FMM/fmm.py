import numpy as np
from scipy.special import binom
from particle import Particle
from index import Index
from numpy.typing import NDArray
from typing import List


class FMM:
    def __init__(self, precision: int, max_level: int, particles: List[Particle]) -> None:
        self.precision = precision
        self.max_level = max_level
        self.particles = particles
        self.box_positions = self.generate_box_positions()
        self.multi_expansion_arrays = self.generate_expansion_arrays()
        self.local_expansion_arrays = self.generate_expansion_arrays()
        self.particle_clusters = self.cluster_particles(self.max_level)

    def generate_box_positions(self) -> List[NDArray]:
        box_positions = [np.zeros((2 ** i, 2 ** i), dtype=complex)
                         for i in range(self.max_level + 1)]

        for level, array in enumerate(box_positions):
            half_width = 2 ** (-level - 1)
            coords = np.arange(half_width, 1, 2 * half_width)

            X, Y = np.meshgrid(coords, coords)
            array[...] = X + 1.0j * Y

        return box_positions

    def generate_expansion_arrays(self) -> List[NDArray]:
        expansion_arrays = [np.zeros((2 ** i, 2 ** i, self.precision + 1),
                                     dtype=complex) for i in range(self.max_level + 1)]

        return expansion_arrays

    def calculate_mutipoles(self, level: int) -> None:
        for particle in self.particles:
            index = particle.index(level).index
            z_0 = particle.centre - self.box_positions[level][index]
            k_vals = np.arange(1, self.precision + 1)

            a_vals = np.zeros(self.precision + 1, dtype=complex)
            a_vals[0] = particle.charge
            a_vals[1:] = -particle.charge * (z_0 ** k_vals) / k_vals

            self.multi_expansion_arrays[level][index] += a_vals

    def multi_to_multi(self, child_level: int) -> None:
        child_arr = self.multi_expansion_arrays[child_level]
        parent_arr = self.multi_expansion_arrays[child_level - 1]

        max_index = parent_arr.shape[0]
        for i in range(max_index):
            for j in range(max_index):
                parent_index = Index(i, j, child_level - 1)
                for child in parent_index.children():
                    b = np.zeros(self.precision + 1, dtype=complex)
                    a = child_arr[child.index]

                    z_0 = self.box_positions[child_level][child.index] \
                          - self.box_positions[child_level-1][parent_index.index]

                    b[0] = a[0]
                    L = np.arange(1, self.precision + 1)
                    k = np.arange(1, self.precision + 1)

                    b[1:] = -a[0] * z_0 ** L / L
                    binom_terms = binom(L - 1, k[:, np.newaxis] - 1)
                    z_terms = z_0 ** (L - k[:, np.newaxis])
                    b[1:] += np.sum(binom_terms * a[1:, np.newaxis] * z_terms, axis=0)
                    parent_arr[i, j] += b

    def multi_to_local(self, level):
        multi_expansion_array = self.multi_expansion_arrays[level]
        # noinspection DuplicatedCode
        max_index = multi_expansion_array.shape[0]
        k = np.arange(1, self.precision + 1)
        L = np.arange(1, self.precision + 1)

        for i in range(max_index):
            for j in range(max_index):
                index = Index(i, j, level)
                local_expansion_coefficients = self.local_expansion_arrays[level][i, j]
                well_seperated = index.well_seperated()
                for interactor in well_seperated:
                    b = multi_expansion_array[interactor.index]
                    c = np.zeros(self.precision + 1, dtype=complex)

                    z_0 = self.box_positions[level][interactor.index] \
                          - self.box_positions[level][index.index]
                    signs = np.ones(self.precision, dtype=int)
                    signs[::2] = -1

                    c[0] = b[0] * np.log(-z_0) + np.sum(signs * b[1:] / z_0**k)

                    binom_factors = binom(L[:, np.newaxis] + k-1, k-1)
                    c[1:] = -b[0] / (L * z_0**L)
                    c[1:] += 1 / z_0**L * np.sum(binom_factors * signs * b[1:] / z_0**k, axis=1)
                    local_expansion_coefficients += c

    def local_to_local(self, parent_level):
        parent_arr = self.local_expansion_arrays[parent_level]
        child_arr = self.local_expansion_arrays[parent_level + 1]

        L = np.arange(self.precision + 1)
        k = np.arange(self.precision + 1)

        # noinspection DuplicatedCode
        max_index = parent_arr.shape[0]
        for i in range(max_index):
            for j in range(max_index):
                parent_index = Index(i, j, parent_level)
                for child in parent_index.children():
                    c = parent_arr[parent_index.index]

                    z_0 = self.box_positions[parent_level + 1][child.index] \
                          - self.box_positions[parent_level][parent_index.index]

                    d = np.sum(c * binom(k, L[:, np.newaxis]) *
                               z_0 ** (k - L[:, np.newaxis]), axis=1)
                    child_arr[child.index] = d

    def calculate_potentials(self) -> None:
        local_expansion = self.local_expansion_arrays[self.max_level]
        L = np.arange(self.precision+1)

        for i, particle in enumerate(self.particles):
            potential = 0
            current = particle.index(self.max_level)
            neighbours = current.neighbours()

            z = particle.centre \
                - self.box_positions[self.max_level][current.index]
            d_vals = local_expansion[current.index]
            potential += -d_vals @ z**L

            indices = self.particle_clusters[current.index]
            indices.remove(i)
            for cell in neighbours:
                indices = indices.union(self.particle_clusters[cell.index])

            for index in indices:
                other = self.particles[index]
                z = particle.centre - other.centre
                potential += -other.charge * np.log(z)

            particle.potential = potential.real

    def cluster_particles(self, level):
        clusters = np.empty((2**level, 2**level), dtype=object)
        for x in range(2**level):
            for y in range(2**level):
                clusters[x, y] = set()

        for i, particle in enumerate(self.particles):
            index = particle.index(level).index
            clusters[index].add(i)
        return clusters

    def upward_pass(self):
        self.calculate_mutipoles(self.max_level)
        for level in range(self.max_level, 0, -1):
            self.multi_to_multi(level)

    def downward_pass(self):
        for level in range(2, self.max_level):
            self.multi_to_local(level)
            self.local_to_local(level)
        self.multi_to_local(self.max_level)

    def fmm_algorithm(self):
        self.upward_pass()
        self.downward_pass()
        # TODO: calculate forces at each point

    def calculate_exact_potentials(self):
        for i, particle_1 in enumerate(self.particles):
            particle_1.potential = 0
            for particle_2 in self.particles[:i] + self.particles[i+1:]:
                z = particle_1.centre - particle_2.centre
                particle_1.potential += - particle_2.charge * np.log(z).real
