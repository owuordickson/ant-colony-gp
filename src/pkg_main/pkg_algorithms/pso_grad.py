# -*- coding: utf-8 -*-
"""
@author: "Dickson Owuor"
@credits: "Thomas Runkler, Edmond Menya, and Anne Laurent,"
@license: "MIT"
@version: "1.0"
@email: "owuordickson@gmail.com"
@created: "29 April 2021"
@modified: "29 April 2021"

Breath-First Search for gradual patterns (PSO-GRAANK)

"""
import numpy as np
import random
from ypstruct import structure

from .shared.gp import GI, GP
from .shared.dataset_bfs import Dataset
from .shared.profile import Profile
from .shared import config as cfg


class GradPSO:

    def __init__(self, f_path, min_supp):
        self.d_set = Dataset(f_path, min_supp)
        self.d_set.init_gp_attributes()
        self.attr_index = self.d_set.attr_cols
        self.iteration_count = 0
        self.max_it = cfg.MAX_ITERATIONS
        self.W = cfg.VELOCITY
        self.c1 = cfg.PERSONAL_COEFF
        self.c2 = cfg.GLOBAL_COEFF
        # self.target = 1
        # self.target_error = 1e-6
        self.n_particles = cfg.N_PARTICLES
        self.attr_keys = [GI(x[0], x[1].decode()).as_string() for x in self.d_set.valid_bins[:, 0]]

    def run_particle_swarm(self):
        if self.d_set.no_bins:
            return []

        it_count = 0
        max_it = self.max_it
        n_particles = self.n_particles

        particle_position_vector = np.array([self.build_gp_gene() for _ in range(n_particles)])
        pbest_position = particle_position_vector
        pbest_fitness_value = np.array([float('inf') for _ in range(n_particles)])
        gbest_fitness_value = float('inf')
        gbest_position = particle_position_vector[0]  # np.array([float('inf'), float('inf')])

        velocity_vector = ([np.zeros((len(self.attr_keys),)) for _ in range(n_particles)])
        best_pos_arr = np.empty(max_it)
        best_pos = np.inf
        best_patterns = []
        str_plt = ''

        repeated = 0
        while it_count < max_it:
        # while repeated < 1:
            for i in range(n_particles):
                fitness_candidate = self.fitness_function(self.decode_gp(particle_position_vector[i]))

                if pbest_fitness_value[i] > fitness_candidate:
                    pbest_fitness_value[i] = fitness_candidate
                    pbest_position[i] = particle_position_vector[i]

                if gbest_fitness_value > fitness_candidate:
                    gbest_fitness_value = fitness_candidate
                    gbest_position = particle_position_vector[i]
            best_pos = self.fitness_function(self.decode_gp(gbest_position))
            # if abs(gbest_fitness_value - self.target) < self.target_error:
            #    break

            for i in range(n_particles):
                x1 = np.dot(self.W, velocity_vector[i])
                x2 = np.dot(self.c1, random.random())
                x3 = (pbest_position[i] - particle_position_vector[i])
                x4 = np.dot(self.c2, random.random())
                x5 = (gbest_position - particle_position_vector[i])
                # new_velocity = (self.W * velocity_vector[i]) + (self.c1 * random.random()) * (
                #        pbest_position[i] - particle_position_vector[i]) + (self.c2 * random.random()) * (
                #                       gbest_position - particle_position_vector[i])
                new_velocity = x1 + np.dot(x2, x3) + np.dot(x4, x5)
                new_position = new_velocity + particle_position_vector[i]
                particle_position_vector[i] = new_position

            best_gp = self.decode_gp(gbest_position)
            best_gp.support = float(1 / best_pos)
            is_present = GradPSO.is_duplicate(best_gp, best_patterns)
            is_sub = GradPSO.check_anti_monotony(best_patterns, best_gp, subset=True)
            if is_present or is_sub:
                repeated += 1
            else:
                best_patterns.append(best_gp)

            try:
                # Show Iteration Information
                best_pos_arr[it_count] = best_pos
                # print("Iteration {}: Best Position = {}".format(it_count, best_pos_arr[it_count]))
                str_plt += "Iteration {}: Best Position: {} \n".format(it_count, best_pos_arr[it_count])
            except IndexError:
                pass
            it_count += 1

        # Output
        out = structure()
        out.pop = particle_position_vector
        out.best_pos = best_pos_arr
        out.gbest_position = gbest_position
        out.best_patterns = best_patterns
        out.iterations = str_plt

        self.iteration_count = it_count
        return out

    def build_gp_gene(self):
        attr = self.attr_keys
        temp_gene = np.random.choice(a=[0, 1], size=(len(attr),))
        return temp_gene

    def decode_gp(self, gene):
        temp_gp = GP()
        if gene is None:
            return temp_gp
        for i in range(gene.size):
            gene_val = round(gene[i])
            if gene_val >= 1:
                gi = GI.parse_gi(self.attr_keys[i])
                if not temp_gp.contains_attr(gi):
                    temp_gp.add_gradual_item(gi)
        return self.validate_gp(temp_gp)

    def fitness_function(self, gp):
        if gp is None:
            return np.inf
        else:
            if gp.support <= self.d_set.thd_supp:
                return np.inf
            return round((1 / gp.support), 2)

    def validate_gp(self, pattern):
        # pattern = [('2', '+'), ('4', '+')]
        min_supp = self.d_set.thd_supp
        n = self.d_set.attr_size
        gen_pattern = GP()
        bin_arr = np.array([])

        for gi in pattern.gradual_items:
            arg = np.argwhere(np.isin(self.d_set.valid_bins[:, 0], gi.gradual_item))
            if len(arg) > 0:
                i = arg[0][0]
                valid_bin = self.d_set.valid_bins[i]
                if bin_arr.size <= 0:
                    bin_arr = np.array([valid_bin[1], valid_bin[1]])
                    gen_pattern.add_gradual_item(gi)
                else:
                    bin_arr[1] = valid_bin[1].copy()
                    temp_bin = np.multiply(bin_arr[0], bin_arr[1])
                    supp = float(np.sum(temp_bin)) / float(n * (n - 1.0) / 2.0)
                    if supp >= min_supp:
                        bin_arr[0] = temp_bin.copy()
                        gen_pattern.add_gradual_item(gi)
                        gen_pattern.set_support(supp)
        if len(gen_pattern.gradual_items) <= 1:
            return pattern
        else:
            return gen_pattern

    @staticmethod
    def check_anti_monotony(lst_p, pattern, subset=True):
        result = False
        if subset:
            for pat in lst_p:
                result1 = set(pattern.get_pattern()).issubset(set(pat.get_pattern()))
                result2 = set(pattern.inv_pattern()).issubset(set(pat.get_pattern()))
                if result1 or result2:
                    result = True
                    break
        else:
            for pat in lst_p:
                result1 = set(pattern.get_pattern()).issuperset(set(pat.get_pattern()))
                result2 = set(pattern.inv_pattern()).issuperset(set(pat.get_pattern()))
                if result1 or result2:
                    result = True
                    break
        return result

    @staticmethod
    def is_duplicate(pattern, lst_winners):
        for pat in lst_winners:
            if set(pattern.get_pattern()) == set(pat.get_pattern()) or \
                    set(pattern.inv_pattern()) == set(pat.get_pattern()):
                return True
        return False


def init(f_path, min_supp, cores):
    try:
        if cores > 1:
            num_cores = cores
        else:
            num_cores = Profile.get_num_cores()

        pso = GradPSO(f_path, min_supp)
        out = pso.run_particle_swarm()
        list_gp = out.best_patterns

        # Results
        # plt.plot(out.best_pos)
        # plt.semilogy(out.best_pos)
        # plt.xlim(0, pso.max_it)
        # plt.xlabel('Iterations')
        # plt.ylabel('Global Best Position')
        # plt.title('Pattern Swarm Algorithm (PSO)')
        # plt.grid(True)
        # plt.show()

        d_set = pso.d_set
        wr_line = "Algorithm: PSO-GRAANK (v1.0)\n"
        wr_line += "No. of (dataset) attributes: " + str(pso.d_set.col_count) + '\n'
        wr_line += "No. of (dataset) tuples: " + str(pso.d_set.row_count) + '\n'
        wr_line += "Velocity coeff.: " + str(pso.W) + '\n'
        wr_line += "C1 coeff.: " + str(pso.c1) + '\n'
        wr_line += "C2 coeff.: " + str(pso.c2) + '\n'
        wr_line += "No. of particles: " + str(pso.n_particles) + '\n'
        wr_line += "Minimum support: " + str(min_supp) + '\n'
        wr_line += "Number of cores: " + str(num_cores) + '\n'
        wr_line += "Number of patterns: " + str(len(list_gp)) + '\n'
        wr_line += "Number of iterations: " + str(pso.iteration_count) + '\n\n'

        for txt in d_set.titles:
            try:
                wr_line += (str(txt.key) + '. ' + str(txt.value.decode()) + '\n')
            except AttributeError:
                wr_line += (str(txt[0]) + '. ' + str(txt[1].decode()) + '\n')

        wr_line += str("\nFile: " + f_path + '\n')
        wr_line += str("\nPattern : Support" + '\n')

        for gp in list_gp:
            wr_line += (str(gp.to_string()) + ' : ' + str(gp.support) + '\n')

        wr_line += '\n\nIterations \n'
        wr_line += out.iterations
        return wr_line
    except ArithmeticError as error:
        wr_line = "Failed: " + str(error)
        print(error)
        return wr_line
