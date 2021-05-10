# -*- coding: utf-8 -*-
"""
@author: "Dickson Owuor"
@credits: "Thomas Runkler, Edmond Menya, and Anne Laurent,"
@license: "MIT"
@version: "3.0"
@email: "owuordickson@gmail.com"
@created: "12 July 2019"
@modified: "10 May 2021"

Breath-First Search for gradual patterns (ACO-GRAANK)

"""

import numpy as np
from numpy import random as rand
from ypstruct import structure

from .shared.gp import GI, GP
from .shared.dataset_bfs import Dataset
from .shared.profile import Profile
from .shared import config as cfg


class GradACO:

    def __init__(self, f_path, min_supp, eq):
        self.d_set = Dataset(f_path, min_supp, eq)
        self.d_set.init_gp_attributes()
        self.attr_index = self.d_set.attr_cols
        self.e_factor = cfg.EVAPORATION_FACTOR  # evaporation factor
        self.max_it = cfg.MAX_ITERATIONS
        self.iteration_count = 0
        self.p_matrix = np.ones((self.d_set.col_count, 3), dtype=float)

    def update_pheromones(self, pattern):
        lst_attr = []
        for obj in pattern.gradual_items:
            # print(obj.attribute_col)
            attr = obj.attribute_col
            symbol = obj.symbol
            lst_attr.append(attr)
            i = attr
            if symbol == '+':
                self.p_matrix[i][0] += 1
            elif symbol == '-':
                self.p_matrix[i][1] += 1
        for index in self.attr_index:
            if int(index) not in lst_attr:
                i = int(index)
                self.p_matrix[i][2] += 1

    def evaporate_pheromone(self, pattern):
        lst_attr = []
        for obj in pattern.gradual_items:
            # print(obj.attribute_col)
            attr = obj.attribute_col
            symbol = obj.symbol
            lst_attr.append(attr)
            i = attr
            if symbol == '+':
                self.p_matrix[i][0] = (1 - self.e_factor) * self.p_matrix[i][0]
            elif symbol == '-':
                self.p_matrix[i][1] = (1 - self.e_factor) * self.p_matrix[i][1]

    def run_ant_colony(self):
        min_supp = self.d_set.thd_supp
        winner_gps = list()  # subsets
        loser_gps = list()  # supersets
        repeated = 0
        it_count = 0
        max_it = self.max_it

        if self.d_set.no_bins:
            return []

        # Best Cost of Iteration
        best_cost_arr = np.empty(max_it)
        best_cost = np.inf
        str_plt = ''

        # Iterations for ACO
        while it_count < max_it:
        # while repeated < 1:
            rand_gp = self.generate_aco_gp()
            if len(rand_gp.gradual_items) > 1:
                # print(rand_gp.get_pattern())
                exits = GradACO.is_duplicate(rand_gp, winner_gps, loser_gps)
                if not exits:
                    repeated = 0
                    # check for anti-monotony
                    is_super = GradACO.check_anti_monotony(loser_gps, rand_gp, subset=False)
                    is_sub = GradACO.check_anti_monotony(winner_gps, rand_gp, subset=True)
                    if is_super or is_sub:
                        continue
                    gen_gp = self.validate_gp(rand_gp)
                    is_present = GradACO.is_duplicate(gen_gp, winner_gps, loser_gps)
                    is_sub = GradACO.check_anti_monotony(winner_gps, gen_gp, subset=True)
                    if is_present or is_sub:
                        repeated += 1
                    else:
                        if gen_gp.support >= min_supp:
                            self.update_pheromones(gen_gp)
                            winner_gps.append(gen_gp)
                            best_cost = round((1 / gen_gp.support), 2)
                        else:
                            loser_gps.append(gen_gp)
                            # update pheromone as irrelevant with loss_sols
                            self.evaporate_pheromone(gen_gp)
                    if set(gen_gp.get_pattern()) != set(rand_gp.get_pattern()):
                        loser_gps.append(rand_gp)
                else:
                    repeated += 1

            # Show Iteration Information
            best_cost_arr[it_count] = best_cost
            str_plt += "Iteration {}: Best Cost: {} \n".format(it_count, best_cost)
            it_count += 1

        # Output
        out = structure()
        out.best_costs = best_cost_arr
        out.best_patterns = winner_gps
        out.iterations = str_plt

        self.iteration_count = it_count
        return out
        # return winner_gps

    def generate_aco_gp(self):
        p = self.p_matrix
        n = len(self.attr_index)
        pattern = GP()
        attrs = np.random.permutation(n)
        for i in attrs:
            max_extreme = n * 100
            x = float(rand.randint(1, max_extreme) / max_extreme)
            pos = float(p[i][0] / (p[i][0] + p[i][1] + p[i][2]))
            neg = float((p[i][0] + p[i][1]) / (p[i][0] + p[i][1] + p[i][2]))
            if x < pos:
                temp = GI(self.attr_index[i], '+')
            elif (x >= pos) and (x < neg):
                temp = GI(self.attr_index[i], '-')
            else:
                # temp = GI(self.attr_index[i], 'x')
                continue
            pattern.add_gradual_item(temp)
        return pattern

    def validate_gp(self, pattern):
        # pattern = [('2', '+'), ('4', '+')]
        min_supp = self.d_set.thd_supp
        gen_pattern = GP()
        bin_data = np.array([])

        for gi in pattern.gradual_items:
            # if self.d_set.invalid_bins.size > 0 and np.any(np.isin(self.d_set.invalid_bins, gi.gradual_item)):
            #    continue
            # else:
            arg = np.argwhere(np.isin(self.d_set.valid_bins[:, 0], gi.gradual_item))
            if len(arg) > 0:
                i = arg[0][0]
                bin_obj = self.d_set.valid_bins[i]
                if bin_data.size <= 0:
                    bin_data = np.array([bin_obj[1], bin_obj[1]])
                    gen_pattern.add_gradual_item(gi)
                else:
                    bin_data[1] = bin_obj[1]
                    temp_bin, supp = self.bin_and(bin_data, self.d_set.attr_size)
                    if supp >= min_supp:
                        bin_data[0] = temp_bin
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
    def is_duplicate(pattern, lst_winners, lst_losers):
        for pat in lst_losers:
            if set(pattern.get_pattern()) == set(pat.get_pattern()) or \
                    set(pattern.inv_pattern()) == set(pat.get_pattern()):
                return True
        for pat in lst_winners:
            if set(pattern.get_pattern()) == set(pat.get_pattern()) or \
                    set(pattern.inv_pattern()) == set(pat.get_pattern()):
                return True
        return False

    @staticmethod
    def bin_and(bins, n):
        # bin_ = np.zeros((n, n), dtype=bool)
        temp_bin = bins[0] * bins[1]
        supp = float(np.sum(temp_bin)) / float(n * (n - 1.0) / 2.0)
        return temp_bin, supp


def init(f_path, min_supp, cores, eq=False):
    try:
        if cores > 1:
            num_cores = cores
        else:
            num_cores = Profile.get_num_cores()

        ac = GradACO(f_path, min_supp, eq)
        # list_gp = ac.run_ant_colony()
        out = ac.run_ant_colony()
        list_gp = out.best_patterns

        wr_line = "Algorithm: ACO-GRAANK (v2.0)\n"
        wr_line += "No. of (dataset) attributes: " + str(ac.d_set.col_count) + '\n'
        wr_line += "No. of (dataset) tuples: " + str(ac.d_set.row_count) + '\n'
        wr_line += "Evaporation factor: " + str(ac.e_factor) + '\n'

        wr_line += "Minimum support: " + str(min_supp) + '\n'
        wr_line += "Number of cores: " + str(num_cores) + '\n'
        wr_line += "Number of patterns: " + str(len(list_gp)) + '\n'
        wr_line += "Number of iterations: " + str(ac.iteration_count) + '\n\n'

        for txt in ac.d_set.titles:
            try:
                wr_line += (str(txt.key) + '. ' + str(txt.value.decode()) + '\n')
            except AttributeError:
                wr_line += (str(txt[0]) + '. ' + str(txt[1].decode()) + '\n')

        wr_line += str("\nFile: " + f_path + '\n')
        wr_line += str("\nPattern : Support" + '\n')

        for gp in list_gp:
            wr_line += (str(gp.to_string()) + ' : ' + str(gp.support) + '\n')

        # wr_line += "\nPheromone Matrix\n"
        # wr_line += str(ac.p_matrix)
        wr_line += '\n\nIterations \n'
        wr_line += out.iterations
        return wr_line
    except ArithmeticError as error:
        wr_line = "Failed: " + str(error)
        print(error)
        return wr_line
