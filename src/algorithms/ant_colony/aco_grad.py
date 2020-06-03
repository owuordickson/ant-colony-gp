# -*- coding: utf-8 -*-
"""
@author: "Dickson Owuor"
@credits: "Thomas Runkler, Edmond Menya, and Anne Laurent,"
@license: "MIT"
@version: "2.0"
@email: "owuordickson@gmail.com"
@created: "12 July 2019"
@modified: "12 May 2020"

"""

import numpy as np
from numpy import random as rand
# import random as rand
import matplotlib.pyplot as plt
from src.algorithms.common.fuzzy_mf import calculate_time_lag
from src.algorithms.common.gp import GI, GP, TGP
from src.algorithms.common.dataset import Dataset
#from src.algorithms.common.cython.cyt_dataset import Dataset


class GradACO:

    def __init__(self, f_path=None, min_supp=None, eq=None, d_set=None):
        if d_set is None:
            self.data = Dataset(f_path, min_sup=min_supp, eq=eq)
        else:
            self.data = d_set
        self.attr_index = self.data.attr_cols
        self.e_factor = 0.1  # evaporation factor
        self.p_matrix = np.ones((self.data.column_size, 3), dtype=float)

    def init_pheromones(self):
        invalid_pats = GP()
        for obj_gi in self.data.invalid_bins:
            gi = GI(obj_gi[0], obj_gi[1])
            invalid_pats.add_gradual_item(gi)
        self.vaporize_pheromone(invalid_pats, self.e_factor)

    def deposit_pheromone(self, pattern):
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

    def vaporize_pheromone(self, pattern, e_factor):
        lst_attr = []
        for obj in pattern.gradual_items:
            attr = obj.attribute_col
            symbol = obj.symbol
            lst_attr.append(attr)
            i = attr
            if symbol == '+':
                self.p_matrix[i][0] *= e_factor
            elif symbol == '-':
                self.p_matrix[i][1] *= e_factor

    def run_ant_colony(self, min_supp, time_diffs=None):
        if time_diffs is None:
            return self.fetch_gps(min_supp)
        else:
            return self.fetch_tgps(min_supp, time_diffs)

    def fetch_gps(self, min_supp):
        winner_gps = list()  # subsets
        loser_gps = list()  # supersets
        repeated = 0
        while repeated < 1:
            rand_gp = self.generate_rand_pattern()
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
                    gen_gp = self.validate_gp(rand_gp, min_supp)
                    is_present = GradACO.is_duplicate(gen_gp, winner_gps, loser_gps)
                    if gen_gp.support >= min_supp and is_present:
                        repeated += 1
                    elif gen_gp.support >= min_supp and not is_present:
                        winner_gps.append(gen_gp)
                        self.deposit_pheromone(gen_gp)
                    else:
                        loser_gps.append(gen_gp)
                        # update pheromone as irrelevant with loss_sols
                        # self.negate_pheromone(gen_gp)
                    if gen_gp.get_pattern() != rand_gp.get_pattern():
                        loser_gps.append(rand_gp)
                else:
                    repeated += 1
        return winner_gps

    def fetch_tgps(self, min_supp, time_diffs):
        winner_gps = list()  # subsets
        loser_gps = list()  # supersets
        repeated = 0
        while repeated < 1:
            rand_gp = self.generate_rand_pattern()
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
                    gen_gp = self.validate_tgp(rand_gp, min_supp, time_diffs)
                    is_present = GradACO.is_duplicate(gen_gp, winner_gps, loser_gps)
                    if gen_gp.support >= min_supp and is_present:
                        repeated += 1
                    elif gen_gp.support >= min_supp and not is_present:
                        winner_gps.append(gen_gp)
                        self.deposit_pheromone(gen_gp)
                    else:
                        loser_gps.append(gen_gp)
                        # update pheromone as irrelevant with loss_sols
                        # self.negate_pheromone(gen_gp)
                    if gen_gp.get_pattern() != rand_gp.get_pattern():
                        loser_gps.append(rand_gp)
                else:
                    repeated += 1
        return winner_gps

    def generate_rand_pattern(self):
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
                # temp = tuple([self.data.attr_index[i], 'x'])
                continue
            pattern.add_gradual_item(temp)
        return pattern

    def validate_gp(self, pattern, min_supp):
        # pattern = [('2', '+'), ('4', '+')]
        gen_pattern = GP()
        bin_data = np.array([])

        for obj in pattern.get_pattern():
            gi_obj = np.array([obj], dtype='i, O')
            if np.any(np.isin(self.data.invalid_bins, gi_obj)):
                continue
            else:
                # fetch pattern
                # for valid_gi in self.data.valid_gi_paths:
                #    if valid_gi[0] == gi_obj:
                arg = np.argwhere(np.isin(self.data.valid_gi_paths[:, 0], gi_obj))
                if len(arg) > 0:
                    i = arg[0][0]
                    valid_gi = self.data.valid_gi_paths[i]
                    bin_obj = self.data.get_bin(valid_gi[1])
                    if bin_data.size <= 0:
                        bin_data = np.array([bin_obj['bin'], bin_obj['bin']])
                        gi = GI(bin_obj['gi'][0], bin_obj['gi'][1])
                        gen_pattern.add_gradual_item(gi)
                    else:
                        bin_data[1] = bin_obj['bin']
                        temp_bin, supp = self.bin_and(bin_data, self.data.attr_size)
                        if supp >= min_supp:
                            bin_data[0] = temp_bin
                            gi = GI(bin_obj['gi'][0], bin_obj['gi'][1])
                            gen_pattern.add_gradual_item(gi)
                            gen_pattern.set_support(supp)
                        else:
                            bad_pattern = GP()
                            gi = GI(bin_obj['gi'][0], bin_obj['gi'][1])
                            bad_pattern.add_gradual_item(gi)
                            self.vaporize_pheromone(bad_pattern, bin_obj['support'])
                        # break
        if len(gen_pattern.gradual_items) <= 1:
            return pattern
        else:
            return gen_pattern

    def validate_tgp(self, pattern, min_supp, t_diffs):
        # pattern = [('2', '+'), ('4', '+')]
        gen_pattern = GP()
        bin_data = np.array([])

        for obj in pattern.get_pattern():
            gi_obj = np.array([obj], dtype='i, O')
            if np.any(np.isin(self.data.invalid_bins, gi_obj)):
                continue
            else:
                # fetch pattern
                # for valid_gi in self.data.valid_gi_paths:
                #    if valid_gi[0] == gi_obj:
                arg = np.argwhere(np.isin(self.data.valid_gi_paths[:, 0], gi_obj))
                if len(arg) > 0:
                    i = arg[0][0]
                    valid_gi = self.data.valid_gi_paths[i]
                    bin_obj = self.data.get_bin(valid_gi[1])
                    if bin_data.size <= 0:
                        bin_data = np.array([bin_obj['bin'], bin_obj['bin']])
                        gi = GI(bin_obj['gi'][0], bin_obj['gi'][1])
                        gen_pattern.add_gradual_item(gi)
                    else:
                        bin_data[1] = bin_obj['bin']
                        temp_bin, supp = self.bin_and(bin_data, self.data.attr_size)
                        if supp >= min_supp:
                            bin_data[0] = temp_bin
                            gi = GI(bin_obj['gi'][0], bin_obj['gi'][1])
                            gen_pattern.add_gradual_item(gi)
                            gen_pattern.set_support(supp)
                        else:
                            bad_pattern = GP()
                            gi = GI(bin_obj['gi'][0], bin_obj['gi'][1])
                            bad_pattern.add_gradual_item(gi)
                            self.vaporize_pheromone(bad_pattern, bin_obj['support'])
                        # break
        if len(gen_pattern.gradual_items) <= 1:
            tgp = TGP(gp=pattern)
            return tgp
        else:
            # t_lag = FuzzyMF.calculate_time_lag(FuzzyMF.get_patten_indices(bin_data[0]), t_diffs, min_supp)
            t_lag = calculate_time_lag(bin_data[0], t_diffs)
            tgp = TGP(gp=gen_pattern, t_lag=t_lag)
            return tgp

    def plot_pheromone_matrix(self):
        x_plot = np.array(self.p_matrix)
        print(x_plot)
        # Figure size (width, height) in inches
        # plt.figure(figsize=(4, 4))
        plt.title("+: increasing; -: decreasing; x: irrelevant")
        # plt.xlabel("+: increasing; -: decreasing; x: irrelevant")
        # plt.ylabel('Attribute')
        plt.xlim(0, 3)
        plt.ylim(0, len(self.p_matrix))
        x = [0, 1, 2]
        y = []
        for i in range(len(self.data.title)):
            y.append(i)
            plt.text(-0.3, (i+0.5), self.data.title[i][1][:3])
        plt.xticks(x, [])
        plt.yticks(y, [])
        plt.text(0.5, -0.4, '+')
        plt.text(1.5, -0.4, '-')
        plt.text(2.5, -0.4, 'x')
        plt.pcolor(-x_plot, cmap='gray')
        plt.gray()
        plt.grid()
        plt.show()

    @staticmethod
    def check_anti_monotony(lst_p, pattern, subset=True):
        result = False
        if subset:
            for pat in lst_p:
                result = set(pattern.get_pattern()).issubset(set(pat.get_pattern()))
                if result:
                    break
        else:
            for pat in lst_p:
                result = set(pattern.get_pattern()).issuperset(set(pat.get_pattern()))
                if result:
                    break
        return result

    @staticmethod
    def is_duplicate(pattern, lst_winners, lst_losers):
        for pat in lst_losers:
            if pattern.get_pattern() == pat.get_pattern():
                return True
        for pat in lst_winners:
            if pattern.get_pattern() == pat.get_pattern():
                return True
        return False

    @staticmethod
    def bin_and(bins, n):
        temp_bin = np.array(bins[0]) * np.array(bins[1])
        # temp_bin = np.logical_and(bins[0], bins[1])
        supp = float(np.sum(temp_bin)) / float(n * (n - 1.0) / 2.0)
        return temp_bin, supp
