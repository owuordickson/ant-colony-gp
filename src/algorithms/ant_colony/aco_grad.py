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
import random as rand
# import matplotlib.pyplot as plt
from src.algorithms.tgraank.fuzzy_mf import FuzzyMF
from src.algorithms.ant_colony.gp_old import TGP
from src.algorithms.ant_colony.gp import GP, GI


class GradACO:

    def __init__(self, d_set):
        self.data = d_set
        self.attr_index = self.data.attr_cols
        self.e_factor = 0.9  # evaporation factor
        self.p_matrix = np.ones((self.data.column_size, 3), dtype=float)

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
                    gen_gp = self.validate_gp(rand_gp, min_supp, time_diffs=None)
                    is_present = GradACO.is_duplicate(gen_gp, winner_gps, loser_gps)
                    if gen_gp.support >= min_supp and not is_present:
                        winner_gps.append(gen_gp)
                        self.add_pheromone(gen_gp)
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
        all_sols = list()
        win_sols = list()  # subsets
        win_lag_sols = list()
        loss_sols = list()  # supersets
        repeated = 0
        while repeated < 1:
            sol_n = self.generate_rand_pattern()
            if sol_n:
                if sol_n not in all_sols:
                    lag_sols = []
                    repeated = 0
                    all_sols.append(sol_n)
                    if loss_sols:
                        # check for super-set anti-monotony
                        is_super = GradACO.check_anti_monotony(loss_sols, sol_n, False)
                        if is_super:
                            continue
                    if win_sols:
                        # check for sub-set anti-monotony
                        is_sub = GradACO.check_anti_monotony(win_sols, sol_n, True)
                        if is_sub:
                            continue
                    #if time_diffs is None:
                    #    supp, sol_gen = self.evaluate_bin_solution(sol_n, min_supp, time_diffs)
                    #else:
                    supp, lag_sols = self.validate_gp(sol_n, min_supp, time_diffs)
                    #    if supp:
                    #        sol_gen = lag_sols[0]
                    #    else:
                    #        sol_gen = False
                    if supp >= min_supp:
                        sol_gen = lag_sols[0]
                        if [supp, sol_gen] not in win_sols:
                            win_sols.append([supp, sol_gen])
                            self.add_pheromone(sol_gen)
                            # if time_diffs is not None:
                            win_lag_sols.append([supp, lag_sols])
                    else:
                        # self.update_pheromone(sol_n, False)
                        # update pheromone as irrelevant with loss_sols
                        # self.negate_pheromone(sol_gen)
                        loss_sols.append([supp, sol_gen])
                        if sol_gen:
                            all_sols.append(sol_gen)
                else:
                    repeated += 1
        return GradACO.remove_subsets(win_lag_sols, True)

    def generate_rand_pattern(self):
        p = self.p_matrix
        n = len(self.attr_index)
        pattern = GP()
        for i in range(n):
            max_extreme = n * 10
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

    def validate_gp(self, pattern, min_supp, time_diffs):
        # pattern = [('2', '+'), ('4', '+')]
        gen_pattern = GP()
        bin_data = []
        for obj_i in pattern.get_pattern():
            if obj_i in self.data.invalid_bins:
                continue
            else:  # call method
                # fetch pattern
                for obj in self.data.valid_bins:
                    if obj[0] == obj_i:
                        gi = GI(obj_i[0], obj_i[1])
                        gen_pattern.add_gradual_item(gi)
                        bin_data.append([obj[1], obj[2], obj[0]])
                        break
        if len(gen_pattern.gradual_items) <= 1:
            return pattern
        else:
            size = self.data.attr_size
            new_pattern = GradACO.perform_bin_and(bin_data, size, min_supp, gen_pattern, time_diffs)
            return new_pattern

    def add_pheromone(self, pattern):
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

    def negate_pheromone(self, pattern):
        lst_attr = []
        for obj in pattern.gradual_items:
            attr = obj.attribute_col
            symbol = obj.symbol
            lst_attr.append(attr)
            i = attr
            if symbol == '+':
                self.p_matrix[i][0] *= self.e_factor
            elif symbol == '-':
                self.p_matrix[i][1] *= self.e_factor

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

    def evaluate_bins(self, raw_gi, bins):
        for obj in self.data.valid_bins:
            if obj[0] == raw_gi:
                gi = GI(raw_gi[0], raw_gi[1])
                # gen_pattern.add_gradual_item(gi)
                # bin_data.append([obj[1], obj[2], obj[0]])
                return gi, supp

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
    def perform_bin_and(unsorted_bins, n, thd_supp, gen_p, t_diffs):
        lst_bin = sorted(unsorted_bins, key=lambda x: x[1])
        #print(lst_bin)
        final_bin = np.array([])
        pattern = GP()
        count = 0
        for obj in lst_bin:
            temp_bin = final_bin
            if temp_bin.size != 0:
                temp_bin = temp_bin & obj[0]
                supp = float(np.sum(temp_bin)) / float(n * (n - 1.0) / 2.0)
                if supp >= thd_supp:
                    final_bin = temp_bin
                    gi = GI(obj[2][0], obj[2][1])
                    pattern.add_gradual_item(gi)
                    count += 1
            else:
                final_bin = obj[0]
                gi = GI(obj[2][0], obj[2][1])
                pattern.add_gradual_item(gi)
                count += 1
        supp = float(np.sum(final_bin)) / float(n * (n - 1.0) / 2.0)
        pattern.set_support(supp)
        if count >= 2:
            if t_diffs is None:
                return pattern
            else:
                t_lag, t_stamp = FuzzyMF.calculate_time_lag(FuzzyMF.get_patten_indices(final_bin), t_diffs, thd_supp)
                if t_lag:
                    temp_p = [pattern, t_lag, t_stamp]
                    return temp_p
                else:
                    pattern.set_support(-1)
                    return pattern
        else:
            gen_p.set_support(-1)
            return gen_p

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
        temp_bin = bins[0] & bins[1]
        supp = float(np.sum(temp_bin)) / float(n * (n - 1.0) / 2.0)
        return supp
