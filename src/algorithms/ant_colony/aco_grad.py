# -*- coding: utf-8 -*-
"""
@author: "Dickson Owuor"
@credits: "Thomas Runkler, Edmond Menya, and Anne Laurent,"
@license: "MIT"
@version: "1.0"
@email: "owuordickson@gmail.com"
@created: "12 July 2019"

"""

import numpy as np
import random as rand
# import matplotlib.pyplot as plt
# from src import FuzzyMF
from algorithms.tgraank.fuzzy_mf import FuzzyMF


class GradACO:

    def __init__(self, d_set):
        self.data = d_set
        self.attr_index = self.data.attr_index
        self.e_factor = 0  # evaporation factor
        self.p_matrix = np.ones((self.data.column_size, 3), dtype=int)
        self.valid_bins = []
        self.invalid_bins = []

    def run_ant_colony(self, min_supp, time_diffs=None):
        all_sols = list()
        win_sols = list()
        win_lag_sols = list()
        loss_sols = list()
        invalid_sols = list()
        # count = 0
        # converging = False
        # while not converging:
        repeated = 0
        while repeated < 5:
            # count += 1
            sol_n = self.generate_rand_pattern()
            # print(sol_n)
            if sol_n:
                if sol_n not in all_sols:
                    lag_sols = []
                    repeated = 0
                    all_sols.append(sol_n)
                    if loss_sols or invalid_sols:
                        # check for super-set anti-monotony
                        is_super = GradACO.check_anti_monotony(loss_sols, sol_n, False)
                        is_invalid = GradACO.check_anti_monotony(invalid_sols, sol_n, False)
                        if is_super or is_invalid:
                            continue
                    if win_sols:
                        # check for sub-set anti-monotony
                        is_sub = GradACO.check_anti_monotony(win_sols, sol_n, True)
                        if is_sub:
                            continue
                    if time_diffs is None:
                        supp, sol_gen = self.evaluate_bin_solution(sol_n, min_supp, time_diffs)
                    else:
                        supp, lag_sols = self.evaluate_bin_solution(sol_n, min_supp, time_diffs)
                        if supp:
                            sol_gen = lag_sols[0]
                        else:
                            sol_gen = False
                    # print(supp)
                    if supp and (supp >= min_supp):  # and ([supp, sol_gen] not in win_sols):
                        if [supp, sol_gen] not in win_sols:
                            win_sols.append([supp, sol_gen])
                            self.update_pheromone(sol_gen)
                            if time_diffs is not None:
                                win_lag_sols.append([supp, lag_sols])
                            # converging = self.check_convergence()
                    elif supp and (supp < min_supp):  # and ([supp, sol_gen] not in loss_sols):
                        if [supp, sol_gen] not in loss_sols:
                            loss_sols.append([supp, sol_gen])
                        # self.update_pheromone(sol_n, False)
                    else:
                        invalid_sols.append([supp, sol_n])
                        if sol_gen:
                            invalid_sols.append([supp, sol_gen])
                        # self.update_pheromone(sol_n, False)
                else:
                    repeated += 1
            # converging = self.check_convergence(repeated)
            # is_member = GradACO.check_convergence(win_sols, sol_n)
        # print("All: "+str(len(all_sols)))
        # print("Winner: "+str(len(win_sols)))
        # print("Losers: "+str(len(loss_sols)))
        # print(count)
        if time_diffs is None:
            return win_sols
        else:
            return win_lag_sols

    def generate_rand_pattern(self):
        p = self.p_matrix
        n = len(self.attr_index)
        pattern = list()
        count = 0
        for i in range(n):
            max_extreme = len(self.attr_index)
            x = float(rand.randint(1, max_extreme) / max_extreme)
            pos = float(p[i][0] / (p[i][0] + p[i][1] + p[i][2]))
            neg = float((p[i][0] + p[i][1]) / (p[i][0] + p[i][1] + p[i][2]))
            if x < pos:
                temp = tuple([self.attr_index[i], '+'])
            elif (x >= pos) and (x < neg):
                temp = tuple([self.attr_index[i], '-'])
            else:
                # temp = tuple([self.data.attr_index[i], 'x'])
                continue
            pattern.append(temp)
            count += 1
        if count <= 1:
            pattern = False
        return pattern

    def evaluate_bin_solution(self, pattern, min_supp, time_diffs):
        # pattern = [('2', '+'), ('4', '+')]
        lst_bin = self.data.lst_bin
        gen_pattern = []
        bin_data = []
        count = 0
        for obj_i in pattern:
            if obj_i in self.invalid_bins:
                continue
            elif obj_i in self.valid_bins:
                # fetch pattern
                for obj in lst_bin:
                    if obj[0] == obj_i:
                        gen_pattern.append(obj[0])
                        bin_data.append([obj[1], obj[2], obj[0]])
                        count += 1
                        break
            else:
                attr_data = False
                for obj in self.data.attr_data:
                    if obj[0] == obj_i[0]:
                        attr_data = obj
                        break
                if attr_data:
                    supp, temp_bin = self.data.get_bin_rank(attr_data, obj_i[1])
                    if supp >= min_supp:
                        self.valid_bins.append(tuple([obj_i[0], '+']))
                        self.valid_bins.append(tuple([obj_i[0], '-']))
                        gen_pattern.append(obj_i)
                        bin_data.append([temp_bin, supp, obj_i])
                        count += 1
                    else:
                        self.invalid_bins.append(tuple([obj_i[0], '+']))
                        self.invalid_bins.append(tuple([obj_i[0], '-']))
                else:
                    # binary does not exist
                    return False, False
        if count <= 1:
            return False, False
        else:
            size = len(self.data.attr_data[0][1])
            supp, new_pattern = GradACO.perform_bin_and(bin_data, size, min_supp, gen_pattern, time_diffs)
            return supp, new_pattern

    def update_pheromone(self, pattern):
        lst_attr = []
        for obj in pattern:
            attr = int(obj[0])
            lst_attr.append(attr)
            symbol = obj[1]
            i = attr - 1
            if symbol == '+':
                self.p_matrix[i][0] += 1
            elif symbol == '-':
                self.p_matrix[i][1] += 1
        for index in self.data.attr_index:
            if int(index) not in lst_attr:
                # print(obj)
                i = int(index) - 1
                self.p_matrix[i][2] += 1

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
    def check_anti_monotony(lst_p, p_arr, ck_sub):
        result = False
        if ck_sub:
            for obj in lst_p:
                result = set(p_arr).issubset(set(obj[1]))
                if result:
                    break
        else:
            for obj in lst_p:
                result = set(p_arr).issuperset(set(obj[1]))
                if result:
                    break
        return result

    @staticmethod
    def perform_bin_and(unsorted_bins, n, thd_supp, gen_p, t_diffs):
        lst_bin = sorted(unsorted_bins, key=lambda x: x[1])
        final_bin = np.array([])
        pattern = []
        count = 0
        for obj in lst_bin:
            temp_bin = final_bin
            if temp_bin.size != 0:
                temp_bin = temp_bin & obj[0]
                supp = float(np.sum(temp_bin)) / float(n * (n - 1.0) / 2.0)
                if supp >= thd_supp:
                    final_bin = temp_bin
                    pattern.append(obj[2])
                    count += 1
            else:
                final_bin = obj[0]
                pattern.append(obj[2])
                count += 1
        supp = float(np.sum(final_bin)) / float(n * (n - 1.0) / 2.0)
        if count >= 2:
            if t_diffs is None:
                return supp, pattern
            else:
                t_lag = FuzzyMF.calculate_time_lag(FuzzyMF.get_patten_indices(final_bin), t_diffs, thd_supp)
                if t_lag:
                    temp_p = [pattern, t_lag]
                    return supp, temp_p
                else:
                    return -1, pattern
        else:
            return -1, gen_p
