# -*- coding: utf-8 -*-
"""
@author: "Dickson Owuor"
@credits: "Thomas Runkler, Edmond Menya, and Anne Laurent,"
@license: "MIT"
@version: "1.0"
@email: "owuordickson@gmail.com"

"""


import numpy as np
import random as rand
import matplotlib.pyplot as plt


class GradACO:

    def __init__(self, steps, max_combs, d_set):
        self.steps = steps
        self.max_combs = max_combs
        self.data = d_set
        self.e_factor = 0  # evaporation factor
        self.p_matrix = np.ones((self.data.column_size, 3), dtype=int)
        self.bin_patterns = []
        self.invalid_patterns = []

    def run_ant_colony(self):
        all_sols = list()
        win_sols = list()
        loss_sols = list()
        invalid_sols = list()
        for t in range(self.steps):
            for n in range(self.max_combs):
                sol_n = self.generate_rand_pattern()
                # print(sol_n)
                if sol_n and (sol_n not in all_sols):
                    all_sols.append(sol_n)
                    if loss_sols:
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
                    supp = self.evaluate_bin_solution(sol_n)
                    if supp and (supp >= self.data.thd_supp):
                        win_sols.append([supp, sol_n])
                        self.update_pheromone(sol_n)
                    elif supp and (supp < self.data.thd_supp):
                        loss_sols.append([supp, sol_n])
                        # self.update_pheromone(sol_n, False)
                    else:
                        invalid_sols.append([supp, sol_n])
                        # self.update_pheromone(sol_n, False)
        return win_sols

    def run_ant_colony(self, min_supp):
        all_sols = list()
        win_sols = list()
        loss_sols = list()
        invalid_sols = list()
        for t in range(self.steps):
            for n in range(self.max_combs):
                sol_n = self.generate_rand_pattern()
                # print(sol_n)
                if sol_n and (sol_n not in all_sols):
                    all_sols.append(sol_n)
                    if loss_sols:
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
                    supp, sol_gen = self.evaluate_bin_solution(sol_n, min_supp)
                    if supp and (supp >= min_supp):
                        win_sols.append([supp, sol_gen])
                        self.update_pheromone(sol_gen)
                    elif supp and (supp < min_supp):
                        loss_sols.append([supp, sol_gen])
                        # self.update_pheromone(sol_n, False)
                    else:
                        invalid_sols.append([supp, sol_gen])
                        # self.update_pheromone(sol_n, False)
        return win_sols

    def generate_rand_pattern(self):
        p = self.p_matrix
        n = self.data.column_size
        pattern = list()
        count = 0
        for i in range(n):
            a = 1  # float(self.data.attr_matrix[i])
            x = float(rand.randint(1, self.max_combs) / self.max_combs)
            pos = float((p[i][0] * a) / (p[i][0] + p[i][1] + p[i][2]))
            neg = float(((p[i][0] + p[i][1]) * a) / (p[i][0] + p[i][1] + p[i][2]))
            if x < pos:
                temp = tuple([self.data.attr_index[i], '+'])
            elif (x >= pos) and (x < neg):
                temp = tuple([self.data.attr_index[i], '-'])
            else:
                # temp = tuple([self.data.attr_index[i], 'x'])
                continue
            pattern.append(temp)
            count += 1
        if count <= 1:
            pattern = False
        return pattern

    def evaluate_bin_solution(self, pattern):
        # pattern = [('2', '+'), ('4', '+')]
        lst_bin = self.data.lst_bin
        bin_data = []
        count = 0
        for obj_i in pattern:
            # fetch pattern
            for obj_j in lst_bin:
                if obj_j[0] == obj_i:
                    bin_data.append(obj_j[1])
                    count += 1
                    break
        if count <= 1:
            return False
        else:
            supp = GradACO.perform_bin_and(bin_data, self.data.get_size())
            return supp

    def evaluate_bin_solution(self, pattern, min_supp):
        # pattern = [('2', '+'), ('4', '+')]
        lst_bin = self.data.lst_bin
        gen_pattern = []
        bin_data = []
        # invalid = False
        count = 0
        for obj_i in pattern:
            if obj_i in self.invalid_patterns:
                continue
            elif obj_i in self.bin_patterns:
                # fetch pattern
                for obj in lst_bin:
                    if obj[0] == obj_i:
                        gen_pattern.append(obj[0])
                        bin_data.append(obj[1])
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
                    # self.bin_patterns.append(tuple([obj_i[0], '+']))
                    # self.bin_patterns.append(tuple([obj_i[0], '-']))
                    # if supp < min_supp:
                        # self.update_pheromone([tuple([obj_i[0], 'x'])], False)
                        # i = int(obj_i[0])
                        # self.data.attr_matrix[(i - 1)] = 0
                        # invalid = True
                        # break
                        # continue
                    # else:
                        # self.update_pheromone([obj_i], True)
                        # i = int(obj_i[0])
                        # self.data.attr_matrix[(i - 1)] = supp
                    if supp >= min_supp:
                        self.bin_patterns.append(tuple([obj_i[0], '+']))
                        self.bin_patterns.append(tuple([obj_i[0], '-']))
                        gen_pattern.append(obj_i)
                        bin_data.append(temp_bin)
                        count += 1
                    else:
                        self.invalid_patterns.append(tuple([obj_i[0], '+']))
                        self.invalid_patterns.append(tuple([obj_i[0], '-']))
                else:
                    # binary does not exist
                    return False
        if count <= 1:
            return False, False
        else:
            supp = GradACO.perform_bin_and(bin_data, self.data.get_size())
            return supp, gen_pattern

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
    def perform_bin_and(lst_bin, n):
        temp_bin = np.array([])
        for obj in lst_bin:
            if temp_bin.size != 0:
                temp_bin = temp_bin & obj
            else:
                temp_bin = obj
        supp = float(np.sum(temp_bin)) / float(n * (n - 1.0) / 2.0)
        return supp
