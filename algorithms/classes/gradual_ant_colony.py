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
import networkx as nx


class GradACO:

    def __init__(self, steps, max_combs, d_set, min_supp):
        self.steps = steps
        self.max_combs = max_combs
        self.thd_supp = min_supp
        self.data = d_set
        self.e_factor = 0.5  # evaporation factor
        self.p_matrix = d_set.p_matrix

    def run_ant_colony(self):
        p = self.p_matrix
        print(p)
        all_sols = list()
        win_sols = list()
        loss_sols = list()
        for t in range(self.steps):
            for n in range(self.max_combs):
                sol_n = self.generate_rand_pattern()
                if (sol_n != []) and (sol_n not in all_sols):
                    all_sols.append(sol_n)
                    if loss_sols:
                        # check for super-set anti-monotony
                        true = GradACO.check_anti_monotony(loss_sols, sol_n, False)
                        if true:
                            continue
                    if win_sols:
                        # check for sub-set anti-monotony
                        true = GradACO.check_anti_monotony(win_sols, sol_n, True)
                        if true:
                            continue
                    supp = self.evaluate_bin_solution(sol_n)
                    if supp and (supp >= self.thd_supp):
                        win_sols.append([supp, sol_n])
                        self.update_pheromone(sol_n, supp)
                    elif supp and (supp < self.thd_supp):
                        loss_sols.append([supp, sol_n])
        return win_sols

    def generate_rand_pattern(self):
        p = self.p_matrix
        pattern = list()
        count = 0
        for i in range(self.data.column_size):
            x = (rand.randint(1, self.max_combs) / self.max_combs)
            pos = p[i][0] / (p[i][0] + p[i][1] + p[i][2])
            neg = (p[i][0] + p[i][1]) / (p[i][0] + p[i][1] + p[i][2])
            if x < pos:
                temp = tuple([self.data.attr_indxs[i], '+'])
            elif (x >= pos) and (x < neg):
                temp = tuple([self.data.attr_indxs[i], '-'])
            else:
                # temp = 'x'
                continue
            pattern.append(temp)
            count += 1
        if count <= 1:
            pattern = []
        return pattern

    def evaluate_bin_solution(self, pattern):
        # [('2', '+'), ('4', '+')]
        lst_bin = self.data.lst_bin
        temp_bins = []
        count = 0
        for obj_i in pattern:
            for obj_j in lst_bin:
                if obj_j[0] == obj_i:
                    temp_bins.append(obj_j[1])
                    count += 1
        if count <= 1:
            return False
        else:
            supp = GradACO.perform_bin_and(temp_bins, self.data.get_size())
            return supp

    def update_pheromone(self, sol, supp):
        for obj in sol:
            attr = int(obj[0])
            symbol = obj[1]
            i = attr - 1
            if symbol == '+':
                j = 0
            elif symbol == '-':
                j = 1
            else:
                j = 2
            for k in range(len(self.p_matrix[i])):
                if k == j:
                    old = self.p_matrix[i][j]
                    self.p_matrix[i][j] = (old * (1 - self.e_factor)) + supp
                else:
                    old = self.p_matrix[i][k]
                    self.p_matrix[i][k] = (old * (1 - self.e_factor))

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
