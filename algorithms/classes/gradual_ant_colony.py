# -*- coding: utf-8 -*-

"""
@author: "Dickson Owuor"
@credits: "Anne Laurent, Joseph Orero"
@license: "MIT"
@version: "1.0"
@email: "owuordickson@gmail.com"

"""


import numpy as np
import random as rand
import matplotlib.pyplot as plt


class GradualAntColony:

    def __init__(self, steps, max_combs, data, min_supp):
        self.steps = steps
        self.max_combs = max_combs
        self.thd_supp = min_supp
        self.dataset = data
        self.feature = ['+', '-', 'x']
        self.p_matrix = np.ones((len(self.dataset.attributes), len(self.feature)),
                                dtype=float)

    def run_ant_colony(self):
        p = self.p_matrix
        all_sols = []
        sols_win = []
        for t in range(self.steps):
            for n in range(self.max_combs):
                sol_n = []
                for i in range(len(self.dataset.attributes)):
                    x = (rand.randint(1, self.max_combs) / self.max_combs)
                    pos = p[i][0] / (p[i][0] + p[i][1] + p[i][2])
                    neg = (p[i][0] + p[i][1]) / (p[i][0] + p[i][1] + p[i][2])
                    if x < pos:
                        temp_n = [self.dataset.attributes[i], '+']
                    elif (x >= pos) and x < neg:
                        temp_n = [self.dataset.attributes[i], '-']
                    else:
                        # temp_n = 'x'
                        continue
                    if temp_n not in sol_n:
                        sol_n.append(temp_n)
                if (sol_n != []) and (sol_n not in all_sols):
                    all_sols.append(sol_n)
                    # print(sol_n)
                    supp = self.evaluate_solution(sol_n)
                    if supp and (supp >= self.thd_supp):
                        temp = [supp, sol_n]
                        sols_win.append(temp)
                        self.update_pheromone(sol_n, supp)
        return sols_win

    def evaluate_solution(self, pattern):
        # [['2', '+'], ['4', '+']]
        lst_graph = self.dataset.lst_graph
        Graphs = []
        for obj_i in pattern:
            for obj_j in lst_graph:
                temp = [obj_j[0], obj_j[1]]
                if temp == obj_i:
                    G = obj_j[2]
                    Graphs.append(G)
                    # print(temp)
                    # print(G.edges)
        if len(Graphs) == len(pattern):
            supp = GradualAntColony.find_path(Graphs)
        else:
            supp = False
        return supp

    def update_pheromone(self, sol, supp):
        # [['2', '+'], ['4', '+']], 0.6
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
                    self.p_matrix[i][j] += supp
                # else:
                #    self.p_matrix[i][k] -= supp

    def plot_pheromone_matrix(self):
        x_plot = np.array(self.p_matrix)
        # take care of irrelevant attributes
        for i in range(len(x_plot)):
            if x_plot[i][0] == x_plot[i][1] and x_plot[i][1] == x_plot[i][2]:
                x_plot[i][2] += 1
        print(x_plot)
        # Figure size (width, height) in inches
        plt.figure(figsize=(4, 4))
        plt.title("Attribute Gray Plot")
        plt.xlabel("+: increasing; -: decreasing; x: irrelevant")
        # plt.ylabel('Attribute')
        plt.xlim(0, 3)
        plt.ylim(0, len(self.p_matrix))
        x = [0, 1, 2]
        y = []
        for i in range(len(self.dataset.title)):
            y.append(i)
            plt.text(-0.3, (i+0.5), self.dataset.title[i][1][:3])
        plt.xticks(x, [])
        plt.yticks(y, [])
        plt.text(0.5, -0.2, '+')
        plt.text(1.5, -0.2, '-')
        plt.text(2.5, -0.2, 'x')
        plt.pcolor(-x_plot, cmap='gray')
        plt.gray()
        plt.grid()
        plt.show()

    @staticmethod
    def find_path(lst_Gs):
        if len(lst_Gs) >= 2:
            # print("Yes")
            return 0.6
        else:
            # print("No")
            return False
