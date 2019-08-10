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


class GradualAntColony:

    def __init__(self, steps, max_combs, d_set, min_supp):
        self.steps = steps
        self.max_combs = max_combs
        self.thd_supp = min_supp
        self.data = d_set
        self.e_factor = 0.5
        self.p_matrix = d_set.p_matrix

    def run_ant_colony(self):
        p = self.p_matrix
        all_sols = []
        sols_win = list()
        for t in range(self.steps):
            for n in range(self.max_combs):
                sol_n = list()
                for i in range(len(self.data.attr_indxs)):
                    x = (rand.randint(1, self.max_combs) / self.max_combs)
                    pos = p[i][0] / (p[i][0] + p[i][1] + p[i][2])
                    neg = (p[i][0] + p[i][1]) / (p[i][0] + p[i][1] + p[i][2])
                    if x < pos:
                        temp_n = [self.data.attr_indxs[i], '+']
                    elif (x >= pos) and x < neg:
                        temp_n = [self.data.attr_indxs[i], '-']
                    else:
                        # temp_n = 'x'
                        continue
                    if temp_n not in sol_n:
                        sol_n.append(temp_n)
                if (sol_n != []) and (sol_n not in all_sols):
                    all_sols.append(sol_n)
                    # print(sol_n)
                    if sols_win:
                        # check for anti-monotony
                        is_sub = GradualAntColony.check_anti_monotony(sols_win, sol_n)
                        if is_sub:
                            continue
                    supp = self.evaluate_bin_solution(sol_n)
                    if supp and (supp >= self.thd_supp):
                        sol_w = []
                        [sol_w.append(tuple(obj)) for obj in sol_n]
                        sols_win.append([supp, sol_w])
                        self.update_pheromone(sol_n, supp)
        return sols_win

    def evaluate_bin_solution(self, pattern):
        # [['2', '+'], ['4', '+']]
        lst_bin = self.data.lst_bin
        temp_bins = []
        for obj_i in pattern:
            for obj_j in lst_bin:
                if obj_j[0] == obj_i:
                    temp_bins.append(obj_j[1])
        if len(temp_bins) == len(pattern):
            supp = GradualAntColony.perform_bin_and(temp_bins, self.data.get_size())
        else:
            supp = False
        return supp

    def evaluate_graph_solution(self, pattern):
        # [['2', '+'], ['4', '+']]
        lst_graph = self.data.lst_graph
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
            supp = GradualAntColony.find_graph_path(Graphs, self.data.get_size())
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
    def check_anti_monotony(lst_p, p_arr):
        result = False
        tuple_p = []
        [tuple_p.append(tuple(obj)) for obj in p_arr]
        # print(tuple_p)
        for obj in lst_p:
            # print(obj[1])
            result = set(tuple_p).issubset(set(obj[1]))
            if result:
                break
        return result

    @staticmethod
    def perform_bin_and(lst_bin, n):
        temp_bin = []
        if len(lst_bin) >= 2:
            for obj in lst_bin:
                if temp_bin != []:
                    # print(temp_bin)
                    temp_bin = temp_bin & obj
                else:
                    temp_bin = obj
            supp = float(np.sum(temp_bin)) / float(n * (n - 1.0) / 2.0)
            return supp
        else:
            return False

    @staticmethod
    def find_graph_path(lst_GHs, all_len):
        lst_Hs = []
        if len(lst_GHs) >= 2:
            G = lst_GHs[0]
            # print("G")
            # print(G.edges)
            for i in range(1, (len(lst_GHs))):
                H = lst_GHs[i]
                lst_Hs.append(H)
                # print("H"+str(i))
                # print(H.edges)
            p_len = GradualAntColony.get_graph_path_length(G, lst_Hs)
            supp = p_len / all_len
            # print("Support: "+str(supp))
            return supp
        else:
            # print("No")
            return False

    @staticmethod
    def get_graph_path_length(G, lst_Hs):
        length = 0
        g_edge_arr = list(G.edges)
        g_node_arr = list(G.nodes)
        g_len = len(g_edge_arr)
        k = g_len - 1
        lt_node = g_edge_arr[k]
        # print(lt_node)
        for i in range(g_len):
            st_node = g_edge_arr[i]
            g_rem = (g_len + 1 - i)
            new_g_arr = g_node_arr[i:].copy()
            if (g_rem > length) and (st_node != lt_node):
                # print(st_node)
                # modify by adding paths for other graphs
                short_len = 0
                n = 0
                for H in lst_Hs:
                    n += 1
                    # print("H"+str(n))
                    # print(H.edges)
                    try:
                        path_nodes = nx.shortest_path(H, st_node[0], lt_node[1])
                        temp_nodes = set(new_g_arr).intersection(set(path_nodes))
                        temp_l = len(temp_nodes)
                        # print(new_g_arr)
                        # print(path_nodes)
                        # print(temp_nodes)
                        # if g_rem < temp_l:
                        #    print("switched")
                        #    temp_l = g_rem
                        if short_len == 0 or temp_l < short_len:
                            # we take the shortest common path
                            short_len = temp_l
                            # print("short path found")
                            # print(temp_nodes)
                    except nx.NetworkXException:
                        short_len = 0
                        # print("no path")
                        break
                if short_len > length:
                    # print("path found len=" + str(short_len))
                    length = short_len
            else:
                break
        return length
