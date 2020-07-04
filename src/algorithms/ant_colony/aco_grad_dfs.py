# -*- coding: utf-8 -*-
"""
@author: "Dickson Owuor"
@credits: "Thomas Runkler, Edmond Menya, and Anne Laurent,"
@license: "MIT"
@version: "3.0"
@email: "owuordickson@gmail.com"
@created: "12 July 2019"
@modified: "16 June 2020"

"""

import numpy as np
from numpy import random as rand
import matplotlib.pyplot as plt
from ..common.gp import GI, GP
from ..common.dataset_v2 import Dataset


class GradACO:

    def __init__(self, f_path, min_supp, eq):
        print("GradACO: Version 2.0")
        self.d_set = Dataset(f_path, min_supp, eq)
        self.d_set.init_gp_attributes()
        self.c_matrix = self.d_set.cost_matrix
        self.p_matrix = np.ones((self.d_set.column_size, 3), dtype=int)
        # Data set reduction and update: (p_matrix, attr_index)
        #self.reduce_data()
        self.attr_index = self.d_set.attr_cols
        # self.e_factor = 0.1  # evaporation factor
        print(self.d_set.encoded_data)
        print(self.p_matrix)

    def reduce_data(self):
        c_matrix = self.c_matrix
        # 1. remove invalid attributes
        valid_a1 = list()
        valid_a2 = [-2, -1]
        for i in range(len(self.d_set.attr_cols)):
            a = self.d_set.attr_cols[i]
            valid = (c_matrix[a][0] < c_matrix[a][2]) or \
                    (c_matrix[a][1] < c_matrix[a][2])
            if valid:
                valid_a1.append(i)
                valid_a2.append(i)
            else:
                self.p_matrix[a][0] = 0
                self.p_matrix[a][1] = 0
        self.d_set.attr_cols = self.d_set.attr_cols[valid_a1]
        valid_a2 = np.array(valid_a2) + 2
        self.d_set.encoded_data = self.d_set.encoded_data[:, valid_a2]

        # 2. merge similar patterns
        # 2a. get indices
        vals, inverse, count = np.unique(self.d_set.encoded_data[:, 2:],
                                         return_inverse=True,
                                         return_counts=True, axis=0)
        idx_vals_repeated = np.where(count > 1)[0]
        rows, cols = np.where(inverse == idx_vals_repeated[:, np.newaxis])
        _, inverse_rows = np.unique(rows, return_index=True)
        res = np.split(cols, inverse_rows[1:])
        # 2b. use indices to merge
        if len(res) > 1:
            n_data = list()
            for obj in res:
                row = self.d_set.encoded_data[obj]
                nodes = row[:, [0, 1]]
                data = row[0, 2:]
                n_data.append([nodes, data])
            self.d_set.encoded_data = np.array(n_data)

    def run_ant_colony(self):
        min_supp = self.d_set.thd_supp
        winner_gps = list()  # subsets
        loser_gps = list()  # supersets
        repeated = 0
        while repeated < 1:
            rand_gp = self.generate_random_gp()
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
                    if gen_gp.support >= min_supp:
                        self.deposit_pheromone(gen_gp)
                        is_present = GradACO.is_duplicate(gen_gp, winner_gps, loser_gps)
                        is_sub = GradACO.check_anti_monotony(winner_gps, gen_gp, subset=True)
                        if is_present or is_sub:
                            repeated += 1
                        else:
                            winner_gps.append(gen_gp)
                    else:
                        loser_gps.append(gen_gp)
                        # update pheromone as irrelevant with loss_sols
                        # self.vaporize_pheromone(gen_gp, self.e_factor)
                    if set(gen_gp.get_pattern()) != set(rand_gp.get_pattern()):
                        loser_gps.append(rand_gp)
                else:
                    repeated += 1
        return winner_gps

    def generate_random_gp(self):
        p = self.p_matrix
        c = self.c_matrix
        attrs = self.attr_index.copy()
        np.random.shuffle(attrs)
        n = len(attrs)
        pattern = GP()
        for i in attrs:
            max_extreme = n  # * 100
            x = float(rand.randint(1, max_extreme) / max_extreme)
            p0 = p[i][0] * (1 / c[i][0])
            p1 = p[i][1] * (1 / c[i][1])
            p2 = p[i][2] * (1 / c[i][2])
            pos = float(p0 / (p0 + p1 + p2))
            neg = float((p0 + p1) / (p0 + p1 + p2))
            if x < pos:
                temp = GI(i, '+')
            elif (x >= pos) and (x < neg):
                temp = GI(i, '-')
            else:
                # temp = GI(self.attr_index[i], 'x')
                continue
            pattern.add_gradual_item(temp)
        return pattern

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

    def validate_gp(self, pattern):
        # pattern = [('2', '+'), ('4', '+')]
        min_supp = self.d_set.thd_supp
        n = self.d_set.attr_size
        attrs, symbs = pattern.get_attributes()
        le, raw = self.find_longest_path(attrs, symbs)
        supp = float(le / n)
        if supp > min_supp:
            gen_pattern = GP()
            for a, v in raw:
                if v == 1:
                    gen_pattern.add_gradual_item(GI(a, '+'))
                elif v == -1:
                    gen_pattern.add_gradual_item(GI(a, '-'))
                # else:
                #    gen_pattern.add_gradual_item(GI(a, 'x'))
            gen_pattern.set_support(supp)
            return gen_pattern
        return pattern

    def find_longest_path(self, attrs, syms):
        # 1. remove invalid attributes
        lst_attr = []
        lst_sym = []
        for a, b in zip(attrs, syms):
            valid = (self.c_matrix[a][0] < self.c_matrix[a][2]) or \
                    (self.c_matrix[a][1] < self.c_matrix[a][2])
            if valid:
                lst_attr.append(a)
                lst_sym.append(b)
        if len(lst_attr) <= 0:
            return 0, zip(attrs, syms)

        # 2. Find longest length
        lst_attr, lst_sym = zip(*sorted(zip(lst_attr, lst_sym)))
        enc_data = self.d_set.encoded_data
        length = 0
        indx = 300  # self.d_set.start_node[0]

        lst_indx = [np.argwhere(self.attr_index == x)[0][0] for x in lst_attr]
        # print(str(lst_attr) + ' + ' + str(self.attr_index) + ' = ' + str(lst_indx))
        # path = np.where((lst_sym == st_nodes[2][:, lst_attr]), True, False)
        # node = GradACO.find_node(indx, arr_p[2][:, lst_attr], lst_sym)
        # print(str(lst_attr) + ' : ' + str(arr_p[2][:, lst_attr]) + ' + ' + str(lst_sym) + ' = ' + str(node))
        while 0 < indx < (self.d_set.attr_size - 1):
            # print("running " + str(indx) + " ...")
            arr_p = enc_data[indx]
            node = GradACO.find_node(indx, arr_p[1][:, lst_indx], lst_sym)
            if len(node) > 0:
                if length == 0:
                    length += 2
                else:
                    length += 1
                indx = node[1]
            else:
                indx = -1
        print(str(lst_attr) + ' , ' + str(lst_sym) + ' : length = ' + str(length))
        return length, zip(lst_attr, lst_sym)

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
        for i in range(len(self.d_set.title)):
            y.append(i)
            plt.text(-0.3, (i+0.5), self.d_set.title[i][1][:3])
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
    def index_count(idxs, n):
        arr_1 = idxs[0]
        arr_n = idxs[1:]
        temp = []
        start = 0
        ok = False
        for item in np.nditer(arr_1):
            for arr in arr_n:
                arg = np.argwhere(arr == item)
                if arg.size > 0:
                    ok = (arg[0][0] >= start)
                if not ok:
                    break
            if ok:
                temp.append(item)
            start += 1
        supp = float(len(temp) / n)
        return np.array(temp), supp

    @staticmethod
    def find_node(i, arr, pat):
        # lst_nodes = []
        temp = []
        paths = np.argwhere(np.all(arr == pat, axis=1))
        # for p in paths:
        if len(paths) > 0:
            temp = [i, (i + paths[0][0]) + 1]
        return temp
            # lst_nodes.append(temp)
        # return lst_nodes


