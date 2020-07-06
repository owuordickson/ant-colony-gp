# -*- coding: utf-8 -*-
"""
@author: "Dickson Owuor"
@credits: "Thomas Runkler, Edmond Menya, and Anne Laurent,"
@license: "MIT"
@version: "3.0"
@email: "owuordickson@gmail.com"
@created: "06 July 2020"

Depth-First Search for gradual patterns (ACO-ParaMiner)

"""

import numpy as np
from numpy import random as rand
from .aco_grad_bfs import GradACO
from ..common.gp import GI, GP
from ..common.dataset_dfs import Dataset_dfs


class GradACO_dfs(GradACO):

    def __init__(self, f_path, min_supp, eq):
        print("GradACO: Version 2.0")
        self.d_set = Dataset_dfs(f_path, min_supp, eq)
        self.d_set.init_gp_attributes()
        self.c_matrix = self.d_set.cost_matrix
        self.p_matrix = np.ones((self.d_set.column_size, 3), dtype=int)
        # Data set reduction and update: (p_matrix, attr_index)
        self.bins, self.indices = self.reduce_data()
        self.attr_index = self.d_set.attr_cols
        # self.e_factor = 0.1  # evaporation factor

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
                                         return_counts=True,
                                         axis=0)
        return vals, inverse

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
        # 1. Find longest length
        lst_attr, lst_sym = zip(*sorted(zip(attrs, syms)))
        enc_data = self.d_set.encoded_data
        length = 0
        lst_indx = [np.argwhere(self.attr_index == x)[0][0] for x in lst_attr]
        temp_i = np.argwhere(np.all(self.bins[:, lst_indx] == lst_sym, axis=1))
        if temp_i.size <= 0:
            return length, zip(lst_attr, lst_sym)
        else:
            indices = []
            for i in temp_i:
                temp = np.argwhere(self.indices == i[0]).ravel()
                indices.extend(temp)
            arr_nodes = enc_data[indices, :2]
            length = np.unique(arr_nodes[:, 0], axis=0).size + 1
            # length = GradACO.count_nodes(arr_nodes)
            # print(nodes)
            # print("Indices of " + str(lst_sym) + " are: " + str(indices))
            # print(str(lst_attr) + ' + ' + str(self.attr_index) + ' = ' + str(lst_indx))
            # print(str(str(lst_sym) + ' is at index ' + str(temp_i)))
            # print(str(lst_attr) + ' , ' + str(lst_sym) + ' : length = ' + str(length))
            return length, zip(lst_attr, lst_sym)
        # indx = 300  # self.d_set.start_node[0]

        # lst_indx = [np.argwhere(self.attr_index == x)[0][0] for x in lst_attr]
        # print(str(lst_attr) + ' + ' + str(self.attr_index) + ' = ' + str(lst_indx))
        # path = np.where((lst_sym == st_nodes[2][:, lst_attr]), True, False)
        # node = GradACO.find_node(indx, arr_p[2][:, lst_attr], lst_sym)
        # print(str(lst_attr) + ' : ' + str(arr_p[2][:, lst_attr]) + ' + ' + str(lst_sym) + ' = ' + str(node))
        # while 0 < indx < (self.d_set.attr_size - 1):
            # print("running " + str(indx) + " ...")
        #    arr_p = enc_data[indx]
        #    node = GradACO.find_node(indx, arr_p[1][:, lst_indx], lst_sym)
        #    if len(node) > 0:
        #        if length == 0:
        #            length += 2
        #        else:
        #            length += 1
        #        indx = node[1]
        #    else:
        #        indx = -1
        # print(str(lst_attr) + ' , ' + str(lst_sym) + ' : length = ' + str(length))
        # return length, zip(lst_attr, lst_sym)
