# -*- coding: utf-8 -*-
"""
@author: "Dickson Owuor"
@credits: "Thomas Runkler, Edmond Menya, and Anne Laurent,"
@license: "MIT"
@version: "1.0"
@email: "owuordickson@gmail.com"
@created: "29 April 2021"
@modified: "29 April 2021"

Breath-First Search for gradual patterns (GA-GRAANK)

"""
import numpy as np
from src.common.gp import GI, GP
from src.common.dataset_bfs import Dataset


class GradGA:

    def __init__(self, f_path, min_supp):
        self.d_set = Dataset(f_path, min_supp)
        self.d_set.init_gp_attributes()
        self.attr_index = self.d_set.attr_cols
        self.iteration_count = 0
        self.d, self.attr_keys = self.generate_d()  # distance matrix (d) & attributes corresponding to d

    def generate_d(self):
        v_bins = self.d_set.valid_bins
        # 1. Fetch valid bins group
        attr_keys = [GI(x[0], x[1].decode()).as_string() for x in v_bins[:, 0]]

        # 2. Initialize an empty d-matrix
        n = len(attr_keys)
        d = np.zeros((n, n), dtype=np.dtype('i8'))  # cumulative sum of all segments
        for i in range(n):
            for j in range(n):
                if GI.parse_gi(attr_keys[i]).attribute_col == GI.parse_gi(attr_keys[j]).attribute_col:
                    # Ignore similar attributes (+ or/and -)
                    continue
                else:
                    bin_1 = v_bins[i][1]
                    bin_2 = v_bins[j][1]
                    # Cumulative sum of all segments for 2x2 (all attributes) gradual items
                    d[i][j] += np.sum(np.multiply(bin_1, bin_2))
        # print(d)
        return d, attr_keys

    def run_genetic_algorithm(self):
        min_supp = self.d_set.thd_supp
        a = self.d_set.attr_size
        it_count = 0
        max_it = 100

        if self.d_set.no_bins:
            return []

        # 1. Remove d[i][j] < frequency-count of min_supp
        fr_count = ((min_supp * a * (a - 1)) / 2)
        self.d[self.d < fr_count] = 0

        while it_count < max_it:
            pass

        return []

    def validate_gp(self, pattern):
        # pattern = [('2', '+'), ('4', '+')]
        min_supp = self.d_set.thd_supp
        n = self.d_set.attr_size
        gen_pattern = GP()
        bin_arr = np.array([])

        for gi in pattern.gradual_items:
            arg = np.argwhere(np.isin(self.d_set.valid_bins[:, 0], gi.gradual_item))
            if len(arg) > 0:
                i = arg[0][0]
                valid_bin = self.d_set.valid_bins[i]
                if bin_arr.size <= 0:
                    bin_arr = np.array([valid_bin[1], valid_bin[1]])
                    gen_pattern.add_gradual_item(gi)
                else:
                    bin_arr[1] = valid_bin[1].copy()
                    temp_bin = np.multiply(bin_arr[0], bin_arr[1])
                    supp = float(np.sum(temp_bin)) / float(n * (n - 1.0) / 2.0)
                    if supp >= min_supp:
                        bin_arr[0] = temp_bin.copy()
                        gen_pattern.add_gradual_item(gi)
                        gen_pattern.set_support(supp)
        if len(gen_pattern.gradual_items) <= 1:
            return pattern
        else:
            return gen_pattern

    def decode_gp(self, gene):
        a = self.attr_keys

    @staticmethod
    def crossover(p_1, p_2):
        c_1 = p_1.copy()
        c_2 = p_1.copy()
        choice = np.random.randint(2, size=c_1.gene.size).reshape(c_1.gene.shape).astype(bool)
        c_1.gene = np.where(choice, p_1.gene, p_2.gene)
        c_2.gene = np.where(choice, p_2.gene, p_1.gene)
        return c_1, c_2

    @staticmethod
    def mutate(p_x):
        p_y = p_x.copy()
        rand_val_1 = np.random.randint(0, p_x.gene.shape[0])
        rand_val_2 = np.random.randint(0, p_x.gene.shape[1])
        if p_y.gene[rand_val_1, rand_val_2] == 0:
            p_y.gene[rand_val_1, rand_val_2] = 1
        else:
            p_y.gene[rand_val_1, rand_val_2] = 0
        return p_y

    @staticmethod
    def build_gene(prob, shape):
        temp_gene = []
        for i in range(shape[0]):
            temp = np.random.choice(a=prob.vals, size=(shape[1],))
            temp_gene.append(temp)
        return np.array(temp_gene)

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
