# -*- coding: utf-8 -*-
"""
@author: "Dickson Owuor"
@credits: "Thomas Runkler and Anne Laurent,"
@license: "MIT"
@version: "2.4"
@email: "owuordickson@gmail.com"
@created: "19 November 2019"
@modified: "11 June 2020"

Description: updated version that uses aco-graank and parallel multi-processing

"""


import numpy as np
from ...common.hdf5.dataset_h5 import Dataset_h5
from .aco_grad_h5 import GradACO_h5
from ..aco_tgrad import T_GradACO
from ...common.fuzzy_mf import calculate_time_lag
from ...common.gp import GP, TGP
from src.algorithms.common.profile_cpu import Profile


class GradACOt_h5 (GradACO_h5):

    def __init__(self, d_set, attr_data, t_diffs):
        self.d_set = d_set
        self.time_diffs = t_diffs
        self.attr_index = self.d_set.attr_cols
        # self.p_matrix = np.ones((self.d_set.column_size, 3), dtype=float)
        # fetch previous p_matrix from memory
        grp = 'dataset/' + self.d_set.step_name + '/p_matrix'
        p_matrix = self.d_set.read_h5_dataset(grp)
        if np.sum(p_matrix) > 0:
            self.p_matrix = p_matrix
        else:
            self.p_matrix = np.ones((self.d_set.column_size, 3), dtype=float)
        self.d_set.update_gp_attributes(attr_data)

    def validate_gp(self, pattern):
        # pattern = [('2', '+'), ('4', '+')]
        min_supp = self.d_set.thd_supp
        gen_pattern = GP()
        bin_data = np.array([])

        for gi in pattern.gradual_items:
            if self.d_set.invalid_bins.size > 0 and np.any(np.isin(self.d_set.invalid_bins, gi.gradual_item)):
                continue
            else:
                grp = 'dataset/' + self.d_set.step_name + '/valid_bins/' + gi.as_string()
                temp = self.d_set.read_h5_dataset(grp)
                if bin_data.size <= 0:
                    bin_data = np.array([temp, temp])
                    gen_pattern.add_gradual_item(gi)
                else:
                    bin_data[1] = temp
                    temp_bin, supp = self.bin_and(bin_data, self.d_set.attr_size)
                    if supp >= min_supp:
                        bin_data[0] = temp_bin
                        gen_pattern.add_gradual_item(gi)
                        gen_pattern.set_support(supp)
        if len(gen_pattern.gradual_items) <= 1:
            tgp = TGP(gp=pattern)
            return tgp
        else:
            # t_lag = FuzzyMF.calculate_time_lag(FuzzyMF.get_patten_indices(bin_data[0]), t_diffs, min_supp)
            t_lag = calculate_time_lag(bin_data[0], self.time_diffs)
            if t_lag.support <= 0:
                gen_pattern.set_support(0)
            tgp = TGP(gp=gen_pattern, t_lag=t_lag)
            return tgp


class T_GradACO_h5(T_GradACO):

    def __init__(self, f_path, eq, ref_item, min_sup, min_rep, cores):
        # For tgraank
        # self.d_set = d_set
        self.d_set = Dataset_h5(f_path, min_sup=min_sup, eq=eq)
        self.d_set.init_h5_groups()
        cols = self.d_set.time_cols
        if len(cols) > 0:
            print("Dataset Ok")
            self.time_ok = True
            self.time_cols = cols
            self.min_sup = min_sup
            self.ref_item = ref_item
            self.d_set.data = self.d_set.read_h5_dataset('dataset/data')
            self.d_set.data = np.array(self.d_set.data).astype('U')
            self.max_step = self.get_max_step(min_rep)
            self.orig_attr_data = self.d_set.data.copy().T
            if cores > 1:
                self.cores = cores
            else:
                self.cores = Profile.get_num_cores()
        else:
            print("Dataset Error")
            self.time_ok = False
            self.time_cols = []
            raise Exception('No date-time data found')

    def fetch_patterns(self, step):
        step += 1  # because for-loop is not inclusive from range: 0 - max_step
        # 1. Transform data
        d_set = self.d_set
        attr_data, time_diffs = self.transform_data(step)

        # 2. Execute aco-graank for each transformation
        ac = GradACOt_h5(d_set, attr_data, time_diffs)
        list_gp = ac.run_ant_colony()
        # print("\nPheromone Matrix")
        # print(ac.p_matrix)
        if len(list_gp) > 0:
            return list_gp
        return False
