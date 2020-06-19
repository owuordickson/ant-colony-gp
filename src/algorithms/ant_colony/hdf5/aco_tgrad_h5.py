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
        self.d_set.update_attributes(attr_data)

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


class T_GradACO_h5:

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

    def get_max_step(self, min_rep):  # optimized
        all_rows = len(self.d_set.data)
        return all_rows - int(min_rep * all_rows)

    def run_tgraank(self):
        patterns = list()
        for step in range(self.max_step):
            t_pattern = self.fetch_patterns(step)
            if t_pattern:
                patterns.append(t_pattern)
        return patterns

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

    def transform_data(self, step):  # optimized
        # NB: Restructure dataset based on reference item
        if self.time_ok:
            # 1. Calculate time difference using step
            ok, time_diffs = self.get_time_diffs(step)
            if not ok:
                msg = "Error: Time in row " + str(time_diffs[0]) \
                      + " or row " + str(time_diffs[1]) + " is not valid."
                raise Exception(msg)
            else:
                ref_col = self.ref_item
                if ref_col in self.time_cols:
                    msg = "Reference column is a 'date-time' attribute"
                    raise Exception(msg)
                elif (ref_col < 0) or (ref_col >= len(self.d_set.title)):
                    msg = "Reference column does not exist\nselect column between: " \
                          "0 and " + str(len(self.d_set.title) - 1)
                    raise Exception(msg)
                else:
                    # 1. Split the transpose data set into column-tuples
                    attr_data = self.orig_attr_data

                    # 2. Transform the data using (row) n+step
                    new_attr_data = list()
                    size = len(attr_data)
                    for k in range(size):
                        col_index = k
                        tuples = attr_data[k]
                        n = tuples.size
                        # temp_tuples = np.empty(n, )
                        # temp_tuples[:] = np.NaN
                        if col_index in self.time_cols:
                            # date-time attribute
                            temp_tuples = tuples[:]
                        elif col_index == ref_col:
                            # reference attribute
                            temp_tuples = tuples[0: n - step]
                        else:
                            # other attributes
                            temp_tuples = tuples[step: n]
                        # print(temp_tuples)
                        new_attr_data.append(temp_tuples)
                    return new_attr_data, time_diffs
        else:
            msg = "Fatal Error: Time format in column could not be processed"
            raise Exception(msg)

    def get_time_diffs(self, step):  # optimized
        data = self.d_set.data
        size = len(data)
        time_diffs = []
        for i in range(size):
            if i < (size - step):
                # for col in self.time_cols:
                col = self.time_cols[0]  # use only the first date-time value
                temp_1 = str(data[i][int(col)])
                temp_2 = str(data[i + step][int(col)])
                stamp_1 = Dataset_h5.get_timestamp(temp_1)
                stamp_2 = Dataset_h5.get_timestamp(temp_2)
                if (not stamp_1) or (not stamp_2):
                    return False, [i + 1, i + step + 1]
                time_diff = (stamp_2 - stamp_1)
                # index = tuple([i, i + step])
                # time_diffs.append([time_diff, index])
                time_diffs.append([time_diff, i])
        return True, np.array(time_diffs)
