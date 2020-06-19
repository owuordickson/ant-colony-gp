# -*- coding: utf-8 -*-
"""
@author: "Dickson Owuor"
@credits: "Thomas Runkler and Anne Laurent,"
@license: "MIT"
@version: "3.0"
@email: "owuordickson@gmail.com"
@created: "19 November 2019"
@modified: "15 June 2020"

Description: updated version that uses aco-graank and parallel multi-processing

"""


import numpy as np
import gc
from ..aco_grad import GradACO
from ...common.fuzzy_mf import calculate_time_lag
from ...common.gp import GP, TGP
from ...common.hdf5.dataset_h5 import Dataset_h5
#from src.algorithms.ant_colony.cython.cyt_aco_grad import GradACO
#from src.algorithms.common.cython.cyt_dataset import Dataset


class Dataset_t(Dataset_h5):

    def __init__(self, file_path=None, min_sup=None, eq=False, h5f=None):
        if h5f is not None:
            print("Fetching data from h5 file")

            self.thd_supp = min_sup
            self.equal = eq
            self.title = h5f['dataset/title'][:]
            self.time_cols = h5f['dataset/time_cols'][:]
            self.attr_cols = h5f['dataset/attr_cols'][:]
            size = h5f['dataset/size'][:]
            self.column_size = size[0]
            self.size = size[1]
            self.attr_size = 0
            self.step_name = ''
            self.invalid_bins = np.array([])  # to be removed

            self.data = h5f['dataset/data'][:]
            self.data = np.array(self.data).astype('U')
        else:
            data = Dataset_t.read_csv(file_path)
            if len(data) <= 1:
                self.data = np.array([])
                data = None
                print("csv file read error")
                raise Exception("Unable to read csv file or file has no data")
            else:
                print("Data fetched from csv file")
                self.data = np.array([])
                self.thd_supp = min_sup
                self.equal = eq
                self.title = self.get_title(data)  # optimized (numpy)
                self.time_cols = self.get_time_cols()  # optimized (numpy)
                self.attr_cols = self.get_attributes()  # optimized (numpy)
                self.column_size = self.get_attribute_no()  # optimized (numpy)
                self.size = self.get_size()  # optimized (numpy)
                self.attr_size = 0
                self.step_name = ''
                self.invalid_bins = np.array([])
                data = None


class GradACOt (GradACO):

    def __init__(self, d_set, t_diffs, h5f):
        self.d_set = d_set
        self.time_diffs = t_diffs
        self.h5f = h5f
        self.attr_index = self.d_set.attr_cols
        grp = 'dataset/' + self.d_set.step_name + '/p_matrix'
        if grp in h5f:
            p_matrix = h5f[grp][:]
        else:
            p_matrix = np.array([])
        if np.sum(p_matrix) > 0:
            self.p_matrix = p_matrix
        else:
            self.p_matrix = np.ones((self.d_set.column_size, 3), dtype=float)

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

    def validate_gp(self, pattern):  # needs to read from h5 file
        # pattern = [('2', '+'), ('4', '+')]
        min_supp = self.d_set.thd_supp
        gen_pattern = GP()
        bin_data = np.array([])

        for gi in pattern.gradual_items:
            if self.d_set.invalid_bins.size > 0 and np.any(np.isin(self.d_set.invalid_bins, gi.gradual_item)):
                continue
            else:
                ds = 'dataset/' + self.d_set.step_name + '/valid_bins'# + gi.as_string()
                if ds in self.h5f:
                    temp = self.h5f[ds][int(gi.attribute_col)][:]
                else:
                    continue
                    # temp = np.array([])
                # for obj in self.d_set.valid_bins:
                #    if obj[0] == gi.gradual_item:
                #        temp = obj[1]
                #        break
                if bin_data.size <= 0:
                    bin_data = np.array([temp, np.array([])])
                    gen_pattern.add_gradual_item(gi)
                else:
                    bin_data[1] = temp
                    temp_bin, supp = self.bin_and(bin_data, self.d_set.attr_size)
                    if supp >= min_supp:
                        bin_data[0] = temp_bin
                        gen_pattern.add_gradual_item(gi)
                        gen_pattern.set_support(supp)
        gc.collect()
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


class T_GradACO:

    def __init__(self, d_set, ref_item, min_rep):
        # For tgraank
        cols = d_set.time_cols
        if len(cols) > 0:
            print("Dataset Ok")
            self.time_ok = True
            self.d_set = d_set
            self.time_cols = cols
            self.min_sup = d_set.thd_supp
            self.ref_item = ref_item
            self.max_step = self.get_max_step(min_rep)
            self.attr_data = d_set.data.copy().T
            # self.d_set.data = self.d_set.read_h5_dataset('dataset/data')
            # self.d_set.data = np.array(self.d_set.data).astype('U')
        else:
            print("Dataset Error")
            self.time_ok = False
            self.time_cols = []
            raise Exception('No date-time data found')

    def get_max_step(self, min_rep):  # optimized
        all_rows = len(self.d_set.data)
        return all_rows - int(min_rep * all_rows)

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
                    attr_data = self.attr_data

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
                stamp_1 = Dataset_t.get_timestamp(temp_1)
                stamp_2 = Dataset_t.get_timestamp(temp_2)
                if (not stamp_1) or (not stamp_2):
                    return False, [i + 1, i + step + 1]
                time_diff = (stamp_2 - stamp_1)
                # index = tuple([i, i + step])
                # time_diffs.append([time_diff, index])
                time_diffs.append([time_diff, i])
        return True, np.array(time_diffs)

    def construct_bins(self, col, n, col_data):
        # execute binary rank to calculate support of pattern
        incr = np.array((col, '+'), dtype='i, S1')
        decr = np.array((col, '-'), dtype='i, S1')
        temp_pos = Dataset_t.bin_rank(col_data, equal=self.d_set.equal)
        supp = float(np.sum(temp_pos)) / float(n * (n - 1.0) / 2.0)

        if supp < self.min_sup:
            return False, [incr, decr]
        else:
            return True, temp_pos
