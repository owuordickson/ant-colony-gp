# -*- coding: utf-8 -*-
"""
@author: "Dickson Owuor"
@credits: "Joseph Orero and Anne Laurent,"
@license: "MIT"
@version: "2.4"
@email: "owuordickson@gmail.com"
@created: "19 November 2019"
@modified: "11 June 2020"

Description: updated version that uses aco-graank and parallel multi-processing

"""


import numpy as np
import h5py
from pathlib import Path
import os
import multiprocessing as mp
from src.algorithms.ant_colony.aco_grad_v2 import GradACO
from src.algorithms.common.fuzzy_mf import calculate_time_lag
from src.algorithms.common.gp import GP, TGP
from src.algorithms.common.dataset import Dataset
#from src.algorithms.ant_colony.cython.cyt_aco_grad import GradACO
#from src.algorithms.common.cython.cyt_dataset import Dataset
from src.algorithms.common.profile_cpu import Profile


class Dataset_t(Dataset):

    def __init__(self, file_path, min_sup=0, eq=False):
        self.h5_file = str(Path(file_path).stem) + str('.h5')
        if os.path.exists(self.h5_file):
            print("Fetching data from h5 file")
            h5f = h5py.File(self.h5_file, 'r')
            self.title = h5f['dataset/title'][:]
            self.time_cols = h5f['dataset/time_cols'][:]
            self.attr_cols = h5f['dataset/attr_cols'][:]
            size = h5f['dataset/size'][:]
            self.column_size = size[0]
            self.size = size[1]
            self.attr_size = size[2]
            self.step_name = 'step_' + str(int(self.size - self.attr_size))
            self.invalid_bins = h5f['dataset/' + self.step_name + '/invalid_bins'][:]
            h5f.close()
            self.thd_supp = min_sup
            self.equal = eq
            self.data = None
        else:
            data = Dataset.read_csv(file_path)
            if len(data) <= 1:
                self.data = np.array([])
                print("csv file read error")
                raise Exception("Unable to read csv file or file has no data")
            else:
                print("Data fetched from csv file")
                self.data = np.array([])
                self.title = self.get_title(data)  # optimized (numpy)
                self.time_cols = self.get_time_cols()  # optimized (numpy)
                self.attr_cols = self.get_attributes()  # optimized (numpy)
                self.column_size = self.get_attribute_no()  # optimized (numpy)
                self.size = self.get_size()  # optimized (numpy)
                self.attr_size = 0
                self.step_name = ''
                self.thd_supp = min_sup
                self.equal = eq
                self.invalid_bins = np.array([])
                data = None


class GradACOt (GradACO):

    def __init__(self, d_set, attr_data, t_diffs):
        self.d_set = d_set
        self.time_diffs = t_diffs
        self.attr_index = self.d_set.attr_cols
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
            tgp = TGP(gp=gen_pattern, t_lag=t_lag)
            return tgp


class T_GradACO:

    def __init__(self, f_path, eq, ref_item, min_sup, min_rep, cores):
        # For tgraank
        # self.d_set = d_set
        self.d_set = Dataset_t(f_path, min_sup=min_sup, eq=eq)
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
            self.cores = cores
        else:
            print("Dataset Error")
            self.time_ok = False
            self.time_cols = []
            raise Exception('No date-time data found')

    def get_max_step(self, min_rep):  # optimized
        all_rows = len(self.d_set.data)
        return all_rows - int(min_rep * all_rows)

    def run_tgraank(self, parallel=False):
        if parallel:
            # implement parallel multi-processing
            if self.cores > 1:
                num_cores = self.cores
            else:
                num_cores = Profile.get_num_cores()

            self.cores = num_cores
            steps = range(self.max_step)
            # pool = mp.Pool(num_cores)
            with mp.Pool(num_cores) as pool:
                patterns = pool.map(self.fetch_patterns, steps)
                # pool.close()
                # pool.join()
            return patterns
        else:
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
        # d_set.update_attributes(attr_data)
        ac = GradACOt(d_set, attr_data, time_diffs)
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
                stamp_1 = Dataset.get_timestamp(temp_1)
                stamp_2 = Dataset.get_timestamp(temp_2)
                if (not stamp_1) or (not stamp_2):
                    return False, [i + 1, i + step + 1]
                time_diff = (stamp_2 - stamp_1)
                # index = tuple([i, i + step])
                # time_diffs.append([time_diff, index])
                time_diffs.append([time_diff, i])
        return True, np.array(time_diffs)
