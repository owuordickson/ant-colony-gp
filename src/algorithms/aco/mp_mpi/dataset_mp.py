# -*- coding: utf-8 -*-
"""
@author: "Dickson Owuor"
@credits: "Anne Laurent"
@license: "MIT"
@version: "2.2"
@email: "owuordickson@gmail.com"
@created: "22 June 2020"

Changes
-------
1. save attribute gradual item sets binaries as json file and retrieve them as dicts
   - this frees primary memory from storing nxn matrices
2. Fetch all binaries during initialization
3. Replaced loops for fetching binary rank with numpy function

"""


import numpy as np
import multiprocessing as mp
from src.algorithms.common.dataset_bfs import Dataset


class Dataset_mp(Dataset):

    def __init__(self, file_path, min_sup=0, eq=False, cores=1):
        super().__init__(file_path, min_sup, eq)
        self.attr_data = np.array([])
        self.cores = cores

    def construct_bins(self, attr_data):
        # execute binary rank to calculate support of pattern
        # valid_bins = list()  # numpy is very slow for append operations
        n = self.attr_size
        self.attr_data = attr_data
        valid_bins = list()
        invalid_bins = list()
        with mp.Pool(self.cores) as pool:
            lst_temp = pool.map(self.fetch_bins, self.attr_cols)

        for temp in lst_temp:
            if temp[0]:
                for obj in temp[0]:
                    invalid_bins.append(obj)
            else:
                for obj in temp[1]:
                    valid_bins.append(obj)
        self.valid_bins = np.array(valid_bins)
        self.invalid_bins = np.array(invalid_bins)

    def fetch_bins(self, col):
        valid_bins = list()
        invalid_bins = list()
        n = self.attr_size
        col_data = np.array(self.attr_data[col], dtype=float)
        incr = np.array((col, '+'), dtype='i, S1')
        decr = np.array((col, '-'), dtype='i, S1')
        temp_pos = Dataset.bin_rank(col_data, equal=self.equal)
        supp = float(np.sum(temp_pos)) / float(n * (n - 1.0) / 2.0)

        if supp < self.thd_supp:
            invalid_bins.append(incr)
            invalid_bins.append(decr)
            return [invalid_bins, False]
        else:
            valid_bins.append(np.array([incr.tolist(), temp_pos]))
            valid_bins.append(np.array([decr.tolist(), temp_pos.T]))
            return [False, valid_bins]
