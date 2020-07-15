# -*- coding: utf-8 -*-
"""
@author: "Dickson Owuor"
@credits: "Anne Laurent"
@license: "MIT"
@version: "3.0"
@email: "owuordickson@gmail.com"
@created: "12 July 2019"
@modified: "20 June 2020"

Changes
-------
1. save attribute gradual item sets binaries as json file and retrieve them as dicts
   - this frees primary memory from storing nxn matrices
2. Fetch all binaries during initialization
3. Replaced loops for fetching binary rank with numpy function
4. Used HDF5 storage

"""

import numpy as np
from pathlib import Path
import h5py
import gc
import os
from ..dataset_bfs import Dataset

# import tables
# from src.algorithms.common.gp import GI, GP
# from cython.parallel import prange


class Dataset_h5(Dataset):

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
                data = None
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
            #    self.init_attributes()

    def init_gp_attributes(self):
        # (check) implement parallel multiprocessing
        if self.data is not None:
            # transpose csv array data
            attr_data = self.data.copy().T
            self.attr_size = len(attr_data[self.attr_cols[0]])
            # create h5 groups to store class attributes
            self.init_h5_groups()
            # construct and store 1-item_set valid bins
            self.construct_bins(attr_data)
            attr_data = None
        gc.collect()

    def construct_bins(self, attr_data):
        # execute binary rank to calculate support of pattern
        n = self.attr_size
        self.step_name = 'step_' + str(int(self.size - self.attr_size))
        invalid_bins = list()
        for col in self.attr_cols:
            col_data = np.array(attr_data[col], dtype=float)
            incr = np.array((col, '+'), dtype='i, S1')
            decr = np.array((col, '-'), dtype='i, S1')
            temp_pos = Dataset.bin_rank(col_data, equal=self.equal)
            supp = float(np.sum(temp_pos)) / float(n * (n - 1.0) / 2.0)

            if supp < self.thd_supp:
                invalid_bins.append(incr)
                invalid_bins.append(decr)
            else:
                grp = 'dataset/' + self.step_name + '/valid_bins/' + str(col) + '_pos'
                self.add_h5_dataset(grp, temp_pos)
                grp = 'dataset/' + self.step_name + '/valid_bins/' + str(col) + '_neg'
                self.add_h5_dataset(grp, temp_pos.T)
        self.invalid_bins = np.array(invalid_bins)
        grp = 'dataset/' + self.step_name + '/invalid_bins'
        self.add_h5_dataset(grp, self.invalid_bins)
        data_size = np.array([self.column_size, self.size, self.attr_size])
        self.add_h5_dataset('dataset/size', data_size)
        gc.collect()

    def init_h5_groups(self):
        if os.path.exists(self.h5_file):
            pass
        else:
            h5f = h5py.File(self.h5_file, 'w')
            grp = h5f.require_group('dataset')
            grp.create_dataset('title', data=self.title)
            data = np.array(self.data.copy()).astype('S')
            grp.create_dataset('data', data=data, compression="gzip", compression_opts=9)
            grp.create_dataset('time_cols', data=self.time_cols)
            grp.create_dataset('attr_cols', data=self.attr_cols)
            h5f.close()
            data = None
            self.data = None

    def read_h5_dataset(self, group):
        temp = np.array([])
        h5f = h5py.File(self.h5_file, 'r')
        if group in h5f:
            temp = h5f[group][:]
        h5f.close()
        return temp

    def add_h5_dataset(self, group, data):
        h5f = h5py.File(self.h5_file, 'r+')
        if group in h5f:
            del h5f[group]
        h5f.create_dataset(group, data=data, compression="gzip", compression_opts=9)
        h5f.close()
