# -*- coding: utf-8 -*-
"""
@author: "Dickson Owuor"
@credits: "Anne Laurent"
@license: "MIT"
@version: "2.2"
@email: "owuordickson@gmail.com"
@created: "12 July 2019"
@modified: "25 May 2020"

Changes
-------
1. save attribute gradual item sets binaries as json file and retrieve them as dicts
   - this frees primary memory from storing nxn matrices
2. Fetch all binaries during initialization
3. Replaced loops for fetching binary rank with numpy function

"""
import csv
from dateutil.parser import parse
import time
import numpy as np
from pathlib import Path
import h5py
import gc
import os

# import tables
# from src.algorithms.common.gp import GI, GP
# from cython.parallel import prange


class Dataset:

    def __init__(self, file_path, min_sup=0, eq=False, init=True):
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
            self.table_name = 'step_' + str(int(self.size - self.attr_size))
            self.invalid_bins = h5f['dataset/' + self.table_name + '/invalid_bins'][:]
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
                self.table_name = ''
                self.thd_supp = min_sup
                self.equal = eq
                self.invalid_bins = np.array([])
                data = None
                self.init_attributes(init)

    def get_size(self):
        size = self.data.shape[0]
        # if self.title.size > 0:
        #    size += 1
        return size

    def get_attribute_no(self):
        count = self.data.shape[1]
        return count

    def get_title(self, data):
        # data = self.raw_data
        if data[0][0].replace('.', '', 1).isdigit() or data[0][0].isdigit():
            title = self.convert_data_to_array(data)
            return title
        else:
            if data[0][1].replace('.', '', 1).isdigit() or data[0][1].isdigit():
                title = self.convert_data_to_array(data)
                return title
            else:
                title = self.convert_data_to_array(data, has_title=True)
                return title

    def convert_data_to_array(self, data, has_title=False):
        # convert csv data into array
        title = np.array([])
        if has_title:
            keys = np.arange(len(data[0]))
            values = np.array(data[0], dtype='S')
            title = np.rec.fromarrays((keys, values), names=('key', 'value'))
            data = np.delete(data, 0, 0)
        # convert csv data into array
        self.data = np.asarray(data)
        return title

    def get_attributes(self):
        all_cols = np.arange(self.get_attribute_no())
        # attr_cols = np.delete(all_cols, self.time_cols)
        attr_cols = np.setdiff1d(all_cols, self.time_cols)
        return attr_cols

    def get_time_cols(self):
        time_cols = list()
        # for k in range(10, len(self.data[0])):
        #    time_cols.append(k)
        # time_cols.append(0)
        n = len(self.data[0])
        for i in range(n):  # check every column for time format
            row_data = str(self.data[0][i])
            try:
                time_ok, t_stamp = Dataset.test_time(row_data)
                if time_ok:
                    time_cols.append(i)
            except ValueError:
                continue
        if len(time_cols) > 0:
            return np.array(time_cols)
        else:
            return np.array([])

    def init_attributes(self, init):
        # (check) implement parallel multiprocessing
        # create h5 groups to store class attributes
        self.init_h5_groups()
        if init:
            # transpose csv array data
            attr_data = self.data.copy().T
            self.attr_size = len(attr_data[self.attr_cols[0]])
            # self.construct_bins_v1(attr_data)
            self.construct_bins_v4(attr_data)
            attr_data = None
        # else:
            # 1. do not construct bins (due to transformation)
        self.data = None
        gc.collect()

    def update_attributes(self, attr_data):
        self.attr_size = len(attr_data[self.attr_cols[0]])
        # self.construct_bins_v1(attr_data)
        self.construct_bins_v4(attr_data)
        gc.collect()

    def construct_bins_v1(self, attr_data):
        # execute binary rank to calculate support of pattern
        # valid_bins = list()  # numpy is very slow for append operations
        n = self.attr_size
        valid_bins = list()
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
                valid_bins.append(np.array([incr.tolist(), temp_pos]))
                valid_bins.append(np.array([decr.tolist(), temp_pos.T]))
        self.valid_bins = np.array(valid_bins)
        self.invalid_bins = np.array(invalid_bins)

    def construct_bins_v4(self, attr_data):
        # execute binary rank to calculate support of pattern
        n = self.attr_size
        self.table_name = 'step_' + str(int(self.size - self.attr_size))
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
                grp = 'dataset/' + self.table_name + '/valid_bins/' + str(col) + '_pos'
                self.add_h5_dataset(grp, temp_pos)
                grp = 'dataset/' + self.table_name + '/valid_bins/' + str(col) + '_neg'
                self.add_h5_dataset(grp, temp_pos.T)
        self.invalid_bins = np.array(invalid_bins)
        grp = 'dataset/' + self.table_name + '/invalid_bins'
        self.add_h5_dataset(grp, self.invalid_bins)
        data_size = np.array([self.column_size, self.size, self.attr_size])
        self.add_h5_dataset('dataset/size', data_size)

    def init_h5_groups(self):
        with h5py.File(self.h5_file, 'w') as h5f:
            grp = h5f.require_group('dataset')
            grp.create_dataset('title', data=self.title)
            data = np.array(self.data.copy()).astype('S')
            grp.create_dataset('data', data=data)
            grp.create_dataset('time_cols', data=self.time_cols)
            grp.create_dataset('attr_cols', data=self.attr_cols)
            h5f.close()
            data = None

    def read_h5_dataset(self, group):
        h5f = h5py.File(self.h5_file, 'r')
        if group in h5f:
            temp = h5f[group][:]
            h5f.close()
            return temp
        else:
            return np.array([])

    def add_h5_dataset(self, group, data):
        h5f = h5py.File(self.h5_file, 'r+')
        if group in h5f:
            del h5f[group]
        h5f.create_dataset(group, data=data)
        h5f.close()

    @staticmethod
    def bin_rank(arr, equal=False):
        with np.errstate(invalid='ignore'):
            if not equal:
                temp_pos = arr < arr[:, np.newaxis]
            else:
                temp_pos = arr <= arr[:, np.newaxis]
                np.fill_diagonal(temp_pos, 0)
            return temp_pos

    @staticmethod
    def read_csv(file):
        # 1. retrieve data-set from file
        with open(file, 'r') as f:
            dialect = csv.Sniffer().sniff(f.readline(), delimiters=";,' '\t")
            f.seek(0)
            reader = csv.reader(f, dialect)
            temp = list(reader)
            f.close()
        return temp

    @staticmethod
    def test_time(date_str):
        # add all the possible formats
        try:
            if type(int(date_str)):
                return False, False
        except ValueError:
            try:
                if type(float(date_str)):
                    return False, False
            except ValueError:
                try:
                    date_time = parse(date_str)
                    t_stamp = time.mktime(date_time.timetuple())
                    return True, t_stamp
                except ValueError:
                    raise ValueError('no valid date-time format found')

    @staticmethod
    def get_timestamp(time_data):
        try:
            ok, stamp = Dataset.test_time(time_data)
            if ok:
                return stamp
            else:
                return False
        except ValueError:
            return False
