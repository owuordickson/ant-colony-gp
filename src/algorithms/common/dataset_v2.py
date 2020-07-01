# -*- coding: utf-8 -*-
"""
@author: "Dickson Owuor"
@credits: "Anne Laurent"
@license: "MIT"
@version: "1.0"
@email: "owuordickson@gmail.com"
@created: "23 June 2020"

Changes
-------
1. This frees primary memory from storing nx1 matrices
2. Fetch all binaries during initialization
3. Replaced loops for fetching binary rank with numpy function

"""
import csv
from dateutil.parser import parse
import time
import numpy as np
import pandas as pd
import gc
from .gp import GI, GP


class Dataset:

    def __init__(self, file_path, min_sup=0, eq=False):
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
            data = None
            self.cost_matrix = np.zeros((self.column_size, 2), dtype=int)
            self.encoded_data = np.array([])
            # self.init_attributes()

    def get_size(self):
        size = self.data.shape[0]
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

    def init_attributes(self):
        # (check) implement parallel multiprocessing
        # transpose csv array data
        attr_data = self.data.copy().T
        self.attr_size = len(attr_data[self.attr_cols[0]])
        # construct and store 1-item_set valid bins
        self.construct_bins_v2(attr_data)
        attr_data = None
        gc.collect()

    def update_attributes(self, attr_data):
        self.attr_size = len(attr_data[self.attr_cols[0]])
        # self.construct_bins_v1(attr_data)
        self.construct_bins(attr_data)
        gc.collect()

    def construct_bins(self, attr_data):
        # execute binary rank to calculate support of pattern
        # valid_bins = list()  # numpy is very slow for append operations
        n = self.attr_size
        valid_idxs = list()
        invalid_bins = list()
        for col in self.attr_cols:
            col_data = np.array(attr_data[col], dtype=float)
            incr = np.array((col, '+'), dtype='i, S1')
            decr = np.array((col, '-'), dtype='i, S1')
            # temp_pos = Dataset.bin_rank(col_data, equal=self.equal)
            temp_pos, supp = Dataset.index_rank(col_data, n)
            # supp = float(len(temp_pos) / n)

            if supp < self.thd_supp:
                invalid_bins.append(incr)
                invalid_bins.append(decr)
            else:
                temp_neg = temp_pos[::-1]
                valid_idxs.append(np.array([incr.tolist(), temp_pos]))
                valid_idxs.append(np.array([decr.tolist(), temp_neg]))
        self.valid_idxs = np.array(valid_idxs)
        self.invalid_bins = np.array(invalid_bins)

    def construct_bins_v2(self, attr_data):
        # Encoding data for Depth-First Search
        encode_type = np.dtype([('id', 'i'),
                                ('seq', 'i, i'),
                                ('pattern', [('col', 'i'),
                                             ('var', 'i')],
                                 (len(self.attr_cols),)),
                                ('cost', 'f')
                                ])
        self.encoded_data = np.array(self.encode_data(attr_data),
                                     dtype=encode_type)
        self.update_cost()
        # print(self.encoded_data)
        print(self.cost_matrix)
        gc.collect()

    def encode_data(self, attr_data):
        size = self.attr_size
        encoded_data = list()
        for i in range(size):
            for j in range(i, size):
                if i != j:
                    gp = []
                    for col in self.attr_cols:
                        row_i = attr_data[col][i]
                        row_j = attr_data[col][j]
                        if row_i > row_j:
                            # gp.append(np.array((col, 1), dtype='i, i'))
                            gp.append(tuple([col, 1]))
                            self.cost_matrix[col][0] += 1
                        elif row_i < row_j:
                            # gp.append(np.array((col, -1), dtype='i, i'))
                            gp.append(tuple([col, -1]))
                            self.cost_matrix[col][1] += 1
                        else:
                            # gp.append(np.array((col, 0), dtype='i, i'))
                            gp.append(tuple([col, 0]))
                    encoded_data.append([(i, (i, j), gp, size)])
        return encoded_data

    def update_cost(self):
        size = self.attr_size
        for obj in self.encoded_data:
            gp = obj['pattern'][0]
            cost = 0
            for gi in gp:
                if gi[1] == 1:
                    cost += self.cost_matrix[gi[0]][0]
                elif gi[1] == -1:
                    cost += self.cost_matrix[gi[0]][1]
            if cost > 0:
                cost = size / cost
                obj['cost'][0] = cost
        return self.cost_matrix

    @staticmethod
    def index_rank(arr, n):
        sort_idx = np.argsort(arr)
        # index = np.arange(len(arr))
        # values, cnt = np.unique(arr, return_counts=True)
        # temp_idx = np.split(index[sort_idx], np.cumsum(cnt[:-1]))
        # temp_idx = np.unique(arr)  # slower than pandas
        temp_idx = pd.unique(arr)  # faster O(N)
        supp = float(len(temp_idx) / n)
        # print(str(arr) + ' - ' + str(temp_idx) + ' : ' + str(supp) + '\n')
        return sort_idx, supp

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
