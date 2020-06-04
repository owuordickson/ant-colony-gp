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
import os
import json
# from cython.parallel import prange


class Dataset:

    def __init__(self, file_path, min_sup=0, eq=False, init=True):
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
            self.column_size = self.get_attribute_no()  # optimized (cdef)
            self.size = self.get_size()  # optimized (cdef)
            self.attr_size = 0
            self.thd_supp = min_sup
            self.equal = eq
            # self.valid_bins = np.array([])
            self.valid_gi_paths = np.array([])
            self.invalid_bins = np.array([])
            self.gen_paths = list()
            if init:
                self.init_attributes()

    def get_size(self):
        size = self.data.shape[0]
        if self.title.size > 0:
            size += 1
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
        if has_title:
            keys = np.arange(len(data[0]))
            values = data[0]
            title = np.rec.fromarrays((keys, values), names=('key', 'value'))
            # del data[0]
            data = np.delete(data, 0, 0)
            # convert csv data into array
            self.data = np.asarray(data)
            return np.array(title)
        else:
            self.data = np.asarray(data)
            return np.array([])

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

    def get_bin(self, gi_path):
        return Dataset.read_json(gi_path)

    def clean_memory(self):
        for gi_obj in self.valid_gi_paths:
            Dataset.delete_file(gi_obj[1])

        for file in self.gen_paths:
            Dataset.delete_file(file)

        self.valid_gi_paths = np.array([])
        self.gen_paths = list()

    def init_attributes(self):
        # (check) implement parallel multiprocessing
        # transpose csv array data
        # if attr:
        # r, c = self.data.shape
        attr_data = self.data.T
        # attr_data = np.transpose(self.data)
        # self.attr_size = attr_data.shape[1]
        self.attr_size = len(attr_data[self.attr_cols[0]])
        self.construct_bins(attr_data)

    def update_attributes(self, attr_data):
        self.attr_size = len(attr_data[self.attr_cols[0]])
        self.construct_bins(attr_data)

    def construct_bins(self, attr_data):
        # execute binary rank to calculate support of pattern
        # valid_bins = list()  # numpy is very slow for append operations
        n = self.attr_size
        valid_paths = list()
        invalid_bins = list()
        for col in self.attr_cols:
            col_data = np.array(attr_data[col], dtype=float)
            incr = tuple([col, '+'])
            decr = tuple([col, '-'])
            temp_pos, temp_neg = Dataset.bin_rank(col_data, equal=self.equal)
            supp = float(np.sum(temp_pos)) / float(n * (n - 1.0) / 2.0)

            if supp < self.thd_supp:
                invalid_bins.append(incr)
                invalid_bins.append(decr)
            else:
                # k = np.count_nonzero(np.isnan(col_data))
                path_pos = 'gi_' + str(col) + 'pos' + str(n) + '.json'
                path_neg = 'gi_' + str(col) + 'neg' + str(n) + '.json'
                content_pos = {"gi": [int(col), '+'],
                               "bin": temp_pos.tolist(), "support": supp}
                content_neg = {"gi": [int(col), '-'],
                               "bin": temp_neg.tolist(), "support": supp}
                Dataset.write_file(json.dumps(content_pos), path_pos)
                Dataset.write_file(json.dumps(content_neg), path_neg)
                valid_paths.append([incr, path_pos])
                valid_paths.append([decr, path_neg])
        self.valid_gi_paths = np.asarray(valid_paths)
        self.invalid_bins = np.array(invalid_bins, dtype='i, O')
        # self.data = np.array([])

    @staticmethod
    def bin_rank(arr, equal=False):
        with np.errstate(invalid='ignore'):
            if not equal:
                temp_pos = arr < arr[:, np.newaxis]
            else:
                temp_pos = arr <= arr[:, np.newaxis]
                np.fill_diagonal(temp_pos, 0)
            temp_neg = temp_pos.T
            return temp_pos, temp_neg

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
    def read_json(file):
        with open(file, 'r') as f:
            data = json.load(f)
        return data

    @staticmethod
    def write_file(data, path):
        with open(path, 'w') as f:
            f.write(data)
            f.close()

    @staticmethod
    def delete_file(file):
        if os.path.exists(file):
            os.remove(file)

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
