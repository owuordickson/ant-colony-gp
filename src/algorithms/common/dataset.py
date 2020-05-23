# -*- coding: utf-8 -*-
"""
@author: "Dickson Owuor"
@credits: "Thomas Runkler, Edmond Menya, and Anne Laurent"
@license: "MIT"
@version: "1.0"
@email: "owuordickson@gmail.com"
@created: "12 July 2019"

"""
import csv
from dateutil.parser import parse
import time
import numpy as np
import os
import json


class Dataset:

    def __init__(self, file_path):
        data = Dataset.read_csv(file_path)
        if len(data) == 0:
            self.data = np.array([])
            print("csv file read error")
            raise Exception("Unable to read csv file")
        else:
            print("Data fetched from csv file")
            self.data = np.array([])
            self.title = self.get_title(data)  # optimized (numpy)
            self.time_cols = self.get_time_cols()  # optimized (numpy)
            self.attr_cols = self.get_attributes()  # optimized (numpy)
            self.column_size = self.get_attribute_no()  # optimized (cdef)
            self.size = self.get_size()  # optimized (cdef)
            self.attr_size = 0
            self.thd_supp = 0
            self.equal = False
            # self.valid_bins = np.array([])  # optimized (numpy & numba)
            self.valid_gi_paths = np.array([])
            self.invalid_bins = list()

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
        attr_cols = np.delete(all_cols, self.time_cols)
        return attr_cols

    def get_time_cols(self):
        time_cols = list()
        # for k in range(10, len(self.data[0])):
        #    time_cols.append(k)
        # time_cols.append(0)
        # time_cols.append(1)
        # time_cols.append(2)
        # time_cols.append(3)
        # time_cols.append(4)
        # time_cols.append(5)
        # time_cols.append(6)
        # time_cols.append(7)
        # time_cols.append(8)
        # time_cols.append(9)
        for i in range(len(self.data[0])):  # check every column for time format
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
            Dataset.delete_file(gi_obj.path)

    def init_attributes(self, min_sup, eq):
        # (check) implement parallel multiprocessing
        # transpose csv array data
        self.thd_supp = min_sup
        self.equal = eq
        # r, c = self.data.shape
        attr_data = self.data.T
        # attr_data = np.transpose(self.data)
        self.attr_size = attr_data.shape[1]
        self.construct_bins(attr_data)

    def construct_bins(self, attr_data):
        # execute binary rank to calculate support of pattern
        # valid_bins = list()  # numpy is very slow for append operations
        n = self.attr_size
        valid_paths = list()
        for col in self.attr_cols:
            col_data = np.array(attr_data[col], dtype=float)
            incr = tuple([col, '+'])
            decr = tuple([col, '-'])
            temp_pos, temp_neg = Dataset.bin_rank(col_data, equal=self.equal)
            supp = float(np.sum(temp_pos)) / float(n * (n - 1.0) / 2.0)

            if supp < self.thd_supp:
                self.invalid_bins.append(incr)
                self.invalid_bins.append(decr)
            else:
                path_pos = 'gi_' + str(col) + 'pos' + '.json'
                path_neg = 'gi_' + str(col) + 'neg' + '.json'
                content_pos = {"gi": [int(col), '+'],
                               "bin": temp_pos.tolist(), "support": supp}
                content_neg = {"gi": [int(col), '-'],
                               "bin": temp_neg.tolist(), "support": supp}
                Dataset.write_file(json.dumps(content_pos), path_pos)
                Dataset.write_file(json.dumps(content_neg), path_neg)
                if len(valid_paths) > 0:
                    valid_paths[0].append(incr)
                    valid_paths[1].append(path_pos)
                    valid_paths[0].append(decr)
                    valid_paths[1].append(path_neg)
                else:
                    valid_paths.append([incr])
                    valid_paths.append([path_pos])
                    valid_paths[0].append(decr)
                    valid_paths[1].append(path_neg)
        valid_paths = np.asarray(valid_paths)
        self.valid_gi_paths = np.rec.fromarrays((valid_paths[0], valid_paths[1]), names=('gi', 'path'))
        self.data = np.array([])

    @staticmethod
    def bin_rank(arr, equal=False):
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
