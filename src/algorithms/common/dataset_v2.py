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
import gc


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
            self.cost_matrix = np.ones((self.column_size, 3), dtype=int)
            self.encoded_data = np.array([])

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

    def init_gp_attributes(self):
        # (check) implement parallel multiprocessing
        # transpose csv array data
        attr_data = self.data.copy().T
        self.attr_size = len(attr_data[self.attr_cols[0]])
        # attr_data = attr_data[self.attr_cols]
        self.construct_bins(attr_data)
        attr_data = None
        gc.collect()

    def update_gp_attributes(self, attr_data):
        self.attr_size = len(attr_data[self.attr_cols[0]])
        # self.construct_bins_v1(attr_data)
        self.construct_bins(attr_data)
        gc.collect()

    def construct_bins(self, attr_data):
        # 1. Encoding data for Depth-First Search
        # [row_i, row_j, ..., data, ...]
        self.encoded_data = np.array(self.encode_data(attr_data))

    def encode_data(self, attr_data):
        size = self.attr_size  # np.arange(self.attr_size)
        n = len(self.attr_cols) + 2
        encoded_data = list()
        for i in range(size):
            j = i + 1
            if j >= size:
                continue

            temp_arr = np.empty([n, (size - j)], dtype=int)
            temp_arr[0] = np.repeat(i, (size - j))
            temp_arr[1] = np.arange(j, size)
            k = 2
            for col in self.attr_cols:
                row_in = attr_data[col][i]
                row_js = attr_data[col][(i+1):size]
                row = np.where(row_js > row_in, 1, np.where(row_js < row_in, -1, 0))
                temp_arr[k] = row
                k += 1
                pos_cost = np.count_nonzero(row == 1)
                neg_cost = np.count_nonzero(row == -1)
                inv_cost = np.count_nonzero(row == 0)
                self.cost_matrix[col][0] += (neg_cost + inv_cost)
                self.cost_matrix[col][1] += (pos_cost + inv_cost)
                self.cost_matrix[col][2] += (pos_cost + neg_cost)
            temp_arr = temp_arr.T
            # node = np.empty([2, (size - j)], dtype=int)
            # node[0] = np.repeat(i, (size - j))
            # node[1] = np.arange(j, size)
            # temp_zip = list(zip(node.T, temp_arr))
            encoded_data.extend(temp_arr)
        gc.collect()
        return encoded_data

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
