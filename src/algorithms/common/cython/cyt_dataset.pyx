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
cimport numpy as np
import cython
from cython.parallel import prange


cdef struct title_struct:
    int key
    char[20] value


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.profile(True)
cdef class Dataset:

    def __cinit__(self, str file_path):
        cdef list data
        data = Dataset.read_csv(file_path)
        if len(data) == 0:
            self.d_set = np.array([])
            print("csv file read error")
            raise Exception("Unable to read csv file")
        else:
            print("Data fetched from csv file")
            self.d_set = np.array([])
            self.title = self.get_title(data)  # optimized (numpy)
            self.time_cols = self.get_time_cols()  # optimized (numpy)
            self.attr_cols = self.get_attributes()  # optimized (numpy)
            self.column_size = self.get_attribute_no()  # optimized (cdef)
            self.size = self.get_size()  # optimized (cdef)
            self.attr_size = 0
            self.thd_supp = 0
            self.equal = False
            self.valid_gi_paths = np.array([])
            self.invalid_bins = np.array([])

    cdef int get_size(self):
        cdef int size
        size = self.d_set.shape[0]
        if self.title.size > 0:
            size += 1
        return size

    cdef int get_attribute_no(self):
        cdef int count
        count = self.d_set.shape[1]
        return count

    cdef np.ndarray get_attributes(self):
        cdef np.ndarray all_cols, attr_cols
        all_cols = np.arange(self.get_attribute_no())
        attr_cols = np.setdiff1d(all_cols, self.time_cols)
        return attr_cols

    cdef np.ndarray get_title(self, list data):
        cdef np.ndarray title
        if data[0][0].replace('.', '', 1).isdigit() or data[0][0].isdigit():
            title = self.convert_data_to_array(data, False)
        else:
            if data[0][1].replace('.', '', 1).isdigit() or data[0][1].isdigit():
                title = self.convert_data_to_array(data, False)
            else:
                title = self.convert_data_to_array(data, True)
        return title

    cdef np.ndarray convert_data_to_array(self, list data, bint has_title):
        # convert csv data into array
        cdef int size
        cdef np.ndarray keys, temp_data
        cdef list values
        if has_title:
            size = len(data[0])
            keys = np.arange(size)
            values = data[0]
            title = np.rec.fromarrays((keys, values), names=('key', 'value'))
            temp_data = np.asarray(data)
            # temp_data = np.delete(np.asarray(data), 0, 0)
            # convert csv data into array
            self.d_set = np.asarray(temp_data[1:])
            return np.asarray(title)
        else:
            self.d_set = np.asarray(data)
            return np.array([])

    cdef np.ndarray get_time_cols(self):
        cdef list time_cols
        cdef str row_data
        cdef bint time_ok
        cdef float t_stamp
        cdef int n
        time_cols = list()
        n = len(self.d_set[0])
        for i in range(n):  # check every column for time format
            row_data = str(self.d_set[0][i])
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


    cpdef dict get_bin(self, str gi_path):
        return Dataset.read_json(gi_path)

    cpdef void clean_memory(self):
        for gi_obj in self.valid_gi_paths:
            Dataset.delete_file(gi_obj[1])

    cpdef void init_attributes(self, float min_sup, bint eq):
        # (check) implement parallel multiprocessing
        # transpose csv array data
        self.thd_supp = min_sup
        self.equal = eq
        attr_data = self.d_set.T
        self.attr_size = attr_data.shape[1]
        self.construct_bins(attr_data)


    cdef void construct_bins(self, np.ndarray attr_data):
        # execute binary rank to calculate support of pattern
        cdef int n
        cdef float supp
        cdef list valid_paths, invalid_bins
        cdef tuple incr, decr
        cdef str path_pos, path_neg
        cdef dict content_pos, content_neg
        cdef np.ndarray col_data, bins, temp_pos, temp_neg
        n = self.attr_size
        valid_paths = list()
        invalid_bins = list()
        for col in self.attr_cols:
            col_data = np.array(attr_data[col], dtype=float)
            incr = tuple([col, '+'])
            decr = tuple([col, '-'])
            bins = self.bin_rank(col_data)
            temp_pos = bins[0]
            temp_neg = bins[1]
            supp = float(np.sum(temp_pos)) / float(n * (n - 1.0) / 2.0)

            if supp < self.thd_supp:
                invalid_bins.append(incr)
                invalid_bins.append(decr)
            else:
                path_pos = 'gi_' + str(col) + 'pos' + '.json'
                path_neg = 'gi_' + str(col) + 'neg' + '.json'
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

    cdef np.ndarray bin_rank(self, np.ndarray arr):
        cdef np.ndarray bin_pos, bin_neg, all_bins
        if not self.equal:
            bin_pos = arr < arr[:, np.newaxis]
        else:
            bin_pos = arr <= arr[:, np.newaxis]
            np.fill_diagonal(bin_pos, 0)
        bin_neg = bin_pos.T
        all_bins = np.array([bin_pos, bin_neg])
        return all_bins

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
