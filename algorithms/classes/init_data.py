# -*- coding: utf-8 -*-
"""
@author: "Dickson Owuor"
@credits: "Thomas Runkler, Edmond Menya, and Anne Laurent,"
@license: "MIT"
@version: "1.0"
@email: "owuordickson@gmail.com"

"""
import csv
from dateutil.parser import parse
import time
import numpy as np
import networkx as nx
from algorithms.classes.tuple_node import TupleNode


class InitData:

    def __init__(self, file_path):
        self.raw_data = InitData.read_csv(file_path)
        if len(self.raw_data) == 0:
            self.data = False
            print("Data-set error")
            raise Exception("Unable to read csv file")
        else:
            self.data = self.raw_data
            self.title = self.get_title()
            self.attr_indxs = self.get_attributes()
            self.time_columns = self.get_time_cols()
            self.column_size = self.get_attribute_no()
            self.size = self.get_size()
            self.attr_data = []
            self.lst_bin = []
            self.lst_graph = []

    def get_size(self):
        size = len(self.raw_data)
        return size

    def get_attribute_no(self):
        count = len(self.raw_data[0])
        return count

    def get_title(self):
        data = self.raw_data
        if data[0][0].replace('.', '', 1).isdigit() or data[0][0].isdigit():
            return False
        else:
            if data[0][1].replace('.', '', 1).isdigit() or data[0][1].isdigit():
                return False
            else:
                title = []
                for i in range(len(data[0])):
                    # sub = (str(i + 1) + ' : ' + data[0][i])
                    # sub = data[0][i]
                    sub = [str(i+1), data[0][i]]
                    title.append(sub)
                del self.data[0]
                return title

    def get_attributes(self):
        attr = []
        for i in range(len(self.title)):
            temp_attr = self.title[i]
            attr.append(temp_attr[0])
        return attr

    def get_time_cols(self):
        time_cols = list()
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
                time_ok, t_stamp = InitData.test_time(row_data)
                if time_ok:
                    time_cols.append(i)
            except ValueError:
                continue
        if time_cols:
            return time_cols
        else:
            return False

    def init_bin_attributes(self, thd_supp):
        # Arrange rank attributes to generate Graph attribute
        temp = self.data
        cols = self.get_attribute_no()
        time_cols = self.get_time_cols()
        lst_raw_attrs = []
        for col in range(cols):
            if time_cols and (col in time_cols):
                # exclude date-time column
                continue
            else:
                # get all tuples of an attribute/column
                raw_tuples = []
                for row in range(len(temp)):
                    raw_tuples.append(float(temp[row][col]))
                lst_raw_attrs.append([self.title[col][0], raw_tuples])
        self.lst_bin = InitData.init_bin_rank(lst_raw_attrs)
        self.attr_data = lst_raw_attrs
        print(self.attr_data)
        n = len(lst_raw_attrs[0][1])
        rem_bins = []
        for obj in self.lst_bin:
            supp = float(np.sum(obj[1])) / float(n * (n - 1.0) / 2.0)
            if supp < thd_supp:
                rem_bins.append(obj)
        for obj in rem_bins:
            self.lst_bin.remove(obj)

    def init_graph_attributes(self, thd_supp):
        # Arrange rank attributes to generate Graph attribute
        temp = self.data
        cols = self.get_attribute_no()
        time_cols = self.get_time_cols()
        lst_graph_attrs = []
        for col in range(cols):
            if time_cols and (col in time_cols):
                # exclude date-time column
                continue
            else:
                # get all tuples of an attribute/column
                raw_tuples = []
                for row in range(len(temp)):
                    raw_tuples.append(float(temp[row][col]))
                # rank in ascending order and assign pheromones
                for d in {'+', '-'}:
                    supp, graph_attr = InitData.init_graph_rank(d, raw_tuples)
                    if supp >= thd_supp:
                        temp_attr = [self.title[col][0], d, graph_attr]
                        lst_graph_attrs.append(temp_attr)
        self.lst_graph = lst_graph_attrs

    @staticmethod
    def init_bin_rank(raw_attrs, eq=False):
        # temp_attr = [self.title[col][0], d, graph_attr]
        lst_bin = []
        n = len(raw_attrs[0][1])
        for i in range(len(raw_attrs)):
            attr_data = raw_attrs[i]
            incr = [attr_data[0], '+']
            decr = [attr_data[0], '-']
            temp_pos = np.zeros((n, n), dtype='bool')
            temp_neg = np.zeros((n, n), dtype='bool')
            var_tuple = attr_data[1]
            for j in range(n):
                for k in range(j + 1, n):
                    if var_tuple[j] > var_tuple[k]:
                        temp_pos[j][k] = 1
                        temp_neg[k][j] = 1
                    else:
                        if var_tuple[j] < var_tuple[k]:
                            temp_neg[j][k] = 1
                            temp_pos[k][j] = 1
                        else:
                            if eq:
                                temp_neg[j][k] = 1
                                temp_pos[k][j] = 1
                                temp_pos[j][k] = 1
                                temp_neg[k][j] = 1
            lst_bin.append([incr, temp_pos])
            lst_bin.append([decr, temp_neg])
        return lst_bin

    @staticmethod
    def init_graph_rank(order, raw_attr):
        lst_tuple = []
        for i in range(len(raw_attr)):
            var_node = [i, raw_attr[i]]
            lst_tuple.append(var_node)
        if order == '+':
            ordered_t = sorted(lst_tuple, key=lambda x: x[1])
        elif order == '-':
            ordered_t = sorted(lst_tuple, key=lambda x: x[1], reverse=True)
        # print(ordered_t)
        G = nx.DiGraph()
        for i in range(len(ordered_t)):
            # generate Graph
            try:
                node = TupleNode(ordered_t[i][0], ordered_t[i][1])
                nxt_node = TupleNode(ordered_t[i + 1][0], ordered_t[i + 1][1])
                while node.value == nxt_node.value:
                    i += 1
                    nxt_node = TupleNode(ordered_t[i + 1][0], ordered_t[i + 1][1])
                # str_print = [node.index, nxt_node.index]
                # print(str_print)
                G.add_edge(node.index, nxt_node.index)
            except IndexError as e:
                break
        support = len(nx.dag_longest_path(G)) / len(raw_attr)
        return support, G

    @staticmethod
    def read_csv(file):
        # 1. retrieve data-set from file
        with open(file, 'r') as f:
            dialect = csv.Sniffer().sniff(f.read(1024), delimiters=";,' '\t")
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
