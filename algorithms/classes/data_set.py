# -*- coding: utf-8 -*-

"""
@author: "Dickson Owuor"
@credits: "Anne Laurent, Joseph Orero"
@license: "MIT"
@version: "1.0"
@email: "owuordickson@gmail.com"

"""
import csv
from dateutil.parser import parse
import time


class DataSet:

    def __init__(self, file_path):
        self.raw_data = DataSet.read_csv(file_path)
        if len(self.raw_data) == 0:
            self.data = False
            print("Data-set error")
            raise Exception("Unable to read csv file")
        else:
            self.data = self.raw_data
            self.title = self.get_title()
            self.column_size = self.get_attribute_no()
            self.time_columns = self.get_time_cols()
            self.attributes = []

    def get_attribute_no(self):
        length = len(self.raw_data)
        return length

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
                    sub = (str(i + 1) + ' : ' + data[0][i])
                    title.append(sub)
                del self.data[0]
                return title

    def get_time_cols(self):
        time_cols = list()
        for i in range(len(self.data[0])):  # check every column for time format
            row_data = str(self.data[0][i])
            try:
                time_ok, t_stamp = DataSet.test_time(row_data)
                if time_ok:
                    time_cols.append(i)
            except ValueError:
                continue
        if time_cols:
            return time_cols
        else:
            return False

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
