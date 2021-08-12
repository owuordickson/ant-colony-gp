# -*- coding: utf-8 -*-
"""
@author: "Dickson OWUOR"
@credits: "Anne LAURENT and Joseph ORERO"
@version: "4.8"
@email: "owuordickson@gmail.com"
@created: "20 May 2020"
@modified: "10 Mar 2021"

GI: Gradual Item (0, +)
GP: Gradual Pattern {(0, +), (1, -), (3, +)}
TGP: Temporal Gradual Pattern

"""
import numpy as np


class GI:

    def __init__(self, attr_col, symbol):
        self.attribute_col = attr_col
        self.symbol = symbol
        self.gradual_item = np.array((attr_col, symbol), dtype='i, S1')
        self.tuple = tuple([attr_col, symbol])
        self.rank_sum = 0

    def inv(self):
        if self.symbol == '+':
            # temp = tuple([self.attribute_col, '-'])
            temp = np.array((self.attribute_col, '-'), dtype='i, S1')
        elif self.symbol == '-':
            # temp = tuple([self.attribute_col, '+'])
            temp = np.array((self.attribute_col, '+'), dtype='i, S1')
        else:
            temp = np.array((self.attribute_col, 'x'), dtype='i, S1')
        return temp

    def as_integer(self):
        if self.symbol == '+':
            temp = [self.attribute_col, 1]
        elif self.symbol == '-':
            temp = [self.attribute_col, -1]
        else:
            temp = [self.attribute_col, 0]
        return temp

    def as_string(self):
        if self.symbol == '+':
            temp = str(self.attribute_col) + '_pos'
        elif self.symbol == '-':
            temp = str(self.attribute_col) + '_neg'
        else:
            temp = str(self.attribute_col) + '_inv'
        return temp

    def to_string(self):
        return str(self.attribute_col) + self.symbol

    def is_decrement(self):
        if self.symbol == '-':
            return True
        else:
            return False

    @staticmethod
    def parse_gi(gi_str):
        txt = gi_str.split('_')
        attr_col = int(txt[0])
        if txt[1] == 'neg':
            symbol = '-'
        else:
            symbol = '+'
        return GI(attr_col, symbol)


class GP:

    def __init__(self):
        self.gradual_items = list()
        self.support = 0

    def set_support(self, support):
        self.support = round(support, 3)

    def add_gradual_item(self, item):
        if item.symbol == '-' or item.symbol == '+':
            self.gradual_items.append(item)
        else:
            pass

    def get_pattern(self):
        pattern = list()
        for item in self.gradual_items:
            pattern.append(item.gradual_item.tolist())
        return pattern

    def get_np_pattern(self):
        pattern = []
        for item in self.gradual_items:
            pattern.append(item.gradual_item)
        return np.array(pattern)

    def get_tuples(self):
        pattern = list()
        for gi in self.gradual_items:
            temp = tuple([gi.attribute_col, gi.symbol])
            pattern.append(temp)
        return pattern

    def get_attributes(self):
        attrs = list()
        syms = list()
        for item in self.gradual_items:
            gi = item.as_integer()
            attrs.append(gi[0])
            syms.append(gi[1])
        return attrs, syms

    def get_index(self, gi):
        for i in range(len(self.gradual_items)):
            gi_obj = self.gradual_items[i]
            if (gi.symbol == gi_obj.symbol) and (gi.attribute_col == gi_obj.attribute_col):
                return i
        return -1

    def inv_pattern(self):
        pattern = list()
        for gi in self.gradual_items:
            pattern.append(gi.inv().tolist())
        return pattern

    def contains(self, gi):
        if gi is None:
            return False
        if gi in self.gradual_items:
            return True
        return False

    def contains_attr(self, gi):
        if gi is None:
            return False
        for gi_obj in self.gradual_items:
            if gi.attribute_col == gi_obj.attribute_col:
                return True
        return False

    def to_string(self):
        pattern = list()
        for item in self.gradual_items:
            pattern.append(item.to_string())
        return pattern

    def to_dict(self):
        gi_dict = {}
        for gi in self.gradual_items:
            gi_dict.update({gi.as_string(): 0})
        return gi_dict


class TimeLag:

    def __init__(self, tstamp=0, supp=0):
        self.timestamp = tstamp
        self.support = round(supp, 3)
        self.sign = self.get_sign()
        if tstamp == 0:
            self.time_lag = np.array([])
            self.valid = False
        else:
            self.time_lag = np.array(self.format_time())
            self.valid = True

    def get_sign(self):
        if self.timestamp < 0:
            sign = "-"
        else:
            sign = "+"
        return sign

    def format_time(self):
        stamp_in_seconds = abs(self.timestamp)
        years = stamp_in_seconds / 3.154e+7
        months = stamp_in_seconds / 2.628e+6
        weeks = stamp_in_seconds / 604800
        days = stamp_in_seconds / 86400
        hours = stamp_in_seconds / 3600
        minutes = stamp_in_seconds / 60
        if int(years) <= 0:
            if int(months) <= 0:
                if int(weeks) <= 0:
                    if int(days) <= 0:
                        if int(hours) <= 0:
                            if int(minutes) <= 0:
                                return [round(stamp_in_seconds, 0), "seconds"]
                            else:
                                return [round(minutes, 0), "minutes"]
                        else:
                            return [round(hours, 0), "hours"]
                    else:
                        return [round(days, 0), "days"]
                else:
                    return [round(weeks, 0), "weeks"]
            else:
                return [round(months, 0), "months"]
        else:
            return [round(years, 0), "years"]

    def to_string(self):
        if len(self.time_lag) > 0:
            txt = ("~ " + self.sign + str(self.time_lag[0]) + " " + str(self.time_lag[1])
                   + " : " + str(self.support))
        else:
            txt = "No time lag found!"
        return txt


class TGP(GP):

    def __init__(self, gp=GP(), t_lag=TimeLag()):
        super().__init__()
        self.gradual_items = gp.gradual_items
        self.support = gp.support
        self.time_lag = t_lag

    def set_timelag(self, t_lag):
        self.time_lag = t_lag
