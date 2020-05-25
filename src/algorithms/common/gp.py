# -*- coding: utf-8 -*-
"""
@author: "Dickson OWUOR"
@credits: "Anne LAURENT, Joseph ORERO"
@version: "3.0"
@email: "owuordickson@gmail.com"
@created: "20 May 2020"

GP: Gradual Pattern
TGP: Temporal Gradual Pattern

"""
import numpy as np


class GI:

    def __init__(self, attr_col, symbol):
        self.attribute_col = attr_col
        self.symbol = symbol
        self.gradual_item = tuple([attr_col, symbol])

    def toStr(self):
        return str(self.attribute_col) + self.symbol


class GP:

    def __init__(self):
        self.gradual_items = list()
        self.support = 0

    def set_support(self, support):
        self.support = support

    def add_gradual_item(self, item):
        self.gradual_items.append(item)

    def get_pattern(self):
        pattern = list()
        for item in self.gradual_items:
            pattern.append(item.gradual_item)
        return pattern

    def print_pattern(self):
        pattern = list()
        for item in self.gradual_items:
            pattern.append(item.toStr())
        return pattern


class TimeLag:

    def __init__(self, tstamp=0, supp=0):
        self.timestamp = tstamp
        self.support = supp
        self.sign = self.get_sign()
        if tstamp == 0:
           self.timelag = np.array([])
        else:
            self.timelag = np.array(self.format_time())

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
                                return [stamp_in_seconds, "seconds"]
                            else:
                                return [minutes, "minutes"]
                        else:
                            return [hours, "hours"]
                    else:
                        return [days, "days"]
                else:
                    return [weeks, "weeks"]
            else:
                return [months, "months"]
        else:
            return [years, "years"]

    def print_lag(self):
        txt = ("~ " + self.sign + str(self.timelag[0]) + " " + str(self.timelag[1])
               + " : " + str(self.support))
        return txt


class TGP(GP):

    def __init__(self, gp=GP(), t_lag=TimeLag()):
        super().__init__()
        self.gradual_items = gp.gradual_items
        self.support = gp.support
        self.time_lag = t_lag

    def set_timeLag(self, t_lag):
        self.time_lag = t_lag

    def set_gradualPattern(self, gp):
        self.gradual_pattern = gp
