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

    def to_string(self):
        return str(self.attribute_col) + self.symbol


class GP:

    def __init__(self):
        self.gradual_items = list()
        self.support = 0

    def set_support(self, support):
        self.support = round(support, 3)

    def add_gradual_item(self, item):
        self.gradual_items.append(item)

    def get_pattern(self):
        pattern = list()
        for item in self.gradual_items:
            pattern.append(item.gradual_item)
        return pattern

    def to_string(self):
        pattern = list()
        for item in self.gradual_items:
            pattern.append(item.to_string())
        return pattern


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

    def set_timeLag(self, t_lag):
        self.time_lag = t_lag
