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
from src.algorithms.ant_colony.fuzzy_mf_v2 import TimeLag


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
