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
