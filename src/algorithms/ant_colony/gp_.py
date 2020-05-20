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


class GP:

    def __init__(self, attr_col, symbol, support):
        self.attribute_col = attr_col
        self.symbol = symbol
        self.support = support
        self.pattern = tuple([attr_col, symbol])

    def print_pattern(self):
        return str(self.attribute_col) + self.symbol
