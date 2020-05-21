# -*- coding: utf-8 -*-
"""
@author: "Dickson OWUOR"
@credits: "Anne LAURENT, Joseph ORERO"
@version: "2.0"
@email: "owuordickson@gmail.com"
@created: "26 March 2020"

GP: Gradual Pattern
TGP: Temporal Gradual Pattern

"""


class GP:

    def __init__(self, gp):
        self.gp = gp
        self.pattern = None
        self.support = 0
        self.format_pattern()

    def format_pattern(self):
        self.pattern = GP.format_gp(self.gp[1])
        self.support = self.gp[0]

    @staticmethod
    def format_gp(obj_gp):
        arr_gp = list(obj_gp)
        new_gp = list()
        for item in arr_gp:
            attr = int(item[0])
            sign = item[1]
            str_gp = str(attr) + sign
            new_gp.append(str_gp)
        return set(new_gp)


class TGP:

    def __init__(self, tgp):
        self.tgp = tgp
        self.pattern = None
        self.support = 0
        self.time_lag = None
        self.format_pattern()

    def format_pattern(self):
        self.pattern = GP.format_gp(self.tgp[1][0])
        self.support = self.tgp[0]
        self.time_lag = self.tgp[1][1]

