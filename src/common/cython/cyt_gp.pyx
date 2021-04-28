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
cimport numpy as np
import cython


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.profile(True)
cdef class GI:

    def __cinit__(self, int attr_col, str symbol):
        self.attribute_col = attr_col
        self.symbol = symbol
        self.gradual_item = tuple([attr_col, symbol])

    cpdef str to_string(self):
        cdef str res
        res = str(self.attribute_col) + self.symbol
        return res


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.profile(True)
cdef class GP:

    def __cinit__(self):
        self.gradual_items = list()
        self.support = 0

    cpdef void set_support(self, float supp):
        self.support = supp

    cpdef void add_gradual_item(self, GI item):
        self.gradual_items.append(item)

    cpdef list get_pattern(self):
        cdef list pattern
        pattern = list()
        for item in self.gradual_items:
            pattern.append(item.gradual_item)
        return pattern

    cpdef list to_string(self):
        cdef pattern
        pattern = list()
        for item in self.gradual_items:
            pattern.append(item.to_string())
        return pattern


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.profile(True)
cdef class TimeLag:

    def __cinit__(self, float tstamp=0, float supp=0):
        self.timestamp = tstamp
        self.support = supp
        self.sign = self.get_sign()
        if tstamp == 0:
           self.time_lag = np.array([])
        else:
            self.time_lag = np.array(self.format_time())

    cdef str get_sign(self):
        cdef str sign
        if self.timestamp < 0:
            sign = "-"
        else:
            sign = "+"
        return sign

    cpdef np.ndarray format_time(self):
        cdef float stamp_in_seconds, years, months, weeks, days, hours, minutes
        cdef np.ndarray res
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
                                res = np.array([stamp_in_seconds, "seconds"])
                            else:
                                res = np.array([minutes, "minutes"])
                        else:
                            res = np.array([hours, "hours"])
                    else:
                        res = np.array([days, "days"])
                else:
                    res = np.array([weeks, "weeks"])
            else:
                res = np.array([months, "months"])
        else:
            res = np.array([years, "years"])
        return res

    cpdef str to_string(self):
        cdef str res
        if len(self.time_lag) > 0:
            res = ("~ " + self.sign + str(self.time_lag[0]) + " " + str(self.time_lag[1])
                   + " : " + str(self.support))
        else:
            res = "No time lag found!"
        return res


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.profile(True)
cdef class TGP(GP):

    def __cinit__(self, GP gp=GP(), TimeLag t_lag=TimeLag()):
        super().__init__()
        self.gradual_items = gp.gradual_items
        self.support = gp.support
        self.time_lag = t_lag

    cpdef void set_timeLag(self, TimeLag t_lag):
        self.time_lag = t_lag
