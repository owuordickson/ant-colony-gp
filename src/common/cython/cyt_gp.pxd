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


cdef class GI:

    cdef public int attribute_col
    cdef public str symbol
    cdef public tuple gradual_item
    cdef dict __dict__

    cpdef str to_string(self)


cdef class GP:

    cdef public list gradual_items
    cdef public float support
    cdef dict __dict__

    cpdef void set_support(self, float supp)
    cpdef void add_gradual_item(self, GI item)
    cpdef list get_pattern(self)
    cpdef list to_string(self)


cdef class TimeLag:

    cdef public float timestamp
    cdef public float support
    cdef public str sign
    cdef public np.ndarray timelag
    cdef dict __dict__

    cdef str get_sign(self)
    cpdef np.ndarray format_time(self)
    cpdef str to_string(self)


cdef class TGP(GP):

    cdef public TimeLag time_lag
    cpdef void set_timeLag(self, TimeLag t_lag)
