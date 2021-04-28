# -*- coding: utf-8 -*-
"""
@author: "Dickson Owuor"
@credits: "Anne Laurent"
@license: "MIT"
@version: "2.2"
@email: "owuordickson@gmail.com"
@created: "12 July 2019"
@modified: "25 May 2020"

Changes
-------
1. save attribute gradual item sets binaries as json file and retrieve them as dicts
   - this frees primary memory from storing nxn matrices
2. Fetch all binaries during initialization
3. Replaced loops for fetching binary rank with numpy function

"""

import numpy as np
cimport numpy as np


cdef class Dataset:

    cdef dict __dict__
    cdef public int size, column_size, attr_size
    cdef float thd_supp
    cdef int equal
    cdef public np.ndarray time_cols
    cdef public np.ndarray attr_cols
    cdef public np.ndarray title
    cdef public np.ndarray data
    cdef public np.ndarray valid_gi_paths
    cdef public np.ndarray invalid_bins

    # Dataset() nogil except +
    cdef int get_size(self)
    cdef int get_attribute_no(self)
    cdef np.ndarray get_attributes(self)
    cdef np.ndarray get_title(self, list data)
    cdef np.ndarray convert_data_to_array(self, list data, bint has_title)
    cdef np.ndarray get_time_cols(self)
    cpdef dict get_bin(self, str gi_path)
    cpdef void clean_memory(self)
    cpdef void init_attributes(self, float min_sup, bint eq)
    cdef void construct_bins(self, np.ndarray attr_data)
    cdef np.ndarray bin_rank(self, np.ndarray arr)
