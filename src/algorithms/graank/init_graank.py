# -*- coding: utf-8 -*-
"""
@author: "Dickson Owuor"
@email: "owuordickson@gmail.com"
@created: "06 December 2019"

Usage:
    $python init_graank.py -f ../data/DATASET.csv -s 0.5

Description:
    f -> file path (CSV)
    s -> minimum support

"""

import sys
from optparse import OptionParser
from src.algorithms.common.profile_cpu import Profile
from src.algorithms.graank.graank_v2 import graank


def init_algorithm(f_path, min_supp, cores, eq=False):
    try:
        # wr_line = ""
        # d_set = Dataset(f_path)
        # if d_set.data:
        #    titles = d_set.title
        #    d_set.init_attributes(min_supp, eq)
        d_set, list_gp = graank(f_path, min_supp, eq)

        if cores > 1:
            num_cores = cores
        else:
            num_cores = Profile.get_num_cores()

        wr_line = "Algorithm: GRAANK \n"
        wr_line += "No. of (dataset) attributes: " + str(d_set.column_size) + '\n'
        wr_line += "No. of (dataset) tuples: " + str(d_set.size) + '\n'
        wr_line += "Minimum support: " + str(min_supp) + '\n'
        wr_line += "Number of cores: " + str(num_cores) + '\n\n'

        for txt in d_set.title:
            wr_line += (str(txt[0]) + '. ' + str(txt[1]) + '\n')

        wr_line += str("\nFile: " + f_path + '\n')
        wr_line += str("\nPattern : Support" + '\n')

        for gp in list_gp:
            wr_line += (str(gp.to_string()) + ' : ' + str(gp.support) + '\n')

        d_set.clean_memory()
        return wr_line
    except Exception as error:
        wr_line = "Failed: " + str(error)
        print(error)
        return wr_line


def write_file(data, path):
    with open(path, 'w') as f:
        f.write(data)
        f.close()

# ------------------------- main method ---------------------------------------------


if __name__ == "__main__":
    if not sys.argv:
        pType = sys.argv[1]
        filePath = sys.argv[2]
        minSup = sys.argv[3]
    else:
        optparser = OptionParser()
        optparser.add_option('-f', '--inputFile',
                             dest='file',
                             help='path to file containing csv',
                             # default=None,
                             default='../data/DATASET.csv',
                             #default='../data/Omnidir.csv',
                             #default='../data/FARSmiss.csv',
                             #default='../data/FluTopicData-testsansdate-blank.csv',
                             type='string')
        optparser.add_option('-s', '--minSupport',
                             dest='minSup',
                             help='minimum support value',
                             default=0.5,
                             type='float')
        optparser.add_option('-e', '--allowEqual',
                             dest='allowEq',
                             help='allow equal',
                             default=None,
                             type='int')
        optparser.add_option('-c', '--cores',
                             dest='numCores',
                             help='number of cores',
                             default=1,
                             type='int')
        (options, args) = optparser.parse_args()

        if options.file is None:
            print("Usage: $python3 init_graank.py -f filename.csv ")
            sys.exit('System will exit')
        else:
            filePath = options.file
        minSup = options.minSup
        allowEq = options.allowEq
        numCores = options.numCores

    import time
    # import tracemalloc

    start = time.time()
    # tracemalloc.start()
    res_text = init_algorithm(filePath, minSup, numCores)
    # snapshot = tracemalloc.take_snapshot()
    end = time.time()

    wr_text = ("Run-time: " + str(end - start) + " seconds\n")
    # wr_text += (Profile.get_quick_mem_use(snapshot) + "\n")
    wr_text += str(res_text)
    f_name = str('res_graank' + str(end).replace('.', '', 1) + '.txt')
    write_file(wr_text, f_name)
    print(wr_text)

