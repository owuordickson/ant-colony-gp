# -*- coding: utf-8 -*-
"""
@author: "Dickson Owuor"
@email: "owuordickson@gmail.com"
@created: "08 July 2020"

Usage:
    $python init_lcmgrad.py -f ../data/DATASET.csv -s 0.5

Description:
    f -> file path (CSV)
    s -> minimum support

"""

import sys
from optparse import OptionParser
from src.common.profile_mem import Profile
from lcm.lcm_grad import LCM_g


def init_algorithm(f_path, min_supp, cores):
    try:
        if cores > 1:
            num_cores = cores
        else:
            num_cores = Profile.get_num_cores()

        lcm = LCM_g(f_path, min_supp, n_jobs=num_cores)
        lst_gp = lcm.fit_discover()

        d_set = lcm.d_set
        wr_line = "Algorithm: LCM-GRAD (1.0) \n"
        wr_line += "No. of (dataset) attributes: " + str(d_set.column_size) + '\n'
        wr_line += "No. of (dataset) tuples: " + str(d_set.size) + '\n'
        wr_line += "Minimum support: " + str(d_set.thd_supp) + '\n'
        wr_line += "Number of cores: " + str(num_cores) + '\n'
        wr_line += "Number of patterns: " + str(len(lst_gp)) + '\n\n'

        for txt in d_set.title:
            wr_line += (str(txt[0]) + '. ' + str(txt[1].decode()) + '\n')

        wr_line += str("\nFile: " + f_path + '\n')
        wr_line += str("\nPattern : Support" + '\n')

        for obj in lst_gp:
            if len(obj) > 1:
                for gp in obj:
                    wr_line += (str(gp.to_string()) + ' : ' + str(gp.support) + '\n')
        # wr_line += str(gp)

        return wr_line
    except ArithmeticError as error:
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
        numCores = sys.argv[4]
    else:
        optparser = OptionParser()
        optparser.add_option('-f', '--inputFile',
                             dest='file',
                             help='path to file containing csv',
                             # default=None,
                             default='../../../data/DATASET.csv',
                             #default='../data/Omnidir.csv',
                             #default='../data/FARSmiss.csv',
                             #default='../data/FluTopicData-testsansdate-blank.csv',
                             type='string')
        optparser.add_option('-s', '--minSupport',
                             dest='minSup',
                             help='minimum support value',
                             default=0.5,
                             type='float')
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
        numCores = options.numCores

    import time
    # import tracemalloc
    # from src.algorithms.common.profile_mem import Profile

    start = time.time()
    # tracemalloc.start()
    res_text = init_algorithm(filePath, minSup, numCores)
    # snapshot = tracemalloc.take_snapshot()
    end = time.time()

    wr_text = ("Run-time: " + str(end - start) + " seconds\n")
    # wr_text += (Profile.get_quick_mem_use(snapshot) + "\n")
    wr_text += str(res_text)
    f_name = str('res_lcm' + str(end).replace('.', '', 1) + '.txt')
    write_file(wr_text, f_name)
    print(wr_text)


