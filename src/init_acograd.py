# -*- coding: utf-8 -*-
"""
@author: "Dickson Owuor"
@credits: "Thomas Runkler, Edmond Menya, and Anne Laurent,"
@license: "MIT"
@version: "2.0"
@email: "owuordickson@gmail.com"
@created: "06 July 2020"

Breath-First Search for gradual patterns (ACO-GRAANK)

Usage:
    $python init_acograd.py -f ../data/DATASET.csv -s 0.5

Description:
    f -> file path (CSV)
    s -> minimum support

"""

import sys
from optparse import OptionParser
from algorithms.common.profile_mem import Profile
from algorithms.ant_colony.aco_grad import GradACO


def init_algorithm(f_path, min_supp, cores, eq=False):
    try:
        if cores > 1:
            num_cores = cores
        else:
            num_cores = Profile.get_num_cores()

        ac = GradACO(f_path, min_supp, eq)
        list_gp = ac.run_ant_colony()

        d_set = ac.d_set
        wr_line = "Algorithm: ACO-GRAANK (2.0)\n"
        wr_line += "No. of (dataset) attributes: " + str(d_set.column_size) + '\n'
        wr_line += "No. of (dataset) tuples: " + str(d_set.size) + '\n'
        wr_line += "Minimum support: " + str(min_supp) + '\n'
        wr_line += "Number of cores: " + str(num_cores) + '\n'
        wr_line += "Number of patterns: " + str(len(list_gp)) + '\n\n'

        for txt in d_set.title:
            try:
                wr_line += (str(txt.key) + '. ' + str(txt.value.decode()) + '\n')
            except AttributeError:
                wr_line += (str(txt[0]) + '. ' + str(txt[1].decode()) + '\n')

        wr_line += str("\nFile: " + f_path + '\n')
        wr_line += str("\nPattern : Support" + '\n')

        for gp in list_gp:
            wr_line += (str(gp.to_string()) + ' : ' + str(gp.support) + '\n')

        wr_line += "\nPheromone Matrix\n"
        wr_line += str(ac.p_matrix)
        # ac.plot_pheromone_matrix()
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
        filePath = sys.argv[1]
        minSup = sys.argv[2]
        allowEq = sys.argv[3]
        numCores = sys.argv[4]
    else:
        optparser = OptionParser()
        optparser.add_option('-f', '--inputFile',
                             dest='file',
                             help='path to file containing csv',
                             # default=None,
                             default='../data/DATASET.csv',
                             #default='../data/DATASET3.csv',
                             #default='../data/Omnidir.csv',
                             #default='../data/FluTopicData-testsansdate-blank.csv',
                             #default='../data/vehicle_silhouette_dataset.csv',
                             #default='../data/FARSmiss.csv',
                             #default='../data/c2k_02k.csv',
                             type='string')
        optparser.add_option('-s', '--minSupport',
                             dest='minSup',
                             help='minimum support value',
                             default=0.9,
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
            print("Usage: $python init_acograd.py -f filename.csv ")
            sys.exit('System will exit')
        else:
            filePath = options.file
        minSup = options.minSup
        allowEq = options.allowEq
        numCores = options.numCores

    import time
    # import tracemalloc
    # from algorithms.common.profile_mem import Profile

    start = time.time()
    # tracemalloc.start()
    res_text = init_algorithm(filePath, minSup, numCores, allowEq)
    # snapshot = tracemalloc.take_snapshot()
    end = time.time()

    wr_text = ("Run-time: " + str(end - start) + " seconds\n")
    # wr_text += (Profile.get_quick_mem_use(snapshot) + "\n")
    wr_text += str(res_text)
    f_name = str('res_aco' + str(end).replace('.', '', 1) + '.txt')
    write_file(wr_text, f_name)
    print(wr_text)

