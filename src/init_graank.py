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
# from src import InitParallel, HandleData, graank
from algorithms.handle_data.multiprocess import InitParallel
from algorithms.handle_data.handle_data import HandleData
from algorithms.ant_colony.graank import graank


def init_algorithm(f_path, min_supp, cores, eq=False):
    try:
        wr_line = ""
        d_set = HandleData(f_path)
        if d_set.data:
            titles = d_set.title
            d_set.init_attributes(eq)
            D1, S1 = graank(d_set.attr_data, min_supp)

            if cores > 1:
                num_cores = cores
            else:
                num_cores = InitParallel.get_num_cores()

            wr_line = "Algorithm: GRAANK \n"
            wr_line += "No. of (dataset) attributes: " + str(d_set.column_size) + '\n'
            wr_line += "No. of (dataset) tuples: " + str(d_set.size) + '\n'
            wr_line += "Minimum support: " + str(min_supp) + '\n'
            wr_line += "Number of cores: " + str(num_cores) + '\n\n'

            for txt in titles:
                wr_line += (str(txt[0]) + '. ' + str(txt[1]) + '\n')

            wr_line += str("\nFile: " + f_path + '\n')
            wr_line += str("\nPattern : Support" + '\n')

            for i in range(len(D1)):
                wr_line += (str(D1[i]) + ' : ' + str(S1[i]) + '\n')

        return wr_line
    except Exception as error:
        wr_line = "Failed: " + str(error)
        print(error)
        return wr_line


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
                             # default='../data/DATASET.csv',
                             default='../data/Omnidir.csv',
                             # default='../data/FARSmiss.csv',
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

    start = time.time()
    res_text = init_algorithm(filePath, minSup, numCores)
    end = time.time()

    wr_text = ("Run-time: " + str(end - start) + " seconds\n")
    wr_text += str(res_text)
    f_name = str('res_graank' + str(end).replace('.', '', 1) + '.txt')
    HandleData.write_file(wr_text, f_name)
    print(wr_text)

