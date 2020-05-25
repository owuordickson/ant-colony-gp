# -*- coding: utf-8 -*-
"""
@author: "Dickson Owuor"
@credits: "Thomas Runkler, Edmond Menya, and Anne Laurent,"
@license: "MIT"
@version: "1.0"
@email: "owuordickson@gmail.com"
@created: "18 November 2019"

Usage:
    $python init_acograd.py -f ../data/DATASET.csv -s 0.5

Description:
    f -> file path (CSV)
    s -> minimum support

"""

import sys
from optparse import OptionParser
from src.algorithms.common.profile_cpu import Profile
#from src.algorithms.common.dataset import Dataset
from src.algorithms.common.cyt_dataset import Dataset
from src.algorithms.ant_colony.aco_grad import GradACO


def init_algorithm(f_path, min_supp, cores, eq=False):
    try:
        wr_line = ""
        d_set = Dataset(f_path)
        if d_set.data.size > 0:
            titles = d_set.title
            d_set.init_attributes(min_supp, eq)
            ac = GradACO(d_set)
            ac.init_pheromones()
            list_gp = ac.run_ant_colony(min_supp)

            if cores > 1:
                num_cores = cores
            else:
                num_cores = Profile.get_num_cores()

            wr_line = "Algorithm: ACO-GRAANK \n"
            wr_line += "No. of (dataset) attributes: " + str(d_set.column_size) + '\n'
            wr_line += "No. of (dataset) tuples: " + str(d_set.size) + '\n'
            wr_line += "Minimum support: " + str(min_supp) + '\n'
            wr_line += "Number of cores: " + str(num_cores) + '\n\n'

            for txt in titles:
                wr_line += (str(txt.key) + '. ' + str(txt.value) + '\n')

            wr_line += str("\nFile: " + f_path + '\n')
            wr_line += str("\nPattern : Support" + '\n')

            for gp in list_gp:
                wr_line += (str(gp.print_pattern()) + ' : ' + str(gp.support) + '\n')

            wr_line += "\nPheromone Matrix\n"
            wr_line += str(ac.p_matrix)
            # ac.plot_pheromone_matrix()
            d_set.clean_memory()
        return wr_line
    except ArithmeticError as error:
        wr_line = "Failed: " + str(error)
        print(error)
        return wr_line


# ------------------------- main method ---------------------------------------------


if __name__ == "__main__":
    if not sys.argv:
        pType = sys.argv[1]
        filePath = sys.argv[2]
        # refCol = sys.argv[3]
        minSup = sys.argv[4]
        # minRep = sys.argv[5]
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
                             #default='data/FluTopicData-testsansdate-blank.csv',
                             #default='../data/FARSmiss.csv',
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
            print("Usage: $python init_acograd.py -f filename.csv ")
            sys.exit('System will exit')
        else:
            filePath = options.file
        minSup = options.minSup
        allowEq = options.allowEq
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
    f_name = str('res_acograd' + str(end).replace('.', '', 1) + '.txt')
    # Dataset.write_file(wr_text, f_name)
    print(wr_text)

