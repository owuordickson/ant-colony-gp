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
from algorithms.init_data import InitData
from algorithms.aco_grad import GradACO

def init_algorithm(f_path, min_supp, eq=False):
    try:
        d_set = InitData(f_path)
        if d_set.data:
            for txt in d_set.title:
                print(str(txt[0]) + '. '+txt[1])
            print("\nFile: " + f_path)

            d_set.init_attributes(eq)
            ac = GradACO(d_set)
            list_gp = ac.run_ant_colony(min_supp)
            print("\nPattern : Support")
            for gp in list_gp:
                print(str(gp[1])+' : '+str(gp[0]))
            print("\nPheromone Matrix")
            print(ac.p_matrix)
            # ac.plot_pheromone_matrix()
    except Exception as error:
        print(error)


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
                             # default='../data/DATASET.csv',
                             default='../data/FARSmiss.csv',
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
        (options, args) = optparser.parse_args()

        if options.file is None:
            print("Usage: $python init_acograd.py -f filename.csv -s minSup ")
            sys.exit('System will exit')
        else:
            filePath = options.file
        minSup = options.minSup
        allowEq = options.allowEq
    import time
    start = time.time()
    init_algorithm(filePath, minSup)
    end = time.time()
    print("\n"+str(end-start)+" seconds")

