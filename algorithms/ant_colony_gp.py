# -*- coding: utf-8 -*-

"""
@author: "Dickson Owuor"
@credits: "Anne Laurent, Joseph Orero"
@license: "MIT"
@version: "1.0"
@email: "owuordickson@gmail.com"

Usage:
    $python t_graank.py -t 1 -f DATASET.csv -s 0.5
    $python t_graank.py -t 2 -f DATASET.csv -c 0 -s 0.5 -r 0.5

Description:
    t -> pattern type: GP, temporalGP, emergingGP
    f -> file path (CSV)
    c -> reference column
    s -> minimum support
    r -> minimum representativity

"""

import sys
from optparse import OptionParser

# --------------------- EXECUTE Ant-Colony GP -------------------------------------------


def algorithm_init(file_path, min_sup):
    try:
        print("first run")
    except Exception as error:
        print(error)


# ------------------------- main method -------------------------------------------------


if __name__ == "__main__":
    if not sys.argv:
        pattern_type = sys.argv[1]
        file_name = sys.argv[2]
        ref_col = sys.argv[3]
        min_sup = sys.argv[4]
        min_rep = sys.argv[5]

    else:
        optparser = OptionParser()
        optparser.add_option('-t', '--patternType',
                             dest='pType',
                             help='patterns: FtGP, FtGEP',
                             default=1,
                             type='int')
        optparser.add_option('-f', '--inputFile',
                             dest='file',
                             help='path to file containing csv',
                             default=None,
                             type='string')
        optparser.add_option('-c', '--refColumn',
                             dest='refCol',
                             help='reference column',
                             default=0,
                             type='int')
        optparser.add_option('-s', '--minSupport',
                             dest='minSup',
                             help='minimum support value',
                             default=0.5,
                             type='float')
        optparser.add_option('-r', '--minRepresentativity',
                             dest='minRep',
                             help='minimum representativity',
                             default=0.5,
                             type='float')
        (options, args) = optparser.parse_args()

        inFile = None
        if options.file is None:
            inFile = '../data/DATASET.csv'
            #print("Usage: $python t_graank.py -f filename.csv -c refColumn -s minSup
            # -r minRep")
            #sys.exit('System will exit')
        else:
            inFile = options.file
        file_path = inFile
        pattern_type = options.pType
        ref_col = options.refCol
        min_sup = options.minSup
        min_rep = options.minRep
    #import timeit
    if pattern_type == 1:
        algorithm_init(file_path, min_sup)
