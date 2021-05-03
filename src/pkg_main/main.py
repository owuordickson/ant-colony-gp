# -*- coding: utf-8 -*-
"""
@author: "Dickson Owuor"
@created: "03 May 2021"


Usage:
    $python init_acograd.py -f ../data/DATASET.csv -s 0.5

Description:
    f -> file path (CSV)
    s -> minimum support

"""

import sys
from optparse import OptionParser
import config as cfg
from pkg_algorithms.aco_grad import init


if __name__ == "__main__":
    if not sys.argv:
        filePath = sys.argv[1]
        minSup = sys.argv[2]
        numCores = sys.argv[3]
    else:
        optparser = OptionParser()
        optparser.add_option('-f', '--inputFile',
                             dest='file',
                             help='path to file containing csv',
                             # default=None,
                             default=cfg.DATASET,
                             type='string')
        optparser.add_option('-s', '--minSupport',
                             dest='minSup',
                             help='minimum support value',
                             default=cfg.MIN_SUPPORT,
                             type='float')
        optparser.add_option('-c', '--cores',
                             dest='numCores',
                             help='number of cores',
                             default=cfg.CPU_CORES,
                             type='int')
        (options, args) = optparser.parse_args()

        if options.file is None:
            print("Usage: $python3 main.py -f filename.csv ")
            sys.exit('System will exit')
        else:
            filePath = options.file
        minSup = options.minSup
        numCores = options.numCores

    import time
    import tracemalloc
    from pkg_algorithms.shared.profile_mem import Profile

    start = time.time()
    tracemalloc.start()
    res_text = init(filePath, minSup, numCores)
    snapshot = tracemalloc.take_snapshot()
    end = time.time()

    wr_text = ("Run-time: " + str(end - start) + " seconds\n")
    wr_text += (Profile.get_quick_mem_use(snapshot) + "\n")
    wr_text += str(res_text)
    f_name = str('res_aco' + str(end).replace('.', '', 1) + '.txt')
    # write_file(wr_text, f_name)
    print(wr_text)
