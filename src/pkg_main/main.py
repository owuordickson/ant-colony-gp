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
from pkg_algorithms import aco_grad, ga_grad, pso_grad, graank_v2, aco_lcm, lcm_gp


if __name__ == "__main__":
    if not sys.argv:
        algChoice = sys.argv[1]
        filePath = sys.argv[2]
        minSup = sys.argv[3]
        numCores = sys.argv[4]
    else:
        optparser = OptionParser()
        optparser.add_option('-a', '--algorithmChoice',
                             dest='algChoice',
                             help='select algorithm',
                             default=cfg.ALGORITHM,
                             type='string')
        optparser.add_option('-f', '--inputFile',
                             dest='file',
                             help='path to file containing csv',
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
            print("Usage: $python3 main.py -a 'aco' -f filename.csv ")
            sys.exit('System will exit')
        else:
            filePath = options.file
        algChoice = options.algChoice
        minSup = options.minSup
        numCores = options.numCores

    import time
    import tracemalloc
    from pkg_algorithms.shared.profile import Profile

    if algChoice == 'aco':
        # ACO-GRAANK
        start = time.time()
        tracemalloc.start()
        res_text = aco_grad.init(filePath, minSup, numCores)
        snapshot = tracemalloc.take_snapshot()
        end = time.time()

        wr_text = ("Run-time: " + str(end - start) + " seconds\n")
        wr_text += (Profile.get_quick_mem_use(snapshot) + "\n")
        wr_text += str(res_text)
        f_name = str('res_aco' + str(end).replace('.', '', 1) + '.txt')
        Profile.write_file(wr_text, f_name)
        print(wr_text)
    elif algChoice == 'ga':
        # GA-GRAANK
        start = time.time()
        tracemalloc.start()
        res_text = ga_grad.init(filePath, minSup, numCores)
        snapshot = tracemalloc.take_snapshot()
        end = time.time()

        wr_text = ("Run-time: " + str(end - start) + " seconds\n")
        wr_text += (Profile.get_quick_mem_use(snapshot) + "\n")
        wr_text += str(res_text)
        f_name = str('res_ga' + str(end).replace('.', '', 1) + '.txt')
        Profile.write_file(wr_text, f_name)
        print(wr_text)
    elif algChoice == 'pso':
        # PSO-GRAANK
        start = time.time()
        tracemalloc.start()
        res_text = pso_grad.init(filePath, minSup, numCores)
        snapshot = tracemalloc.take_snapshot()
        end = time.time()

        wr_text = ("Run-time: " + str(end - start) + " seconds\n")
        wr_text += (Profile.get_quick_mem_use(snapshot) + "\n")
        wr_text += str(res_text)
        f_name = str('res_pso' + str(end).replace('.', '', 1) + '.txt')
        Profile.write_file(wr_text, f_name)
        print(wr_text)
    elif algChoice == 'graank':
        # GRAANK
        start = time.time()
        tracemalloc.start()
        res_text = graank_v2.init(filePath, minSup, numCores)
        snapshot = tracemalloc.take_snapshot()
        end = time.time()

        wr_text = ("Run-time: " + str(end - start) + " seconds\n")
        wr_text += (Profile.get_quick_mem_use(snapshot) + "\n")
        wr_text += str(res_text)
        f_name = str('res_graank' + str(end).replace('.', '', 1) + '.txt')
        Profile.write_file(wr_text, f_name)
        print(wr_text)
    elif algChoice == 'acolcm':
        # ACO-LCM
        start = time.time()
        tracemalloc.start()
        res_text = aco_lcm.init(filePath, minSup, numCores)
        snapshot = tracemalloc.take_snapshot()
        end = time.time()

        wr_text = ("Run-time: " + str(end - start) + " seconds\n")
        wr_text += (Profile.get_quick_mem_use(snapshot) + "\n")
        wr_text += str(res_text)
        f_name = str('res_acolcm' + str(end).replace('.', '', 1) + '.txt')
        Profile.write_file(wr_text, f_name)
        print(wr_text)
    elif algChoice == 'lcm':
        # LCM
        start = time.time()
        tracemalloc.start()
        res_text = lcm_gp.init(filePath, minSup, numCores)
        snapshot = tracemalloc.take_snapshot()
        end = time.time()

        wr_text = ("Run-time: " + str(end - start) + " seconds\n")
        wr_text += (Profile.get_quick_mem_use(snapshot) + "\n")
        wr_text += str(res_text)
        f_name = str('res_lcm' + str(end).replace('.', '', 1) + '.txt')
        Profile.write_file(wr_text, f_name)
        print(wr_text)
    else:
        print("Invalid Algorithm Choice!")
