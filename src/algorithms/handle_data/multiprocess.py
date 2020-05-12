# -*- coding: utf-8 -*-
"""
@author: "Dickson Owuor"
@license: "MIT"
@version: "1.0"
@email: "owuordickson@gmail.com"
@created: "5 November 2019"
@modified: "12 May 2020"

"""


import os
import multiprocessing as mp
import tracemalloc
import linecache


class InitParallel:

    @staticmethod
    def get_num_cores():
        num_cores = InitParallel.get_slurm_cores()
        if not num_cores:
            num_cores = mp.cpu_count()
        return num_cores

    @staticmethod
    def get_slurm_cores():
        try:
            cores = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
            return cores
        except ValueError:
            try:
                str_cores = str(os.environ['SLURM_JOB_CPUS_PER_NODE'])
                temp = str_cores.split('(', 1)
                cpus = int(temp[0])
                str_nodes = temp[1]
                temp = str_nodes.split('x', 1)
                str_temp = str(temp[1]).split(')', 1)
                nodes = int(str_temp[0])
                cores = cpus * nodes
                return cores
            except ValueError:
                return False
        except KeyError:
            return False

    @staticmethod
    def get_quick_mem_use(snapshot, key_type='lineno'):
        snapshot = snapshot.filter_traces((
            tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
            tracemalloc.Filter(False, "<unknown>"),
        ))
        top_stats = snapshot.statistics(key_type)
        total = sum(stat.size for stat in top_stats)
        wr_line = ("Total allocated memory size: %.1f KiB" % (total / 1024))
        return wr_line

    @staticmethod
    def get_detailed_mem_use(snapshot, key_type='lineno', limit=10):
        wr_line = ""
        snapshot = snapshot.filter_traces((
            tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
            tracemalloc.Filter(False, "<unknown>"),
        ))
        top_stats = snapshot.statistics(key_type)

        wr_line += ("Top %s lines" % limit)
        for index, stat in enumerate(top_stats[:limit], 1):
            frame = stat.traceback[0]
            # replace "/path/to/module/file.py" with "module/file.py"
            filename = os.sep.join(frame.filename.split(os.sep)[-2:])
            wr_line += ("\n #%s: %s:%s: %.1f KiB" % (index, filename, frame.lineno, stat.size / 1024))
            line = linecache.getline(frame.filename, frame.lineno).strip()
            if line:
                wr_line += ('\n    %s' % line)

        other = top_stats[limit:]
        if other:
            size = sum(stat.size for stat in other)
            wr_line += ("\n %s other: %.1f KiB" % (len(other), size / 1024))
        total = sum(stat.size for stat in top_stats)
        wr_line += ("\n Total allocated size: %.1f KiB" % (total / 1024))
        return wr_line
