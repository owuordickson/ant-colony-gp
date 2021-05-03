# -*- coding: utf-8 -*-

# Variables for Gradual Patterns:
MIN_SUPPORT = 0.5
CPU_CORES = 4
DATASET = "../../data/DATASET.csv"
ALGORITHM = 'aco'
# Uncomment for Terminal:
DATASET = "data/DATASET.csv"


def write_file(data, path):
    # return None
    with open(path, 'w') as f:
        f.write(data)
        f.close()
