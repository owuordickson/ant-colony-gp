# -*- coding: utf-8 -*-
import config as cfg
from pkg_algorithms.aco_grad import init


res = init(cfg.DATASET, cfg.MIN_SUPPORT, cfg.CPU_CORES)
print(res)
