# -*- coding: utf-8 -*-
import config as cfg
import time
import tracemalloc
from pkg_algorithms.pso_grad import execute
from pkg_algorithms.shared.profile import Profile

# res = init(cfg.DATASET, cfg.MIN_SUPPORT, cfg.CPU_CORES)
# print(res)

start = time.time()
tracemalloc.start()
res_text = execute(cfg.DATASET, cfg.MIN_SUPPORT, cfg.CPU_CORES, cfg.MAX_ITERATIONS, cfg.MAX_EVALUATIONS,
                   cfg.N_PARTICLES, cfg.VELOCITY, cfg.PERSONAL_COEFF, cfg.GLOBAL_COEFF, cfg.N_VAR)
snapshot = tracemalloc.take_snapshot()
end = time.time()

wr_text = ("Run-time: " + str(end - start) + " seconds\n")
wr_text += (Profile.get_quick_mem_use(snapshot) + "\n")
wr_text += str(res_text)
f_name = str('res_pso' + str(end).replace('.', '', 1) + '.txt')
Profile.write_file(wr_text, f_name)
print(wr_text)
