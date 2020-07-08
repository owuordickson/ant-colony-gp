from algorithms.fp_growth.lcm.lcm import LCM
from algorithms.common.lcm_grad import LCM_g
import time
import numpy as np

D = [[1, 2, 3, 4, 5, 6], [2, 3, 5], [2, 5]]
Dg = [[0, 1, 0, -1, 3], [0, 2, 0, 1, 3], [0, 3, 0, -1, 3], [0, 4, 0, 1, 3],
      [1, 2, 0, 1, -3], [1, 3, 0, -1, -3], [1, 4, 0, 1, -3], [2, 3, 0, -1, -3],
      [2, 4, 0, 1, -3], [3, 4, 0, 1, -3]]
# print(np.array(Dg))

start = time.time()
#lcm = LCM(min_supp=0.5)
#p = lcm.fit_discover(D, return_tids=True)
lcm = LCM_g(min_supp=0.5)
p = lcm.fit_discover(np.array(Dg), return_tids=True)
end = time.time()
print("Run-time: " + str(end - start) + " seconds\n")
print(p)

