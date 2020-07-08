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
# lcm = LCM(min_supp=0.5)
# p = lcm.fit_discover(D, return_tids=True)
f_path = '../data/DATASET.csv'
lcm = LCM_g(f_path, min_supp=0.5)
p = lcm.fit_discover(return_tids=False)
d_set = lcm.d_set
wr_line = "Algorithm: LCM-GRAD (1.0)\n"
wr_line += "No. of (dataset) attributes: " + str(d_set.column_size) + '\n'
wr_line += "No. of (dataset) tuples: " + str(d_set.size) + '\n'
wr_line += "Minimum support: " + str(lcm.min_supp) + '\n'
# wr_line += "Number of cores: " + str(num_cores) + '\n'
# wr_line += "Number of patterns: " + str(len(list_gp)) + '\n\n'

for txt in d_set.title:
    try:
        wr_line += (str(txt.key) + '. ' + str(txt.value.decode()) + '\n')
    except AttributeError:
        wr_line += (str(txt[0]) + '. ' + str(txt[1].decode()) + '\n')

wr_line += str("\nFile: " + f_path + '\n')
wr_line += str(p)

end = time.time()
wr_text = ("Run-time: " + str(end - start) + " seconds\n")
wr_text += str(wr_line)
print(wr_text)
