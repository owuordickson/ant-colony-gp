from algorithms.fp_growth.lcm.lcm import LCM
import time

D = [[1, 2, 3, 4, 5, 6], [2, 3, 5], [2, 5]]

start = time.time()
lcm = LCM(min_supp=0.5)
p = lcm.fit_discover(D, return_tids=True)
end = time.time()
print("Run-time: " + str(end - start) + " seconds\n")
print(p)

