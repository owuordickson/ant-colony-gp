import random
import h5py
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

data = [random.randint(1, 100) for x in range(4)]

f = h5py.File('parallel_test.hdf5', 'w', driver='mpio', comm=comm)
grp = f.require_group('d_set')
grp.require_group('sub')

for col in range(2):
    for i in range(size):
        arr = np.arange((rank + 2))
        ds = 'test{0}'.format(i) + '/another/' + str(col)
        grp['sub'].create_dataset(ds, data=arr)

if rank == 3:
    print(f['d_set/sub/0/test3/another/1'][:])
f.close()
