[![Build Status](https://travis-ci.org/owuordickson/ant-colony-gp.svg?branch=update)](https://travis-ci.org/owuordickson/ant-colony-gp)

## ACO-GRAANK
A Python implementation of the <em><strong>ACO</strong>-GRAANK</em> algorithm. The algorithm utilizes a pheromone-based (or probabilistic) strategy to optimize the <em>GRAANK</em> algorithm. The algorithm converges as the pheromone matrix values approach saturation. We have optimized the implementation of the algorithm by: (1) using Numpy functions for operations that are time consuming, (2) allowed parallel multi-processing through <em><strong>mpi4py</strong></em> and (3) allowed secondary storage of large data in order to free CPU memory during processing through <em><strong>Parallel HDF5 and Parallel h5py</strong></em>.
<!-- Research paper published at -- link<br> -->

### Requirements:
You will be required to install the following python dependencies before using <em><strong>ACO</strong>-GRAANK</em> algorithm:<br>
```
                   install python (version => 3.6)

```

```
                    $ pip3 install numpy python-dateutil scikit-fuzzy cython h5py mpi4py

```

### Usage:
Use it a command line program with the local package:<br>
To mine gradual patterns:<br>

* Python and HDF5

```
$python3 src/init_acograd.py -f filename.csv -s minSup
```

To mine fuzzy-temporal gradual patterns:<br>

* Python and HDF5

```
$python3 src/init_acotgraank.py -f filename.csv -c refCol -s minSup  -r minRep
```

* Parallel (Python) MPI and Parallel HDF5

```
$mpirun -n nProcs python init_acotgrad_mpi.py -f filename.csv -c refCol -s minSup -r minRep
```

where you specify the input parameters as follows:<br>
* <strong>nProcs</strong> - [required] number of processes (only for MPI) <br>
* <strong>filename.csv</strong> - [required] a file in csv format <br>
* <strong>minSup</strong> - [optional] minimum support ```default = 0.5``` <br>
* <strong>minRep</strong> - [optional] minimum representativity ```default = 0.5``` <br>
* <strong>refCol</strong> - [optional] reference column ```default = 1``` <br>


For example we executed the <em><strong>ACO</strong>-GRAANK</em> algorithm on a sample data-set<br>
```
$python3 init_acograd.py -f ../data/DATASET.csv
```

<strong>Output</strong><br>
```
1. Age
2. Salary
3. Cars
4. Expenses

File: ../data/DATASET.csv

Pattern : Support
[('2', '+'), ('4', '-')] : 0.6
[('1', '-'), ('2', '-')] : 0.6
[('1', '-'), ('4', '+')] : 1.0
[('1', '+'), ('2', '+'), ('4', '-')] : 0.6
[('1', '+'), ('4', '-')] : 1.0
[('2', '-'), ('4', '+')] : 0.6
[('1', '+'), ('2', '+')] : 0.6
[('1', '-'), ('2', '-'), ('4', '+')] : 0.6

Pheromone Matrix
[[4 4 3]
 [4 4 3]
 [1 1 9]
 [4 4 3]]
0.08473014831542969 seconds
```

### License:
* MIT

### References
* Dickson Owuor, Anne Laurent, and Joseph Orero (2019). Mining Fuzzy-temporal Gradual Patterns. In the proceedings of the 2019 IEEE International Conference on Fuzzy Systems (FuzzIEEE). IEEE. doi:10.1109/FUZZ-IEEE.2019.8858883.
* Runkler, T. A. (2005), Ant colony optimization of clustering models. Int. J. Intell. Syst., 20: 1233-1251. doi:10.1002/int.20111
* Anne Laurent, Marie-Jeanne Lesot, and Maria Rifqi. 2009. GRAANK: Exploiting Rank Correlations for Extracting Gradual Itemsets. In Proceedings of the 8th International Conference on Flexible Query Answering Systems (FQAS '09). Springer-Verlag, Berlin, Heidelberg, 382-393.
