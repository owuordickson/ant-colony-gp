[![Build Status](https://travis-ci.org/owuordickson/ant-colony-gp.svg?branch=master)](https://travis-ci.org/owuordickson/ant-colony-gp)
## ACO-GRAANK
A Python implementation of the <em><strong>ACO</strong>-GRAANK</em> algorithm. The algorithm utilizes a pheromone-based (or probabilistic) strategy to optimize the <em>GRAANK</em> algorithm. The algorithm converges as the pheromone matrix values approach saturation. The research paper is available via this link:

* Owuor, D.O., Runkler, T., Laurent, A. et al. Ant colony optimization for mining gradual patterns. Int. J. Mach. Learn. & Cyber. (2021). https://doi.org/10.1007/s13042-021-01390-w

### Requirements:
You will be required to install the following python dependencies before using <em><strong>ACO</strong>-GRAANK</em> algorithm:<br>
```
                   install python (version => 3.6)

```
<!-- python-dateutil scikit-fuzzy cython h5py mpi4py -->
```
                    $ pip3 install numpy pandas ypstruct~=0.0.2 sortedcontainers~=2.4.0 scikit-fuzzy~=0.4.0 python-dateutil~=2.8.2 matplotlib~=3.4.2

```

### Usage:
Use it a command line program with the local package to mine gradual patterns:

For example we executed the <em><strong>ACO</strong>-GRAANK</em> algorithm on a sample data-set<br>
```
$python3 src/main.py -a 'aco' -f data/DATASET.csv
```

where you specify the input parameters as follows:<br>
* <strong>filename.csv</strong> - [required] a file in csv format <br>
* <strong>minSup</strong> - [optional] minimum support ```default = 0.5``` <br>


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
* Dickson Owuor, Anne Laurent, and Joseph Orero (2019). Mining Fuzzy-temporal Gradual Patterns. In the proceedings of the 2019 IEEE International Conference on Fuzzy Systems (FuzzIEEE). IEEE. https://doi.org/10.1109/FUZZ-IEEE.2019.8858883.
* Runkler, T. A. (2005), Ant colony optimization of clustering models. Int. J. Intell. Syst., 20: 1233-1251. https://doi.org/10.1002/int.20111
* Anne Laurent, Marie-Jeanne Lesot, and Maria Rifqi. 2009. GRAANK: Exploiting Rank Correlations for Extracting Gradual Itemsets. In Proceedings of the 8th International Conference on Flexible Query Answering Systems (FQAS '09). Springer-Verlag, Berlin, Heidelberg, 382-393.
