## ACO-GRAANK
A Python implementation of the <em><strong>ACO</strong>-GRAANK</em> algorithm. The algorithm utilizes a pheromone-based (or probabilistic) strategy to optimize the <em>GRAANK</em> algorithm. 
<!-- Research paper published at -- link<br> -->

### Requirements:
You will be required to install the following python dependencies before using <em><strong>ACO</strong>GRAANK</em> algorithm:
```
                   install python (version => 3.0)

```

```
                    $ pip3 install numpy python-dateutil matplotlib

```

### Usage:
Use it a command line program with the local package:
```
$python3 init_acograd.py -f filename.csv -s minSup -t steps -n combinations
```

Example with a sample data-set<br>
```
python3 init_acograd.py -f ../data/DATASET.csv
```

<strong>Output</strong>
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
* Runkler, T. A. (2005), Ant colony optimization of clustering models. Int. J. Intell. Syst., 20: 1233-1251. doi:10.1002/int.20111
* Anne Laurent, Marie-Jeanne Lesot, and Maria Rifqi. 2009. GRAANK: Exploiting Rank Correlations for Extracting Gradual Itemsets. In Proceedings of the 8th International Conference on Flexible Query Answering Systems (FQAS '09). Springer-Verlag, Berlin, Heidelberg, 382-393.
* Dickson Owuor, Anne Laurent, and Joseph Orero (2019). Mining Fuzzy-temporal Gradual Patterns. In the proceedings of the 2019 IEEE International Conference on Fuzzy Systems (FuzzIEEE). IEEE.
