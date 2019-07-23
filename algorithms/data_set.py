"""
@author: "Dickson Owuor"
@credits: "Anne Laurent, Joseph Orero"
@license: "MIT"
@version: "1.0"
@email: "owuordickson@gmail.com"

"""
import csv


def read_csv(file):
    # 1. retrieve data-set from file
    with open(file, 'r') as f:
        dialect = csv.Sniffer().sniff(f.read(1024), delimiters=";,' '\t")
        f.seek(0)
        reader = csv.reader(f, dialect)
        temp = list(reader)
        f.close()
    return temp
