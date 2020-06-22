import csv


def read_csv(file):
    # 1. retrieve data-set from file
    with open(file, 'r') as f:
        dialect = csv.Sniffer().sniff(f.readline(), delimiters=";,' '\t")
        f.seek(0)
        reader = csv.reader(f, dialect)
        temp = list(reader)
    for i in range(1, len(temp)):
        row = temp[i]
        for j in range(len(row)):
            try:
                x = float(temp[i][j])
            except ValueError:
                temp[i][j] = 0
    with open(file, 'w') as f:
        writer = csv.writer(f, delimiter=';')
        for line in temp:
            writer.writerow(line)
    return temp


read_csv('../data/FluTopicData-testsansdate-blank.csv')
