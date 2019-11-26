import sys
from optparse import OptionParser
from src import DataTransform, graank, trad, check_for_pattern, get_maximal_items, mbdll_border


def algorithm_ep_init(filename, ref_item, minsup, minrep):
    try:
        fgp_list = list()  # fuzzy-temporal gradual patterns

        # 1. Load dataset into program
        dataset = DataTransform(filename, ref_item, minrep)

        # 2. TRANSFORM DATA (for each step)
        for s in range(dataset.max_step):
            step = s+1  # because for-loop is not inclusive from range: 0 - max_step
            # 3. Calculate representativity
            chk_rep, rep_info = dataset.get_representativity(step)

            if chk_rep:
                # 4. Transform data
                data, time_diffs = dataset.transform_data(step)

                # 5. Execute GRAANK for each transformation
                title, gp_list, sup_list, tlag_list = graank(trad(list(data)), minsup, time_diffs, eq=False)

                pattern_found = check_for_pattern(ref_item, gp_list)
                if pattern_found:
                    maximal_items = get_maximal_items(gp_list, tlag_list)
                    fgp_list.append(tuple((title, maximal_items)))
        if not fgp_list:
            print("Oops! no frequent patterns were found")
            print("----------------------------------------------------------------")
        else:
            print("Total Data Transformations: " + str(dataset.max_step) + " | " + "Minimum Support: " + str(min_sup))
            print("----------------------------------------------------------------")
            for line in title:
                print(line)
            print('Emerging Pattern | Time Lags: (Transformation n, Transformation m)')

            all_fgps = list()
            for item_list in fgp_list:
                for item in item_list[1]:
                    all_fgps.append(item)

            patterns = 0
            ep_list = list()
            for i in range(len(all_fgps)):
                for j in range(i, len(all_fgps)):
                    if i != j:
                        freq_pattern_1 = all_fgps[i]
                        freq_pattern_2 = all_fgps[j]
                        ep = mbdll_border(tuple(freq_pattern_1[0]), tuple(freq_pattern_2[0]))
                        tlags = tuple((freq_pattern_1[1], freq_pattern_2[1]))
                        if ep:
                            patterns = patterns + 1
                            temp = tuple((ep, tlags))
                            ep_list.append(temp)
                            print(str(temp[0]) + " | " + str(temp[1]))

            print("\nTotal: " + str(patterns) + " FtGEPs found!")
            print("---------------------------------------------------------")
            if patterns == 0:
                print("Oops! no relevant emerging pattern was found")
                print("---------------------------------------------------------")
    except Exception as error:
        print(error)


if __name__ == "__main__":

    if not sys.argv:
        pattern_type = sys.argv[1]
        file_name = sys.argv[2]
        ref_col = sys.argv[3]
        min_sup = sys.argv[4]
        min_rep = sys.argv[5]

    else:
        optparser = OptionParser()
        optparser.add_option('-t', '--patternType',
                             dest='pType',
                             help='patterns: FtGP, FtGEP',
                             default=1,
                             type='int')
        optparser.add_option('-f', '--inputFile',
                             dest='file',
                             help='path to file containing csv',
                             default=None,
                             type='string')
        optparser.add_option('-c', '--refColumn',
                             dest='refCol',
                             help='reference column',
                             default=0,
                             type='int')
        optparser.add_option('-s', '--minSupport',
                             dest='minSup',
                             help='minimum support value',
                             default=0.7,
                             type='float')
        optparser.add_option('-r', '--minRepresentativity',
                             dest='minRep',
                             help='minimum representativity',
                             default=0.5,
                             type='float')

        (options, args) = optparser.parse_args()

        inFile = None
        if options.file is None:
            #inFile = 'DATASET.csv'
            #inFile = '../data/rain_temp1991-2015.csv'
            #inFile = '../data/ICU_household_power_consumption1.csv'
            inFile = '../data/ICU_household_power_consumption2.csv'
            #inFile = '../data/ICU_household_power_consumption.csv'

            #print("Usage: $python t_graank.py -f filename.csv -c refColumn -s minSup  -r minRep")
            #sys.exit('System will exit')
        else:
            inFile = options.file

        file_name = inFile
        pattern_type = options.pType
        ref_col = options.refCol
        min_sup = options.minSup
        min_rep = options.minRep

    #import timeit
    algorithm_ep_init(file_name, ref_col, min_sup, min_rep)