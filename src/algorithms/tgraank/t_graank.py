# -*- coding: utf-8 -*-
"""
@author: "Dickson Owuor"
@credits: "Joseph Orero and Anne Laurent,"
@license: "MIT"
@version: "2.0"
@email: "owuordickson@gmail.com"
@created: "19 November 2019"



Description: updated version that uses aco-graank and parallel multi-processing

"""


from src import HandleData, GradACO


class TgradACO:

    def __init__(self, d_set, ref_item, min_sup, min_rep):
        # For tgraank
        self.d_set = d_set
        cols = d_set.get_time_cols()
        if len(cols) > 0:
            print("Dataset Ok")
            self.time_ok = True
            self.time_cols = cols
            self.min_sup = min_sup
            self.ref_item = ref_item
            self.max_step = self.get_max_step(min_rep)
            # self.multi_data = self.split_dataset()
        else:
            print("Dataset Error")
            self.time_ok = False
            self.time_cols = []
            raise Exception('No date-time data found')

    def run_tgraank(self):
        # implement parallel multi-processing
        patterns = list()
        for s in range(self.max_step):
            step = s+1  # because for-loop is not inclusive from range: 0 - max_step
            # 1. Calculate representativity
            chk_rep, rep_info = self.get_representativity(step)
            # print(rep_info)
            if chk_rep:
                # 2. Transform data
                data, time_diffs = self.transform_data(step)
                d_set = HandleData("", attr_data=[self.d_set.column_size, data])

                # 3. Execute aco-graank for each transformation
                ac = GradACO(d_set)
                list_gp = ac.run_ant_colony(self.min_sup, time_diffs)
                if len(list_gp) > 0:
                    patterns.append(list_gp)
                # print("\nPheromone Matrix")
                # print(ac.p_matrix)
        return patterns

    def transform_data(self, step):
        # NB: Restructure dataset based on reference item
        data = self.d_set.data
        if self.time_ok:
            # 1. Calculate time difference using step
            ok, time_diffs = self.get_time_diffs(step)
            if not ok:
                msg = "Error: Time in row " + str(time_diffs[0]) + " or row " + str(time_diffs[1]) + " is not valid."
                raise Exception(msg)
            else:
                ref_col = self.ref_item
                if ref_col in self.time_cols:
                    msg = "Reference column is a 'date-time' attribute"
                    raise Exception(msg)
                elif (ref_col < 0) or (ref_col >= len(self.d_set.title)):
                    msg = "Reference column does not exist\nselect column between: " \
                          "0 and "+str(len(self.d_set.title) - 1)
                    raise Exception(msg)
                else:
                    # 1. Split the original data-set into column-tuples
                    attr_cols = self.d_set.attr_data

                    # 2. Transform the data using (row) n+step
                    new_data = list()
                    size = len(data)
                    for obj in attr_cols:
                        col_index = int(obj[0])
                        tuples = obj[1]
                        temp_tuples = list()
                        if (col_index - 1) == ref_col:
                            # reference attribute (skip)
                            for i in range(0, (size-step)):
                                temp_tuples.append(tuples[i])
                        else:
                            for i in range(step, size):
                                temp_tuples.append(tuples[i])
                        var_attr = [col_index, temp_tuples]
                        new_data.append(var_attr)
                    return new_data, time_diffs
        else:
            msg = "Fatal Error: Time format in column could not be processed"
            raise Exception(msg)

    def get_representativity(self, step):
        # 1. Get all rows minus the title row (already removed)
        all_rows = len(self.d_set.data)

        # 2. Get selected rows
        incl_rows = (all_rows - step)

        # 3. Calculate representativity
        if incl_rows > 0:
            rep = (incl_rows / float(all_rows))
            info = {"Transformation": "n+"+str(step), "Representativity": rep, "Included Rows": incl_rows,
                    "Total Rows": all_rows}
            return True, info
        else:
            return False, "Representativity is 0%"

    def get_max_step(self, minrep):
        # 1. count the number of steps each time comparing the
        # calculated representativity with minimum representativity
        size = len(self.d_set.data)
        for i in range(size):
            check, info = self.get_representativity(i + 1)
            if check:
                rep = info['Representativity']
                if rep < minrep:
                    return i
            else:
                return 0

    def get_time_diffs(self, step):
        data = self.d_set.data
        size = len(data)
        time_diffs = []
        for i in range(size):
            if i < (size - step):
                # temp_1 = self.data[i][0]
                # temp_2 = self.data[i + step][0]
                temp_1 = temp_2 = ""
                for col in self.time_cols:
                    temp_1 += " "+str(data[i][int(col)])
                    temp_2 += " "+str(data[i + step][int(col)])
                stamp_1 = HandleData.get_timestamp(temp_1)
                stamp_2 = HandleData.get_timestamp(temp_2)
                if (not stamp_1) or (not stamp_2):
                    return False, [i + 1, i + step + 1]
                time_diff = (stamp_2 - stamp_1)
                time_diffs.append(time_diff)
        # print("Time Diff: " + str(time_diff))
        return True, time_diffs
