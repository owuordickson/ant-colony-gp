"""
@author: "Dickson Owuor"
@credits: "Anne Laurent, Joseph Orero"
@license: "MIT"
@version: "1.0"
@email: "owuordickson@gmail.com"

This code approximates a time period (member) using a fuzzy triangular membership function

"""

import skfuzzy as fuzzy
import numpy as np


def init_fuzzy_support(test_members, all_members, minsup):
    boundaries, extremes = get_membership_boundaries(all_members)
    value, sup = approximate_fuzzy_support(minsup, test_members, boundaries, extremes)
    return value, sup


def get_membership_boundaries(members):
    # 1. Sort the members in ascending order
    members.sort()

    # 2. Get the boundaries of membership function
    min = np.min(members)
    q_1 = np.percentile(members, 25)  # Quartile 1
    med = np.percentile(members, 50)
    q_3 = np.percentile(members, 75)
    max = np.max(members)
    boundaries = [q_1, med, q_3]
    extremes = [min, max]

    return boundaries,extremes


def approximate_fuzzy_support(minsup, timelags, orig_boundaries, extremes):
    slice_gap = (0.1*int(orig_boundaries[1]))
    sup = sup1 = 0
    slide_left = slide_right = expand = False
    sample = np.percentile(timelags, 50)

    a = orig_boundaries[0]
    b = b1 = orig_boundaries[1]
    c = orig_boundaries[2]
    min_a = extremes[0]
    max_c = extremes[1]
    boundaries = np.array(orig_boundaries)
    time_lags = np.array(timelags)

    while sup <= minsup:

        if sup > sup1:
            sup1 = sup
            b1 = b

        # Calculate membership of frequent path
        memberships = fuzzy.membership.trimf(time_lags, boundaries)

        # Calculate support
        sup = calculate_support(memberships)

        if sup >= minsup:
            value = get_time_format(b)
            return value, sup
        else:
            if not slide_left:
                # 7. Slide to the left to change boundaries
                # if extreme is reached - then slide right
                if sample <= b:
                # if min_a >= b:
                    a = a - slice_gap
                    b = b - slice_gap
                    c = c - slice_gap
                    boundaries = np.array([a, b, c])
                else:
                    slide_left = True
            elif not slide_right:
                # 8. Slide to the right to change boundaries
                # if extreme is reached - then slide right
                if sample >= b:
                #if max_c <= b:
                    a = a + slice_gap
                    b = b + slice_gap
                    c = c + slice_gap
                    boundaries = np.array([a, b, c])
                else:
                    slide_right = True
            elif not expand:
                # 9. Expand quartiles and repeat 5. and 6.
                a = min_a
                b = orig_boundaries[1]
                c = max_c
                boundaries = np.array([a, b, c])
                slide_left = slide_right = False
                expand = True
            else:
                value = get_time_format(b1)
                return value, False


def calculate_support(memberships):
    support = 0
    if len(memberships) > 0:
        sup_count = 0
        total = len(memberships)
        for member in memberships:
            # if float(member) > 0.5:
            if float(member) > 0:
                sup_count = sup_count + 1
        support = sup_count / total
    return support


def get_time_format(value):
    if value < 0:
        sign = "-"
    else:
        sign = "+"
    p_value, p_type = round_time(abs(value))
    p_format = [sign,p_value,p_type]
    return p_format


def round_time(seconds):
    years = seconds/3.154e+7
    months = seconds/2.628e+6
    weeks = seconds/604800
    days = seconds/86400
    hours = seconds/3600
    minutes = seconds/60

    if int(years) <= 0:
        if int(months) <= 0:
            if int(weeks) <= 0:
                if int(days) <= 0:
                    if int(hours) <= 0:
                        if int(minutes) <= 0:
                            return seconds, "seconds"
                        else:
                            return minutes, "minutes"
                    else:
                        return hours, "hours"
                else:
                    return days, "days"
            else:
                return weeks,"weeks"
        else:
            return months, "months"
    else:
        return years, "years"
