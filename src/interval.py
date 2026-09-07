from interlap import InterLap
from collections import OrderedDict

__author__ = 'Yuxing Xu'


def section(inter_a, inter_b, int_flag=False, just_judgement=False):
    """
    get the section
    :param inter_a:
    :param inter_b:
    :return:
    """
    all = sorted(list(inter_a) + list(inter_b))
    deta = (all[1], all[2])
    if int_flag is False:
        if max(inter_a) >= min(inter_b) and max(inter_b) >= min(inter_a):
            If_inter = True  # Yes
        else:
            If_inter = False  # No

        if just_judgement:
            return If_inter
        else:
            return If_inter, deta

    else:
        if max(inter_a) - min(inter_b) >= -1 and max(inter_b) - min(inter_a) >= -1:
            If_inter = True  # Yes
        else:
            If_inter = False  # No

        if just_judgement:
            return If_inter
        else:
            return If_inter, deta


def merge_intervals(input_list, int=False):
    """
    a function that will merge overlapping intervals
    :param intervals: a list of tuples
                      e.g. intervals = [(1,5),(33,35),(40,33),(10,15),(13,18),(28,23),(70,80),(22,25),(38,50),(40,60)]
    :param int: if the data is all int
    :return: merged list
    """
    intervals = []
    for i in input_list:
        intervals.append(tuple(i))

    sorted_by_lower_bound = sorted(intervals, key=lambda tup: min(tup))
    merged = []

    for higher in sorted_by_lower_bound:
        if not merged:
            merged.append(higher)
        else:
            lower = merged[-1]
            # test for intersection between lower and higher:
            # we know via sorting that lower[0] <= higher[0]
            if int is False:
                if min(higher) <= max(lower):
                    upper_bound = max(lower + higher)
                    # replace by merged interval
                    merged[-1] = (min(lower), upper_bound)
                else:
                    merged.append(higher)
            elif int is True:
                if min(higher) <= max(lower):
                    upper_bound = max(lower + higher)
                    # replace by merged interval
                    merged[-1] = (min(lower), upper_bound)
                elif max(lower) + 1 == min(higher):
                    upper_bound = max(lower + higher)
                    # replace by merged interval
                    merged[-1] = (min(lower), upper_bound)
                else:
                    merged.append(higher)
    return merged


def overturn(inter_list):
    """
    input a list of intervals and the function will overturn it to a list with gap of the intervals
    :param inter_list:
    :return: gap_list
    """
    inter_list = sorted(merge_intervals(inter_list, True))
    output_list = []
    last_right = 0
    for index in range(0, len(inter_list) + 1):
        if index == 0:
            output_list.append((float('-inf'), inter_list[index][0] - 1))
            last_right = inter_list[index][1]
        elif index == len(inter_list):
            output_list.append((last_right + 1, float('inf')))
        else:
            output_list.append((last_right + 1, inter_list[index][0] - 1))
            last_right = inter_list[index][1]
    return output_list


def interval_minus_set(target, bullets):
    if len(bullets) == 0:
        return [target]
    gaps = overturn(bullets)
    output_list = []
    for i in gaps:
        If_inter, deta = section(target, i)
        if If_inter:
            output_list.append(deta)
    return output_list


def overlap_between_interval_set(interval1, interval2, similarity_type='shorter_overlap_coverage'):
    """
    used for compare two interval_set

    similarity_type can be: "shorter_overlap_coverage" or "jaccord_score"

    interval1 = [(1,20), (50,60), (80,100)]
    interval2 = [(50,70), (90,100)]
    """

    interval1 = merge_intervals(interval1)
    interval2 = merge_intervals(interval2)

    interval2_interLap = InterLap(interval2)

    overlap = []
    for it1 in interval1:
        for it2 in interval2_interLap.find(it1):
            flag, deta = section(it1, it2, True)
            if flag:
                overlap.append(deta)

    overlap = merge_intervals(overlap)

    overlap_length = sum_interval_length(overlap)
    if similarity_type == 'shorter_overlap_coverage':
        shorter_len = min(sum_interval_length(interval1),
                          sum_interval_length(interval2))
        if shorter_len > 0:
            overlap_ratio = overlap_length/shorter_len
        else:
            overlap_ratio = 0

    elif similarity_type == 'jaccord_score':
        overlap_ratio = overlap_length / \
            sum_interval_length(merge_intervals(interval1 + interval2))

    return overlap_ratio, overlap_length, overlap


def group_by_intervals(list_input, int=False):
    """
    :param list_input: [(ID1,start,end),(ID2,start,end),(ID3,start,end),(ID4,start,end)]
    :return: dict = {
                    "group1" : {
                                "range" : (start,end)
                                "list" : [ID1,ID2]
                                }
                    "group2" : {
                                "range" : (start,end)
                                "list" : [ID3,ID4]
                                }
                    }
    """
    range_dict = {}
    intervals = []

    for ID, start, end in list_input:
        range_key = (min(start, end), max(start, end))
        if not range_key in range_dict:
            range_dict[range_key] = []
        range_dict[range_key].append(ID)
        intervals.append(range_key)

    intervals = list(set(intervals))

    sorted_by_lower_bound = sorted(intervals, key=lambda tup: min(tup))
    merged = []
    merged_ID = {}
    for higher in sorted_by_lower_bound:
        if not merged:
            merged.append(higher)
            merged_ID[higher] = range_dict[higher]
        else:
            lower = merged[-1]
            # test for intersection between lower and higher:
            # we know via sorting that lower[0] <= higher[0]
            if int is False:
                if min(higher) <= max(lower):
                    upper_bound = max(lower + higher)
                    later_ID = merged_ID[merged[-1]]
                    del merged_ID[merged[-1]]
                    merged_ID[(min(lower), upper_bound)] = later_ID + \
                        list(range_dict[higher])
                    # replace by merged interval
                    merged[-1] = (min(lower), upper_bound)
                else:
                    merged.append(higher)
                    merged_ID[higher] = range_dict[higher]
            elif int is True:
                if min(higher) <= max(lower):
                    upper_bound = max(lower + higher)
                    later_ID = merged_ID[merged[-1]]
                    del merged_ID[merged[-1]]
                    merged_ID[(min(lower), upper_bound)] = later_ID + \
                        list(range_dict[higher])
                    # replace by merged interval
                    merged[-1] = (min(lower), upper_bound)
                elif max(lower) + 1 == min(higher):
                    upper_bound = max(lower + higher)
                    later_ID = merged_ID[merged[-1]]
                    del merged_ID[merged[-1]]
                    merged_ID[(min(lower), upper_bound)] = later_ID + \
                        list(range_dict[higher])
                    # replace by merged interval
                    merged[-1] = (min(lower), upper_bound)
                else:
                    merged.append(higher)
                    merged_ID[higher] = range_dict[higher]
    return merged_ID


def group_by_intervals_with_overlap_threshold(input_dir, overlap_threshold=0.5):
    """
    :param input_dir =
                    {
                    "ID1" : (start,end),
                    "ID2" : (start,end),
                    "ID3" : (start,end),
                    "ID4" : (start,end)
                    }
            input_dir = {
                "A": (100, 500),
                "B": (99, 501),
                "C": (480, 1000),
                "D": (500, 1005),
                "E": (80, 600),
                "F": (600, 1100),
                "G": (2000, 2500),
                "H": (1900, 2600),
                "I": (2100, 2600),
                "J": (430, 500),
                "K": (100, 500),
                "L": (600, 1000),
            }


    :return: dict = {
                    "group1" : {
                                "range" : (start,end)
                                "list" : [ID1,ID2]
                                }
                    "group2" : {
                                "range" : (start,end)
                                "list" : [ID3,ID4]
                                }
                    }
    """

    def higer_in_lower_group(merged_group, lower_group_id, higher, overlap_threshold):
        grouped_flag = 0
        for i in merged_group[lower_group_id]:
            ifflag, deta = section(higher, i)
            if ifflag:
                len_h = max(higher[0], higher[1]) - \
                    min(higher[0], higher[1]) + 1
                len_i = max(i[0], i[1]) - min(i[0], i[1]) + 1
                len_d = max(deta[0], deta[1]) - min(deta[0], deta[1]) + 1
                if len_d / len_h > overlap_threshold or len_d / len_i > overlap_threshold:
                    grouped_flag = 1
                    break
        return grouped_flag

    range_dict = {}
    intervals = []
    for ID in input_dir:
        start, end = input_dir[ID]
        range_key = (min(start, end), max(start, end))
        if not range_key in range_dict:
            range_dict[range_key] = []
        range_dict[range_key].append(ID)
        intervals.append(range_key)
    intervals = list(set(intervals))

    sorted_by_lower_bound = sorted(intervals, key=lambda tup: min(tup))
    merged_group = OrderedDict()
    group_id = 0
    for higher in sorted_by_lower_bound:
        if len(merged_group) == 0:
            group_id = group_id + 1
            merged_group["group_%d" % group_id] = [higher]
        else:
            lower_group_id = list(merged_group.keys())[-1]
            # test for intersection between lower and higher:
            # we know via sorting that lower[0] <= higher[0]
            if higer_in_lower_group(merged_group, lower_group_id, higher, overlap_threshold):
                merged_group["group_%d" % group_id].append(higher)
            else:
                group_id = group_id + 1
                merged_group["group_%d" % group_id] = [higher]

    for group_id in merged_group:
        range_list = merged_group[group_id]
        input_id_list = []
        for i in range_list:
            input_id_list.extend(range_dict[i])
        merged_group[group_id] = {
            "range": merge_intervals(range_list),
            "list": input_id_list
        }

    return merged_group


def sum_interval_length(range_list, close_range=True):
    length_sum = 0
    for i in range_list:
        if close_range:
            length_sum = length_sum + max(i) - min(i) + 1
        else:
            length_sum = length_sum + max(i) - min(i)
    return length_sum