import csv
import numpy as np
import math
import cv2

# img_path_in = './apple/runs/detect/exp13'
csv_path_in = 'pren_output_23122021_neweff_new_lion_4.csv'
csv_path_out = 'processed_output_private.csv'
output_txt = ''

nowIndex = 0

imgNum_list = []
point_list = []
label_list = []
midpoint_list = []
area_list = []
width_list = []
height_list = []


def slope(p1, p2):  # if inf -> return 1e10
    if p1[0] != p2[0]:
        return (p1[1] - p2[1]) / (p1[0] - p2[0])
    return 1e10


def find2d(target, list2d):
    for row in list2d:
        if target in row:
            return 1
    return 0


def tri_area(l1, l2, l3):
    s = (l1 + l2 + l3) / 2
    re = s * (s - l1) * (s - l2) * (s - l3)
    return math.sqrt(re)


def in_rectangle(p_target, p_width, p_height, p_list):  # p_list = [[p1.x, p1.y], [p2.x, p2.y], ...]
    l12 = dist(p_list[0], p_list[1])
    l23 = dist(p_list[1], p_list[2])
    l34 = dist(p_list[2], p_list[3])
    l14 = dist(p_list[0], p_list[3])
    l13 = dist(p_list[0], p_list[2])
    recArea = tri_area(l12, l23, l13) + tri_area(l14, l13, l34)
    p_target_4 = [[p_target[0] - p_width / 4, p_target[1] - p_height / 4],
                  [p_target[0] - p_width / 4, p_target[1] + p_height / 4],
                  [p_target[0] + p_width / 4, p_target[1] - p_height / 4],
                  [p_target[0] + p_width / 4, p_target[1] + p_height / 4]]

    for p in p_target_4:
        l10 = dist(p_list[0], p)
        l20 = dist(p_list[1], p)
        l30 = dist(p_list[2], p)
        l40 = dist(p_list[3], p)
        if abs(slope(p, p_list[0]) - slope(p, p_list[1])) > 1e-6 and \
                abs(slope(p, p_list[1]) - slope(p, p_list[2])) > 1e-6 and \
                abs(slope(p, p_list[2]) - slope(p, p_list[3])) > 1e-6 and \
                abs(slope(p, p_list[3]) - slope(p, p_list[1])) > 1e-6:
            innArea = tri_area(l10, l20, l12) + tri_area(l20, l30, l23) + tri_area(l30, l40, l34) + tri_area(l10, l40,
                                                                                                             l14)
            if abs(recArea - innArea) > 1e-3:
                return 0
        else:
            return 0
    return 1


def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def area_merge(p1_list, p2_list):
    minx = min(p1_list[0][0], p1_list[2][0], p2_list[0][0], p2_list[2][0])
    maxx = max(p1_list[0][0], p1_list[2][0], p2_list[0][0], p2_list[2][0])
    miny = min(p1_list[0][1], p1_list[2][1], p2_list[0][1], p2_list[2][1])
    maxy = max(p1_list[0][1], p1_list[2][1], p2_list[0][1], p2_list[2][1])
    return (maxx - minx) * (maxy - miny)


def compare_x_y_theta(p1, p2, th):
    if -math.pi / 3 < th < math.pi / 3:
        if p1[0] < p2[0]:
            return -1
        elif p1[0] > p2[0]:
            return 1
        else:
            if p1[1] < p2[1]:
                return -1
            elif p1[1] > p2[1]:
                return 1
            else:
                return 0
    else:
        if p1[1] < p2[1]:
            return -1
        elif p1[1] > p2[1]:
            return 1
        else:
            if p1[0] < p2[0]:
                return -1
            elif p1[0] > p2[0]:
                return 1
            else:
                return 0


def sort_area_width_height(a_list, w_list, h_list, e_list):
    for id1 in range(len(a_list) - 1):
        tmp_id = id1
        for id2 in range(id1 + 1, len(a_list)):
            if a_list[id2] < a_list[tmp_id]:
                tmp_id = id2
            elif a_list[id2] == a_list[tmp_id]:
                if w_list[id2] < w_list[tmp_id]:
                    tmp_id = id2
                elif w_list[id2] == w_list[tmp_id]:
                    if h_list[id2] < h_list[tmp_id]:
                        tmp_id = id2
        if tmp_id != id1:
            tmp = a_list[id1]
            a_list[id1] = a_list[tmp_id]
            a_list[tmp_id] = tmp
            tmp = w_list[id1]
            w_list[id1] = w_list[tmp_id]
            w_list[tmp_id] = tmp
            tmp = h_list[id1]
            h_list[id1] = h_list[tmp_id]
            h_list[tmp_id] = tmp
            tmp = e_list[id1]
            e_list[id1] = e_list[tmp_id]
            e_list[tmp_id] = tmp
    return a_list, w_list, h_list, e_list


def sort_theta(th_list, gr_list):
    for id1 in range(len(th_list) - 1):
        tmp_id = id1
        for id2 in range(id1 + 1, len(th_list)):
            if th_list[id2] < th_list[tmp_id]:
                tmp_id = id2
        if tmp_id != id1:
            t = th_list[id1]
            th_list[id1] = th_list[tmp_id]
            th_list[tmp_id] = t
            t = gr_list[id1]
            gr_list[id1] = gr_list[tmp_id]
            gr_list[tmp_id] = t

    return th_list, gr_list


def sort_x_y_unique_theta(th_list, gr_list):
    # sort by x or y
    for gr_id in range(len(gr_list)):
        for w1_id in range(len(gr_list[gr_id]) - 1):
            tmp_id = w1_id
            for w2_id in range(w1_id + 1, len(gr_list[gr_id])):
                cmp = compare_x_y_theta(midpoint_list[gr_list[gr_id][w2_id]], midpoint_list[gr_list[gr_id][tmp_id]],
                                        th_list[gr_id])
                if cmp == -1:
                    tmp_id = w2_id
            if tmp_id != w1_id:
                t = gr_list[gr_id][w1_id]
                gr_list[gr_id][w1_id] = gr_list[gr_id][tmp_id]
                gr_list[gr_id][tmp_id] = t

    # unique 2d list
    id1 = 0
    while id1 < len(gr_list):
        id2 = id1 + 1
        while id2 < len(gr_list):
            if gr_list[id2] == gr_list[id1]:
                del gr_list[id2]
                del th_list[id2]
            else:
                id2 += 1
        id1 += 1

    # sort by theta
    for id1 in range(len(th_list) - 1):
        tmp_id = id1
        for id2 in range(id1 + 1, len(th_list)):
            if th_list[id2] < th_list[tmp_id]:
                tmp_id = id2
        if tmp_id != id1:
            t = th_list[id1]
            th_list[id1] = th_list[tmp_id]
            th_list[tmp_id] = t
            t = gr_list[id1]
            gr_list[id1] = gr_list[tmp_id]
            gr_list[tmp_id] = t

    return th_list, gr_list

def sort_x_y_theta(gr_list):
    for gr_id in range(len(gr_list)):
        if abs(midpoint_list[gr_list[gr_id][0]][0] - midpoint_list[gr_list[gr_id][1]][0]) < 1e-10:
            th = math.pi/2 - 1e-10
        else:
            th = math.atan((midpoint_list[gr_list[gr_id][0]][1] - midpoint_list[gr_list[gr_id][1]][1])/(midpoint_list[gr_list[gr_id][0]][0] - midpoint_list[gr_list[gr_id][1]][0]))
        for w1_id in range(len(gr_list[gr_id]) - 1):
            tmp_id = w1_id
            for w2_id in range(w1_id + 1, len(gr_list[gr_id])):
                cmp = compare_x_y_theta(midpoint_list[gr_list[gr_id][w2_id]], midpoint_list[gr_list[gr_id][tmp_id]], th)
                if cmp == -1:
                    tmp_id = w2_id
            if tmp_id != w1_id:
                t = gr_list[gr_id][w1_id]
                gr_list[gr_id][w1_id] = gr_list[gr_id][tmp_id]
                gr_list[gr_id][tmp_id] = t
    return gr_list

def sort_length(list2d):
    for id1 in range(len(list2d) - 1):
        tmp_id = id1
        for id2 in range(id1 + 1, len(list2d)):
            if len(list2d[id2]) > len(list2d[tmp_id]):
                tmp_id = id2
        if tmp_id != id1:
            t = list2d[id1]
            list2d[id1] = list2d[tmp_id]
            list2d[tmp_id] = t
    return list2d


def remove_short_in_long(sorted_len_list):
    id1 = 0
    while id1 < len(sorted_len_list):
        id2 = id1 + 1
        while id2 < len(sorted_len_list):
            del_flag = 0
            for ele in sorted_len_list[id2]:
                if ele in sorted_len_list[id1]:
                    del sorted_len_list[id2]
                    del_flag = 1
                    break
            if not del_flag:
                id2 += 1
        id1 += 1

    return sorted_len_list

def remove_short_in_long_with_theta(sorted_len_list, th_list):
    id1 = 0
    while id1 < len(sorted_len_list):
        id2 = id1 + 1
        while id2 < len(sorted_len_list):
            del_flag = 0
            for ele in sorted_len_list[id2]:
                if ele in sorted_len_list[id1]:
                    del th_list[id2]
                    del sorted_len_list[id2]
                    del_flag = 1
                    break
            if not del_flag:
                id2 += 1
        id1 += 1

    return sorted_len_list, th_list

with open(csv_path_in, 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter="\n")
    for i, line in enumerate(reader):
        ll = line[0].split(',')
        imgNum_list.append(ll[0])
        point_list.append(
            [[int(ll[1]), int(ll[2])], [int(ll[3]), int(ll[4])], [int(ll[5]), int(ll[6])], [int(ll[7]), int(ll[8])]])
        label_list.append(ll[9])
        midpoint_list.append([(int(ll[1]) + int(ll[3]) + int(ll[5]) + int(ll[7])) / 4,
                              (int(ll[2]) + int(ll[4]) + int(ll[6]) + int(ll[8])) / 4])
        area_list.append(abs(int(ll[1]) - int(ll[5])) * abs(int(ll[2]) - int(ll[6])))
        width_list.append(abs(int(ll[1]) - int(ll[5])))
        height_list.append(abs(int(ll[2]) - int(ll[6])))
    f.close()

[_, imgIndex_list] = np.unique(imgNum_list, return_index=True)
imgIndex_list = imgIndex_list.tolist()
imgIndex_list.pop(0)
imgIndex_list.append(len(imgNum_list))
# print(imgIndex_list)
# input()

for endIndex in imgIndex_list:
    print(str(imgNum_list[nowIndex]), nowIndex, endIndex)
    image = cv2.imread('./private/' + str(imgNum_list[nowIndex]) + '.jpg')
    thickness = 3

    prior_ans = []
    element_list = list(range(nowIndex, endIndex))

    ele1_id = 0
    while ele1_id < len(element_list):
        ele2_id = 0
        ele1_group = []
        while ele2_id < len(element_list) and ele1_id < len(element_list):
            # print(element_list[ele1_id], element_list[ele2_id])
            # print(element_list)
            if ele2_id == ele1_id:
                ele2_id += 1
                continue
            if in_rectangle(midpoint_list[element_list[ele2_id]], width_list[element_list[ele2_id]],
                            height_list[element_list[ele2_id]], point_list[element_list[ele1_id]]):
                if element_list[ele1_id] not in ele1_group:
                    ele1_group.append(element_list[ele1_id])
                ele1_group.append(element_list[ele2_id])
                del element_list[ele2_id]
                # print('merge', ele1_id, ele2_id)
                # print(element_list)
            else:
                ele2_id += 1
        if ele1_group:
            prior_ans.append(ele1_group)
        ele1_id += 1

    ele_id = 0
    while ele_id < len(element_list):
        if label_list[element_list[ele_id]][0].isdigit():
            # print(label_list[element_list[ele_id]])
            del element_list[ele_id]
        else:
            ele_id += 1

    error_area_diff_sum = 0.15
    error_width_diff_sum = 0.3
    error_height_diff_sum = 0.3
    area_ele_list = []
    width_ele_list = []
    height_ele_list = []

    for ele_id in element_list:
        area_ele_list.append(area_list[ele_id])
        width_ele_list.append(width_list[ele_id])
        height_ele_list.append(height_list[ele_id])
    max_area = max(area_ele_list)

    except_area_list = []
    while 1:
        if len(area_ele_list) < 2:
            break
        area_ele_list, width_ele_list, height_ele_list, element_list = sort_area_width_height(area_ele_list,
                                                                                              width_ele_list,
                                                                                              height_ele_list,
                                                                                              element_list)
        # area_ele_list, width_ele_list, height_ele_list, element_list = (list(t) for t in zip(*sorted(
        #     zip(area_ele_list, width_ele_list, height_ele_list, element_list)))) -> error because cannot sort by list
        min_area_diff = 1e10
        id1 = -1
        for ele_id in range(len(element_list) - 1):
            if area_ele_list[ele_id + 1] - area_ele_list[ele_id] <= min_area_diff:
                if [element_list[ele_id], element_list[ele_id + 1]] not in except_area_list:
                    id1 = ele_id
                    min_area_diff = area_ele_list[ele_id + 1] - area_ele_list[ele_id]
        # input()
        # print(id1, id1 + 1)
        if min_area_diff > max_area or id1 == -1:
            break
        if min_area_diff < error_area_diff_sum * (area_ele_list[id1] + area_ele_list[id1 + 1]) and \
                abs(width_ele_list[id1] - width_ele_list[id1 + 1]) < error_width_diff_sum * (
                width_ele_list[id1] + width_ele_list[id1 + 1]) and \
                abs(height_ele_list[id1] - height_ele_list[id1 + 1]) < error_height_diff_sum * (
                height_ele_list[id1] + height_ele_list[id1 + 1]):
            if type(element_list[id1]) is int:
                len_id1 = 1
            else:
                len_id1 = len(element_list[id1])
            if type(element_list[id1 + 1]) is int:
                len_id2 = 1
            else:
                len_id2 = len(element_list[id1 + 1])
            height_ele_list[id1] = int(
                (height_ele_list[id1] * len_id1 + height_ele_list[id1 + 1] * len_id2) / (len_id1 + len_id2) + 0.5)
            width_ele_list[id1] = int(
                (width_ele_list[id1] * len_id1 + width_ele_list[id1 + 1] * len_id2) / (len_id1 + len_id2) + 0.5)
            area_ele_list[id1] = int(
                (area_ele_list[id1] * len_id1 + area_ele_list[id1 + 1] * len_id2) / (len_id1 + len_id2) + 0.5)
            if len_id1 == 1 and len_id2 == 1:
                element_list[id1] = [element_list[id1], element_list[id1 + 1]]
            elif len_id2 == 1:
                element_list[id1].append(element_list[id1 + 1])
            elif len_id1 == 1:
                element_list[id1 + 1].append(element_list[id1])
                element_list[id1] = element_list[id1 + 1]
            else:
                for ele in element_list[id1 + 1]:
                    element_list[id1].append(ele)
            del element_list[id1 + 1]
            del height_ele_list[id1 + 1]
            del width_ele_list[id1 + 1]
            del area_ele_list[id1 + 1]
        else:
            except_area_list.append([element_list[id1], element_list[id1 + 1]])

    group_id = 0
    jump_list = []
    # delete length = 1 and add small theta length = 2 to jump_list(not through theta split)
    while group_id < len(element_list):
        if type(element_list[group_id]) is int:
            del element_list[group_id]
            del area_ele_list[group_id]
            del width_ele_list[group_id]
            del height_ele_list[group_id]
        elif len(element_list[group_id]) == 2:
            if abs(midpoint_list[element_list[group_id][0]][0] - midpoint_list[element_list[group_id][1]][0]) <= \
                    0.7 * (width_list[element_list[group_id][0]] + width_list[element_list[group_id][1]]) \
                    and abs(midpoint_list[element_list[group_id][0]][1] - midpoint_list[element_list[group_id][1]][1]) <= \
                    0.7 * (height_list[element_list[group_id][0]] + height_list[element_list[group_id][1]]) \
                    and area_merge(point_list[element_list[group_id][0]], point_list[element_list[group_id][1]]) <= \
                    2.5 * (area_list[element_list[group_id][0]] + area_list[element_list[group_id][1]]):
                # dis1 = dist(midpoint_list[element_list[group_id][0]], [0, 0])
                # dis2 = dist(midpoint_list[element_list[group_id][1]], [0, 0])
                if abs(midpoint_list[element_list[group_id][0]][0] - midpoint_list[element_list[group_id][1]][0]) < 1e-10:
                    theta = math.pi/2 - 1e-10
                else:
                    theta = math.atan(
                        (midpoint_list[element_list[group_id][0]][1] - midpoint_list[element_list[group_id][1]][1]) / (
                                    midpoint_list[element_list[group_id][0]][0] -
                                    midpoint_list[element_list[group_id][1]][0]))
                if -math.pi / 6 < theta < math.pi / 6:
                    # sort by x
                    if midpoint_list[element_list[group_id][0]][0] < midpoint_list[element_list[group_id][1]][0]:
                        jump_list.append(element_list[group_id])
                    else:
                        jump_list.append([element_list[group_id][1], element_list[group_id][0]])
                    # print(theta, jump_list[-1])
                elif theta > 0.4 * math.pi or theta < -0.4 * math.pi:
                    # sort by y
                    if midpoint_list[element_list[group_id][0]][1] < midpoint_list[element_list[group_id][1]][1]:
                        jump_list.append(element_list[group_id])
                    else:
                        jump_list.append([element_list[group_id][1], element_list[group_id][0]])
                    # print(theta, jump_list[-1])
            del element_list[group_id]
            del area_ele_list[group_id]
            del width_ele_list[group_id]
            del height_ele_list[group_id]
        else:
            group_id += 1
    # print('After height and width')
    # print(element_list)

    # group inner theta
    for group in element_list:  # group = [12, 34, 21]
        theta_list = []
        graph_list = []
        # print('# words: ', len(group))
        # input()
        flag_keep_theta = 1
        for i in range(len(group) - 1):
            except_theta_list = []
            for j in range(i + 1, len(group)):
                if abs(midpoint_list[group[i]][0] - midpoint_list[group[j]][0]) < 1e-6:
                    theta = math.pi / 2 - 1e-10
                else:
                    theta = math.atan(
                        (midpoint_list[group[i]][1] - midpoint_list[group[j]][1]) / (
                                midpoint_list[group[i]][0] - midpoint_list[group[j]][0]))
                if theta > 0.42 * math.pi or theta < -0.42 * math.pi or -math.pi / 10 < theta < math.pi / 10:
                    theta_list.append(theta)
                    graph_list.append([group[i], group[j]])
            # sort by x, y
            theta_list, graph_list = sort_x_y_unique_theta(theta_list, graph_list)
            while flag_keep_theta:
                if len(theta_list) < 2:
                    break
                if len(theta_list) > 200:
                    flag_keep_theta = 0
                    break


                theta_list, graph_list = sort_theta(theta_list, graph_list)
                min_theta_diff = 1e10
                flagVert = 0
                id1 = -1
                id2 = -1
                for ii in range(len(theta_list) - 1):
                    if theta_list[ii + 1] - theta_list[ii] < min_theta_diff:
                        if [graph_list[ii], graph_list[ii + 1]] not in except_theta_list:
                            min_theta_diff = theta_list[ii + 1] - theta_list[ii]
                            id1 = ii
                            id2 = ii + 1
                if abs(theta_list[0] + math.pi - theta_list[-1]) < min_theta_diff \
                        and [graph_list[0], graph_list[-1]] not in except_theta_list:
                    id1 = 0
                    id2 = len(theta_list) - 1
                    min_theta_diff = abs(theta_list[0] + math.pi - theta_list[-1])
                    flagVert = 1

                if min_theta_diff > math.pi / 60 or id1 == -1:
                    break

                midpoint1 = [0, 0]
                midpoint2 = [0, 0]
                for ele_id in range(len(graph_list[id1])):
                    midpoint1[0] += midpoint_list[graph_list[id1][ele_id]][0]
                    midpoint1[1] += midpoint_list[graph_list[id1][ele_id]][1]
                midpoint1[0] /= len(graph_list[id1])
                midpoint1[1] /= len(graph_list[id1])
                for ele_id in range(len(graph_list[id2])):
                    midpoint2[0] += midpoint_list[graph_list[id2][ele_id]][0]
                    midpoint2[1] += midpoint_list[graph_list[id2][ele_id]][1]
                midpoint2[0] /= len(graph_list[id2])
                midpoint2[1] /= len(graph_list[id2])

                if abs(midpoint1[0] - midpoint2[0]) < 1e-10:
                    midTheta = math.pi / 2 - 1e-10
                else:
                    midTheta = math.atan((midpoint1[1] - midpoint2[1]) / (midpoint1[0] - midpoint2[0]))

                if flagVert:
                    theta_list[id1] += math.pi
                newTheta = (theta_list[id1] * (len(graph_list[id1]) - 1) +
                            theta_list[id2] * (len(graph_list[id2]) - 1)) / \
                           (len(graph_list[id1]) + len(graph_list[id2]) - 2)
                if newTheta >= math.pi / 2:
                    newTheta -= math.pi
                elif newTheta < -math.pi / 2:
                    newTheta += math.pi

                if flagVert:
                    theta_list[id1] -= math.pi
                if abs(newTheta - midTheta) < math.pi / 150 or (-math.pi / 150 < abs(
                        newTheta - midTheta) - math.pi < math.pi / 150):
                    theta_list[id1] = newTheta
                    for ele in graph_list[id2]:
                        if ele not in graph_list[id1]:
                            graph_list[id1].append(ele)
                    ex_id = 0
                    while ex_id < len(except_theta_list):
                        if theta_list[id2] in except_theta_list[ex_id]:
                            del except_theta_list[ex_id]
                        else:
                            ex_id += 1
                    del theta_list[id2]
                    del graph_list[id2]
                    # print(id1, id2)
                else:
                    except_theta_list.append([graph_list[id1], graph_list[id2]])
                    # print(id1, id2, 'EXCEPT')
            # print(i)
            # print('length of theta_list: ', len(theta_list))
            # print('length of except_list: ', len(except_theta_list))
            if not flag_keep_theta:
                break
            range_list = range(len(theta_list) - 1, -1, -1)
            for th_id in range_list:
                if len(graph_list[th_id]) < 3:
                    del theta_list[th_id]
                    del graph_list[th_id]
            # sort by x, y
            # theta_list, graph_list = sort_x_y_unique_theta(theta_list, graph_list)
        if not flag_keep_theta:
            graph_list = []
            theta_list = []
        # sort by x, y
        theta_list, graph_list = sort_x_y_unique_theta(theta_list, graph_list)

        # sort by length
        graph_list = sort_length(graph_list)
        # print('after theta')
        # print(graph_list)
        # split by merge area
        gr_id = 0
        while gr_id < len(graph_list):
            for ele_id in range(len(graph_list[gr_id]) - 1):
                if abs(midpoint_list[graph_list[gr_id][ele_id]][0] - midpoint_list[graph_list[gr_id][ele_id + 1]][0]) > \
                        0.7 * (width_list[graph_list[gr_id][ele_id]] + width_list[graph_list[gr_id][ele_id + 1]]) \
                        or abs(midpoint_list[graph_list[gr_id][ele_id]][1] - midpoint_list[graph_list[gr_id][ele_id + 1]][1]) > \
                        0.7 * (height_list[graph_list[gr_id][ele_id]] + height_list[graph_list[gr_id][ele_id + 1]]) \
                        or area_merge(point_list[graph_list[gr_id][ele_id]],point_list[graph_list[gr_id][ele_id + 1]]) > \
                        2.5 * (area_list[graph_list[gr_id][ele_id]] + area_list[graph_list[gr_id][ele_id + 1]]):
                    if len(graph_list[gr_id]) - 1 - ele_id > 1:
                        graph_list.append(graph_list[gr_id][ele_id + 1:len(graph_list[gr_id])])
                    del graph_list[gr_id][ele_id + 1:len(graph_list[gr_id])]
                    if ele_id == 0:
                        del graph_list[gr_id]
                        gr_id -= 1
                    break
            gr_id += 1

        # sort by length
        graph_list = sort_length(graph_list)
        graph_list = sort_x_y_theta(graph_list)
        # delete short list in long list
        graph_list = remove_short_in_long(graph_list)
        for gr in graph_list:
            jump_list.append(gr)
    #     print('after spllit merge')
    #     print(graph_list)
    #     input()
    # print('jump_list: ')
    # print(jump_list)

    # # split by merge area
    # gr_id = 0
    # while gr_id < len(jump_list):
    #     for ele_id in range(len(jump_list[gr_id]) - 1):
    #         if abs(midpoint_list[jump_list[gr_id][ele_id]][0] - midpoint_list[jump_list[gr_id][ele_id + 1]][0]) > \
    #                 0.7 * (width_list[jump_list[gr_id][ele_id]] + width_list[jump_list[gr_id][ele_id + 1]]) \
    #                 or abs(
    #             midpoint_list[jump_list[gr_id][ele_id]][1] - midpoint_list[jump_list[gr_id][ele_id + 1]][1]) > \
    #                 0.7 * (height_list[jump_list[gr_id][ele_id]] + height_list[jump_list[gr_id][ele_id + 1]]) \
    #                 or area_merge(point_list_1[jump_list[gr_id][ele_id]], point_list_1[jump_list[gr_id][ele_id + 1]]) > \
    #                 2.5 * (area_list[jump_list[gr_id][ele_id]] + area_list[jump_list[gr_id][ele_id + 1]]):
    #             if len(jump_list[gr_id]) - 1 - ele_id > 1:
    #                 jump_list.append(jump_list[gr_id][ele_id + 1:len(jump_list[gr_id])])
    #             del jump_list[gr_id][ele_id + 1:len(jump_list[gr_id])]
    #             if ele_id == 0:
    #                 del jump_list[gr_id]
    #                 gr_id -= 1
    #             break
    #     gr_id += 1

    color = (0, 255, 0)

    # get points of prior_ans
    for group in prior_ans:
        newLabel = ''
        minx = 1e10
        maxx = 0
        miny = 1e10
        maxy = 0
        for ele in group:
            newLabel += label_list[ele]
            if point_list[ele][0][0] < minx:
                minx = point_list[ele][0][0]
            if point_list[ele][2][0] > maxx:
                maxx = point_list[ele][2][0]
            if point_list[ele][0][1] < miny:
                miny = point_list[ele][0][1]
            if point_list[ele][2][1] > maxy:
                maxy = point_list[ele][2][1]
        point1 = [minx, miny]
        point2 = [maxx, miny]
        point3 = [maxx, maxy]
        point4 = [minx, maxy]
        output_txt += (imgNum_list[nowIndex] + ',' + str(point1[0]) + ',' + str(point1[1]) + ',' + str(
            point2[0]) + ',' + str(point2[1]) + ',' + str(point3[0]) + ',' + str(point3[1]) + ',' + str(
            point4[0]) + ',' + str(point4[1]) + ',' + newLabel + '\n')
        image = cv2.line(image, (point1[0], point1[1]), (point2[0], point2[1]),
                         color, thickness)
        image = cv2.line(image, (point3[0], point3[1]), (point2[0], point2[1]),
                         color, thickness)
        image = cv2.line(image, (point3[0], point3[1]), (point4[0], point4[1]),
                         color, thickness)
        image = cv2.line(image, (point1[0], point1[1]), (point4[0], point4[1]),
                         color, thickness)

    # get points of merge
    for group in jump_list:
        newLabel = ''
        for ele in group:
            newLabel += label_list[ele]
        if abs(midpoint_list[group[0]][0] - midpoint_list[group[1]][0]) < 1e-10:
            theta = math.pi / 2 - 1e-10
        else:
            theta = math.atan((midpoint_list[group[0]][1] - midpoint_list[group[1]][1]) / (
                        midpoint_list[group[0]][0] - midpoint_list[group[1]][0]))
        # if -math.pi / 60 < theta < math.pi / 60:
        #     minx = 1e10
        #     maxx = 0
        #     miny = 1e10
        #     maxy = 0
        #     for ele in group:
        #         if point_list_1[ele][0][0] < minx:
        #             minx = point_list_1[ele][0][0]
        #         if point_list_1[ele][2][0] > maxx:
        #             maxx = point_list_1[ele][2][0]
        #         if point_list_1[ele][0][1] < miny:
        #             miny = point_list_1[ele][0][1]
        #         if point_list_1[ele][2][1] > maxy:
        #             maxy = point_list_1[ele][2][1]
        #     point1 = [minx, miny]
        #     point2 = [maxx, miny]
        #     point3 = [maxx, maxy]
        #     point4 = [minx, maxy]
        if -math.pi / 3 < theta < math.pi / 3:
            point1 = point_list[group[0]][0]
            point2 = point_list[group[-1]][1]
            point3 = point_list[group[-1]][2]
            point4 = point_list[group[0]][3]
        else:
            point1 = point_list[group[0]][0]
            point2 = point_list[group[0]][1]
            point3 = point_list[group[-1]][2]
            point4 = point_list[group[-1]][3]
        output_txt += (imgNum_list[nowIndex] + ',' + str(point1[0]) + ',' + str(point1[1]) + ',' + str(
            point2[0]) + ',' + str(point2[1]) + ',' + str(point3[0]) + ',' + str(point3[1]) + ',' + str(
            point4[0]) + ',' + str(point4[1]) + ',' + newLabel + '\n')
        image = cv2.line(image, (point1[0], point1[1]), (point2[0], point2[1]),
                         color, thickness)
        image = cv2.line(image, (point3[0], point3[1]), (point2[0], point2[1]),
                         color, thickness)
        image = cv2.line(image, (point3[0], point3[1]), (point4[0], point4[1]),
                         color, thickness)
        image = cv2.line(image, (point1[0], point1[1]), (point4[0], point4[1]),
                         color, thickness)

    for group in prior_ans:
        jump_list.append(group)
    color = (255, 0, 255)
    # get points of not in prior_ans, merge
    for i in range(nowIndex, endIndex):
        if not find2d(i, jump_list):  # i is not in graph_list
            output_txt += (imgNum_list[i] + ',' + str(point_list[i][0][0]) + ',' + str(point_list[i][0][1]) + ',' + str(
                point_list[i][1][0]) + ',' + str(point_list[i][1][1]) + ',' + str(point_list[i][2][0]) + ',' + str(
                point_list[i][2][1]) + ',' + str(point_list[i][3][0]) + ',' + str(point_list[i][3][1]) + ',' +
                           label_list[i] + '\n')
            image = cv2.line(image, (point_list[i][0][0], point_list[i][0][1]),
                             (point_list[i][1][0], point_list[i][1][1]), color, thickness)
            image = cv2.line(image, (point_list[i][2][0], point_list[i][2][1]),
                             (point_list[i][1][0], point_list[i][1][1]), color, thickness)
            image = cv2.line(image, (point_list[i][2][0], point_list[i][2][1]),
                             (point_list[i][3][0], point_list[i][3][1]), color, thickness)
            image = cv2.line(image, (point_list[i][3][0], point_list[i][3][1]),
                             (point_list[i][0][0], point_list[i][0][1]), color, thickness)
            # print(point_list_1[i], label_list[i])

    # cv2.namedWindow('img', 0)
    # cv2.imshow('img', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # input()

    nowIndex = endIndex

with open(csv_path_out, 'w', encoding='utf-8') as f:
    f.write(output_txt)
    f.close()
