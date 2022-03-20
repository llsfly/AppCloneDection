# -*- coding: utf-8 -*-

import os

path = '文件目录'
dir_list = os.listdir(path)

outTxt = open('./compareTxt.txt','a')

count_txt = 0
count_pairs = 0
for dir in dir_list :
    path_dir = path + dir + '/'

    if not os.path.isdir(path_dir) :
        continue

    dir_str = dir.split('_')
    dir1 = dir_str[0]
    dir2 = dir_str[1]

    path_dir_list = os.listdir(path_dir)

    for pdl in path_dir_list :
        path_dir_child = path_dir + pdl + '/'

        if not os.path.isdir(path_dir_child) :
            continue

        dir1_list = []
        dir2_list = []

        path_dir_child_list = os.listdir(path_dir_child)

        for pdcl in path_dir_child_list :
            if dir1 in pdcl :
                dir1_list.append(path_dir_child + pdcl)
                count_txt += 1

            if dir2 in pdcl :
                dir2_list.append(path_dir_child + pdcl)
                count_txt += 1

        for d1l in dir1_list :
            for d2l in dir2_list :
                outTxt.write(d1l+','+d2l+'\n')
                count_pairs += 1

outTxt.close()
print('count_txt is ' + str(count_txt))
print('count_pairs is ' + str(count_pairs))