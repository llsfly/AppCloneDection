# -*- coding: utf-8 -*-

import datetime
import os
import shutil

def two2pair(src_path, target_path, target_path_child, file) :
    """

    :param src_path:  源文件目录
    :param target_path: 目标文件目录，源文件将被移动到该目录下
    :return: 移动成功与否
    """
    if not os.path.exists(target_path) :
        os.mkdir(target_path)

    if not os.path.exists(target_path_child) :
        os.mkdir(target_path_child)

    src_file = src_path + file
    target_file = target_path_child + file

    shutil.copy(src_file, target_file)

def nopairs(src_path, target_path, file) :

    if not os.path.exists(target_path) :
        os.mkdir(target_path)

    src_file = src_path + file
    target_file = target_path + file

    shutil.copy(src_file, target_file)

if __name__ == '__main__' :
    start = datetime.datetime.now()

    path = '***'

    out_path = '*****'
    dir_list = os.listdir(path)

    count_pairs = 0
    count_nopairs = 0
    for dir in dir_list :
        source_path = path + dir
        print(dir)
        if not os.path.isdir(source_path) :
            print('this is .DS_Store')
            continue

        dir_str = dir.split('_')
        dir1 = dir_str[0]
        dir2 = dir_str[1]

        txt_list = os.listdir(source_path)

        dir1_list = []
        dir2_list = []

        for txt in txt_list :
            if dir1 in txt :
                dir1_list.append(txt)
            else :
                dir2_list.append(txt)

        if '.DS_Store' in dir1_list :
            dir1_list.remove('.DS_Store')
        if '.DS_Store' in dir2_list :
            dir2_list.remove('.DS_Store')

        src_path = path + dir + '/'
        target_path = out_path + dir + '/'

        # count = 0
        dir1_temp = dir1_list.copy()
        dir2_temp = dir2_list.copy()
        for dir1_name in dir1_list :
            # count += 1
            name = dir1_name[64:]

            temp_name = name.split('.')[0]

            dir2_name = dir2 + name

            target_path_child = out_path + dir + '/' + temp_name + '/'

            if dir2_name in dir2_list :
                two2pair(src_path, target_path, target_path_child, dir1_name)
                two2pair(src_path, target_path, target_path_child, dir2_name)

                dir1_temp.remove(dir1_name)
                dir2_temp.remove(dir2_name)

                count_pairs += 1


        for dir1_name in dir1_temp:
            nopairs(src_path, target_path, dir1_name)
            count_nopairs += 1

        for dir2_name in dir2_temp :
            nopairs(src_path, target_path, dir2_name)
            count_nopairs += 1


    end = datetime.datetime.now()
    time = end - start
    time_consume = "一共得到的配对 对数为：" + str(count_pairs) + '\n' + "一共得到未配对的txt数量为：" + str(count_nopairs) + '\n' + "所花的时间为：" + str(time) + '\n'
    dir2pairs_txt = open('./pairsinone_timeconsume.txt', 'a')
    dir2pairs_txt.write(time_consume)
    dir2pairs_txt.close()