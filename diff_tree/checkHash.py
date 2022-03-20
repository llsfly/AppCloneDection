# -*- coding: utf-8 -*-

import hashlib
import os
import shutil
import datetime

def get_file_md5(filepath) :
    m = hashlib.md5()
    f = open(filepath, 'rb')
    # 读小文件时用
    # m.update(f.read())

    # 读大文件时用
    while True:
        # 如果不用二进制打开文件，则需要先编码
        # data = f.read(1024).encode('utf-8')
        data = f.read(1024)  # 将文件分块读取
        if not data:
            break
        m.update(data)
    hash = m.hexdigest()
    return hash

def outpath(src_path, target_path, target_path_child, file) :
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

if __name__ == '__main__' :
    begin = datetime.datetime.now()
    path = '输出目录'
    path_list = os.listdir(path)

    out_same = '哈希相同目录'
    out_diff = '哈希不同目录'

    count = 0
    count_same = 0
    count_diff = 0
    for pl in path_list :
        path_pairs = path + pl + '/'
        if not os.path.isdir(path_pairs) :
            continue

        pl_str = pl.split('_')
        dir1 = pl_str[0]
        dir2 = pl_str[1]

        if not os.path.isdir(path_pairs) :
            continue

        path_pairs_list = os.listdir(path_pairs)

        for ppl in path_pairs_list :
            path_pairs_child = path_pairs + ppl + '/'

            if not os.path.isdir(path_pairs_child) :
                continue

            path_pairs_child_list = os.listdir(path_pairs_child)

            dir1_list = []
            dir2_list = []

            for ppcl in path_pairs_child_list :
                if dir1 in ppcl :
                    dir1_list.append(ppcl)

                if dir2 in ppcl :
                    dir2_list.append(ppcl)

            for d1l in dir1_list :
                name = d1l.split('.')[0]
                num = name.split('_')[1]

                d2l_temp = dir2 + '_' +num + '.txt'

                if d2l_temp in dir2_list :
                    hash1 = get_file_md5(path_pairs_child + d1l)
                    hash2 = get_file_md5(path_pairs_child + d2l_temp)

                    if hash1 == hash2 :
                        outpath(path_pairs_child, out_same+pl, out_same+pl+'/'+ppl+'/', d1l)
                        outpath(path_pairs_child, out_same+pl, out_same+pl+'/'+ppl+'/', d2l_temp)
                        count_same += 2
                    else :
                        # print('hash1: '+hash1)
                        # print('hash2: '+hash2)
                        # print('d1l: ' + d1l)
                        # print('d2l: ' + d2l_temp)
                        outpath(path_pairs_child, out_diff+pl, out_diff+pl+'/'+ppl+'/', d1l)
                        outpath(path_pairs_child, out_diff+pl, out_diff+pl+'/'+ppl+'/', d2l_temp)
                        count_diff += 2

        print(pl)

        count += 1
        if count == 50 :
            break

    print('count: ' + str(count))

    end = datetime.datetime.now()

    time_consume = end - begin
    summary = '一共获得相似txt的个数：'+str(count_same) + ' 个' + '\n' \
              + '一共获得不相似txt的个数：' + str(count_diff)  + ' 个' + '\n' \
              + '总运行时长为：' + str(time_consume) + '\n'
    txt = open('./checkHash.txt','a')
    txt.write(summary)
    txt.close()