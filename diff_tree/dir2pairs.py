# -*- coding: utf-8 -*-

import datetime
import os
import shutil

def read_apk2dir(path) :
    """

    :param path: apk2dir 地址
    :return: 该地址下所有的文件夹名称列表
    """
    apk2dir_list = os.listdir(path)
    return apk2dir_list

def read_repackage_pairs(path) :
    """
    读取 配对 文件
    :param path: pairs的文件路径
    :return:  文件内容列表
    """
    repackage_pairs = open(path)
    content_repackage_pairs = repackage_pairs.readlines()
    return content_repackage_pairs[1:]

def dir2other(src_path, target_path) :
    """

    :param src_path:  源文件目录
    :param target_path: 目标文件目录，源文件将被移动到该目录下
    :return: 移动成功与否
    """
    if not os.path.exists(target_path) :
        os.mkdir(target_path)

    # 判断是否是目录
    if os.path.isdir(src_path) and os.path.isdir(target_path) :
        # 源目录下文件及目录列表
        filelist_src = os.listdir(src_path)
        for file in filelist_src :
            # 将该 file 添加入路径中
            path = os.path.join(os.path.abspath(src_path), file)
            # 如果是目录，递归查找
            if os.path.isdir(path) :
                path1 = os.path.join(os.path.abspath(target_path), file)
                if not os.path.exists(path1) :
                    os.mkdir(path1)
                dir2other(path, path1)
            # 如果是文件，读取文件内容，并写入到新的目录下
            else :
                with open(path, 'rb') as read_stream :
                    contents = read_stream.read()
                    path1 = os.path.join(target_path, file)
                    with open(path1, 'wb') as write_stream :
                        write_stream.write(contents)

        return True
    return False

def pairs2one(src_path, target_path, name) :
    """

    :param src_path: 源文件路径
    :param target_path: 目标文件路径，源文件下的所有 .java 文件将被重新命名，并写入到该路径下
    :param name: 源文件根目录名称
    :return: 写入成功与否
    """
    # 统计共有多少个 .java 文件
    file_count = 0
    source_path = os.path.abspath(src_path)
    target_path = os.path.abspath(target_path)

    # 如果不存在 目标路径 则先创建
    if not os.path.exists(target_path) :
        os.mkdir(target_path)


    if os.path.exists(source_path) :
        # 使用 os.walk() 函数，获取源文件路径下，所有同级子目录、子文件，并保留该同级文件的上一次路径：root
        for root, dirs, files in os.walk(source_path) :
            temp_name = name
            # 分割 root，获取字符串
            root_str = root.split('/')
            idx = -1
            temp = ''
            # 从右往左遍历root_str字符串，直到sources
            while root_str[idx] != 'sources' :
                temp = root_str[idx] + temp
                idx -= 1

            temp_name += temp

            for file in files :
                if '.java' not in file:
                    continue

                file_name = file.split('.')[0] + '.txt'

                src_file = os.path.join(root, file)
                # 得到新的文件名称
                target_file = target_path + '/' + temp_name + file_name
                shutil.copy(src_file, target_file)
                file_count += 1
        return file_count

if __name__ == '__main__' :
    start = datetime.datetime.now()

    apk2dir_path = 'apk的文件夹目录'
    apk2dir_list = read_apk2dir(apk2dir_path)
    print(len(apk2dir_list))

    repackage_pairs_path = './repackaging_pairs.txt'
    pairs_list = read_repackage_pairs(repackage_pairs_path)
    print(len(pairs_list))

    src_path = 'apk的文件夹目录'
    target_path = '目标目录'

    count_pairs = 0

    for pair in pairs_list :
        pair_temp = pair.split('\n')[0]
        pair_str = pair_temp.split(',')

        # 获得两个文件的名称
        pair1 = pair_str[0]
        pair2 = pair_str[1]

        # 两个源文件目录
        src_path_dir1 = src_path + pair1
        src_path_dir2 = src_path + pair2

        # print(src_path_dir1)
        # print(src_path_dir2)
        print(pair)

        # 如果两个文件夹，有一个文件夹不存在，则不选择这一对
        if pair1 not in apk2dir_list or pair2 not in apk2dir_list:
            print('======================================')
            print(pair_str)
            print('======================================')
            continue

        # 两个源文件+sources的目录
        src_path_dir_source1 = src_path_dir1 + '/' + 'sources'
        src_path_dir_source2 = src_path_dir2 + '/' + 'sources'

        # 如果两个文件夹下，有一个文件夹不存在sources目录，则不选择这一对
        if not os.path.exists(src_path_dir_source1) or not os.path.exists(src_path_dir_source2) :
            continue

        target_path_dir1 = target_path + pair1
        target_path_dir2 = target_path + pair2

        # if dir2other(src_path_dir1, target_path_dir1) and dir2other(src_path_dir2, target_path_dir2) :
        #     count_pairs += 1

        count_pairs += 1

        # 两个文件中的所有java文件，写入同一个目录下
        target_path_java = 'apk的文件夹目录' + pair1 + '_' +pair2 + '/'

        # 统计两个文件夹下的java数量
        count_java1 = pairs2one(src_path_dir_source1, target_path_java, pair1)
        count_java2 = pairs2one(src_path_dir_source2, target_path_java, pair2)

        # 将文件名，以及对应的java文件数量，写入文件
        count_java_write = pair + str(count_java1) + ',' + str(count_java2) + '\n'
        file_count = open('./file_count.txt', 'a')
        file_count.write(count_java_write)
        file_count.close()


    end = datetime.datetime.now()

    time = end - start
    time_consume = "一共得到的配对数量为：" + str(count_pairs) + '\n' + "所花的时间为：" + str(time) + "\n"
    dir2pairs_txt = open('./dir2pairs_timeconsume.txt', 'a')
    dir2pairs_txt.write(time_consume)
    dir2pairs_txt.close()