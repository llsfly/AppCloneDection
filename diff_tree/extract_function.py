# -*- coding: utf-8 -*-

import datetime
import os

def extract(src_path, target_path, file) :
    global total
    read_file = open(src_path+file,encoding='utf-8')
    content_file = read_file.readlines()

    # print(target_path)
    if not os.path.exists(target_path) :
        os.mkdir(target_path)

    name = file[0:64]

    stack = 0
    result = []
    temp = []
    # print(content_file)
    for content in content_file :
        if 'class' in content :
            continue

        if ('public' in content or 'private' in content or 'protected' in content) and '{' in content:
            count_left = content.count('{')
            stack += count_left
            # print('count_left is ' + str(count_left))
            count_right = content.count('}')
            stack -= count_right
            temp.append(content)

            if stack == 0 :
                print(temp)
                result.append(temp)
                temp = []
        elif stack > 0 :
            if '{' in content :
                count_left = content.count('{')
                stack += count_left
            if '}' in content :
                count_right = content.count('}')
                stack -= count_right
            temp.append(content)

            if stack == 0 :
                print(temp)
                result.append(temp)
                temp = []
        else :
            continue

    count = 0
    for res in result :
        count += 1
        file_name = name + '_' +str(count) + '.txt'
        txt = open(target_path+file_name,'w+')

        for r in res :
            # print(r)
            txt.write(r)

        txt.close()

    total += count
    return

if __name__ == '__main__' :
    total = 0

    start = datetime.datetime.now()
    # 源文件根目录
    source_path = '***'

    # 读取源文件夹目录下，所有配对好的文件夹
    source_dir_list = os.listdir(source_path)

    target = '****'

    for source_dir in source_dir_list :
        # 每个配对的文件夹路径
        source_dir_path = source_path + source_dir + '/'

        if not os.path.isdir(source_dir_path) :
            continue

        # 读取该路径下，所有的文件夹，或文件，存入列表
        dir_list = os.listdir(source_dir_path)

        target_path = target + source_dir + '/'

        if not os.path.exists(target_path) :
            os.mkdir(target_path)

        for dir in dir_list :
            dir_path = source_dir_path + dir + '/'

            target_dir_path = target_path + dir + '/'

            if not os.path.isdir(dir_path) :
                continue

            file_list = os.listdir(dir_path)

            for file in file_list :
                if '.DS_Store' in file :
                    continue

                extract(dir_path, target_dir_path, file)


    end = datetime.datetime.now()
    time = end - start

    time_consume = "所花的时间为：" + str(time) + '\n'
    dir2pairs_txt = open('./extractfunction_timeconsume.txt', 'a')
    dir2pairs_txt.write(time_consume)
    dir2pairs_txt.close()

    print(total)
