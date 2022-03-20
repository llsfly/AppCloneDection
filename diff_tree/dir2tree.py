# -*- coding: utf-8 -*-

import datetime
import os

def dfs(fileOrdir, path, tabcount) :
    global tree

    path += "/" + fileOrdir + "/"
    content_lists = os.listdir(path)
    # print(content_lists)

    for i in range(0,tabcount) :
        tree += "\t"

    if len(content_lists) == 0 :
        tree += ".addkid(WeirdNode(" + "\"" + fileOrdir + "\"" + "))\n"
        return

    tree += ".addkid(WeirdNode(" + "\"" + fileOrdir + "\"" + ")\n"

    for cl in content_lists :
        if os.path.isfile(path + cl) :
            for i in range(0, tabcount+1):
                tree += "\t"
            tree += ".addkid(WeirdNode(" + "\"" + cl + "\"" + "))\n"
            continue

        dfs(cl, path, tabcount+1)

    for i in range(0,tabcount) :
        tree += "\t"

    tree += ")\n"

    return

tree = ""
if __name__ == '__main__' :
    start = datetime.datetime.now()
    pathdir = "apk的文件夹目录"
    pathtxt = "输出目录"
    content_apk_list = os.listdir(pathdir)

    print(len(content_apk_list))


    tabcount = 1
    count = 0
    for cal in content_apk_list :
        path = pathdir + cal + "/sources/"

        if not os.path.exists(path) :
            no_sources = open('./nosources.txt','a')
            no_sources.write(cal+'\n')
            no_sources.close()
            continue

        content_list = os.listdir(path)
        count += 1
        name = cal + ".txt"
        tree = ""
        tree += "(" + "WeirdNode(" + "\"sources\")\n"



        for cl in content_list:
            if os.path.isfile(path + cl):
                tree += "\t.addkid(WeirdNode(" + "\"" + cl + "\"" + "))\n"
                continue

            dfs(cl, path, tabcount)

        tree += ")"

        txtwrite = open(pathtxt+name,'a')
        txtwrite.write(tree)
        txtwrite.close()

        print("count = " + str(count))
        print()

    end = datetime.datetime.now()
    time = end - start
    time_consume = "time consume is " + str(time)
    time_consume_txt = open('./dir2tree_timeconsume.txt', 'a')
    time_consume_txt.write(time)
    time_consume_txt.close()
    print("time consume is : " + str(time_consume))
