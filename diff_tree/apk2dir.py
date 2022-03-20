# -*- coding: utf-8 -*-

import os
import datetime

start = datetime.datetime.now()

pathapk = "apk的目录"
pathdir = "输出目录"


count = 0
for i in range(0,16) :
    hex = str(i)
    if i >= 10 :
        i += 55
        hex = chr(i)

    pathapk_hex = pathapk + hex + '/'
    apk_list = os.listdir(pathapk_hex)
    for al in apk_list :
        count += 1
        dirname = al.split(".")[0]
        try:
            os.system("jadx目录 " + pathapk_hex + al + " -d " + pathdir + dirname)
        except Exception as e:
            wrong = open("apk2dir-wrong.txt", 'a')
            wrong.write(al + '\n')
            wrong.close()

        print("count : " + str(count))
        print()

    # print("the total count : " + str(count))

end = datetime.datetime.now()

time = end - start
time_consume = "总共得到文件夹数量为：" + str(count) + "\n所用耗时为：" + str(time) + "\n"
time_consume_txt = open('./apk2dir_timeconsume.txt', 'a')
time_consume_txt.write(time_consume)
time_consume_txt.close()
