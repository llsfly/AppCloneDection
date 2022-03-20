# -*- coding:utf-8 -*-
import jieba
import jieba.analyse
import numpy as np
import re

txt1 = r'E:\\test1.txt'
txt2 = r'E:\\test2.txt'

class simhash:
    # 构造函数
    def __init__(self, content):
        self.hash = self.simhash(content)

    def __str__(self):
        return str(self.hash)

    # 生成simhash值
    def simhash(self, content):
        count = 0
        seg = jieba.cut(content)
        # jieba基于TF-IDF提取前10位关键词
        keyWords = jieba.analyse.extract_tags("|".join(seg), topK=10, withWeight=True, allowPOS=())

        keyList = []
        # 获取每个词的权重
        for feature, weight in keyWords:
            #print('feature:{},weight: {}'.format(feature, weight))
            # 每个关键词的权重*总单词数
            weight = int(weight * 10)
            #生成普通的的hash值
            binstr = self.string_hash(feature)
            #打印指纹大小
            if(count == 0):
                print("指纹大小为:", len(binstr))
                count += 1
            temp = []
            for c in binstr:
                if (c == '1'):# 查看当前bit位是否为1,是的话将weight*1加入temp[]
                    temp.append(weight)
                else:#否则的话，将weight*-1加入temp[]
                    temp.append(-weight)
            keyList.append(temp)
        # 将每个关键词的权重变成一维矩阵
        listSum = np.sum(np.array(keyList), axis=0)
        if (keyList == []):#编码读不出来
            return '00'
        simhash = ''
        for i in listSum:
            if (i > 0):
                simhash = simhash + '1'
            else:
                simhash = simhash + '0'
        return simhash# 整个文档的fingerprint为最终各个位>=0的和

    # 求海明距离
    def hamming_distance(self, other):
        t1 = '0b' + self.hash
        t2 = '0b' + other.hash
        n = int(t1, 2) ^ int(t2, 2)
        i = 0
        while n:
            n &= (n - 1)
            i += 1
        return i

    #计算相似度
    def similarity(self, other):
        a = float(self.hash)
        b = float(other.hash)
        print(a, b)
        if a > b:
            return b / a
        #elif a == 0.0 and b == 0.0:
        #return 1
        else:
            return a / b

# 针对source生成hash值   (一个可变长度版本的Python的内置散列)
    def string_hash(self, source):
        if source == "":
            return 0
        else:
            # 将字符转为二进制，并向左移动7位
            x = ord(source[0]) << 7
            m = 1000003
            mask = 2 ** 128 - 1
            # 拼接每个关键词中字符的特征
            for c in source:
                x = ((x * m) ^ ord(c)) & mask
            x ^= len(source)
            if x == -1:
                x = -2
            #通过改变.zfill(16)[-16:]来实现改变指纹大小
            x = bin(x).replace('0b', '').zfill(32)[-32:]
            #print('strint_hash: %s, %s' % (source, x))
            return str(x)

def txt_line(txt1, txt2):
    punc = './ <>_ - - = ", 。，？！“”：‘’@#￥% … &×（）——+【】{};；● &～| \s:'
    #获取文本中的数据
    with open(txt1, 'r', encoding='utf-8') as f:
        list1 = f.read()
        string = ''
        text1 = re.sub(r'[^\w]+', '', list1)
        s = jieba.cut(text1)
        string = string.join(s)
        line1 = re.sub(r"[{}]+".format(punc), "", string)

    with open(txt2, 'r', encoding='utf-8') as f:
        list2 = f.read()
        string = ''
        text2 = re.sub(r'[^\w]+', '', list2)
        s = jieba.cut(text2)
        string = string.join(s)
        line2 = re.sub(r"[{}]+".format(punc), "", string)
        hash1 = simhash(line1)
        hash2 = simhash(line2)
        print("海明距离:", hash1.hamming_distance(hash2))
        print("文本相似度:", hash1.similarity(hash2))

if __name__ == '__main__':
    txt_line(txt1, txt2)

