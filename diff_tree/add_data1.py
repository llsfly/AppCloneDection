import codecs

def addData2(in_file,out_file):
    with codecs.open(in_file,encoding='utf8') as in_read:
         for line in in_read.readlines():
             line=line.strip()
             items=line.split()
             for curword in items:
                out_file.write(curword)
             out_file.write(",1")
             out_file.write('\n')
# def addData(in_file,out_file):
#     with codecs.open(in_file,encoding='utf8') as in_read:
#         for line in in_read.readlines():
#             out_file.write("/home/lenglinshan/"+line)
if __name__ == '__main__':
    out_write=open("compare_new.txt", 'w', encoding='utf8')
    addData2("compareTxt.txt",out_write)