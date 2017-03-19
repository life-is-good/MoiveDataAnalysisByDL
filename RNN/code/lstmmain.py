# -*- coding:utf-8 -*-
import lstm
import xlrd
import os
import jieba
path = os.getcwd()
jieba.load_userdict(path+"/../dict/userdict.txt")

#将excel里面的评论转成符合pandas.read_csv()的读取文件要求
def get_excel_data(path):
    #将积极评论写入训练集
    table1 = xlrd.open_workbook(path + "/../seniment review set/GREATEWALLPOS.xls")
    sheet = table1.sheet_by_index(0)
    data = sheet.col_values(0)
    res = []
    for d in data:
        str = ""
        seg = []
        seg = jieba.cut(d)
        for s in seg:
            str = str+" "+s.replace(",","").strip("\r")
        str = str + ",1"
        res.append(str)
    f = open(path+"/../data/my_data/train.txt","a")
    for r in res:
        f.write(r+"\n")
    f.close()
    #将消极评论写入训练集
    table2 = xlrd.open_workbook(path + "/../seniment review set/GREATEWALLNEG.xls")
    sheet = table2.sheet_by_index(0)
    data = sheet.col_values(0)
    res = []
    for d in data:
        str = ""
        seg = []
        seg = jieba.cut(d)
        for s in seg:
            str = str+" "+s.replace(",","").strip("\r")
        str = str + ",0"
        res.append(str)
    f = open(path+"/../data/my_data/train.txt","a")
    for r in res:
        f.write(r+"\n")
    f.close()
    #将积极测试评论写入测试集
    table3 = xlrd.open_workbook(path + "/../seniment review set/GREATEWALLTESTPOS.xls")
    sheet = table3.sheet_by_index(0)
    data = sheet.col_values(0)
    res = []
    for d in data:
        str = ""
        seg = []
        seg = jieba.cut(d)
        for s in seg:
            str = str+" "+s.replace(",","").strip("\r")
        str = str + ",1"
        res.append(str)
    f = open(path+"/../data/my_data/test.txt","a")
    for r in res:
        f.write(r+"\n")
    f.close()

    
if __name__ == '__main__':
    # 可以赋值一些训练参数之类的.
#     train_lstm(
#         max_epochs=1,
#         test_size=500,
#     )
    # test the model
    get_excel_data(path)
    lstm.test_lstm(dataset='my_data')