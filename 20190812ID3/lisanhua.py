import numpy as np


def lsh(data, num):
    a = []
    for i in range(len(data[0])):  # 获取 data 中的总列数, 循环每一列
        b = []
        data0 = data[:, i]
        data1 = [i for i in data0]
        l = len(data1)
        data1.sort()  # .
        for k in range(num):  # 对排序后的列数据 ,从位置0开始，每隔num个数将其值取过来 保存到 b
            b.append(data1[int(k * l / num)])  # 0,150*1/7,150*2/7,...150*6/7
        for j in range(len(data)): # 获取总行数，循环每一行.
            if data[j, i] >= b[-1]:  # 当前列中、当前行定位的数据值，凡是大于 b 中最大值的，都将其用b的最大值替换.
                data[j, i] = b[-1]
                continue
            for q in range(1, num):
                if data[j, i] < b[q] and data[j, i] >= b[q - 1]:  # 当前列中、当前行定位的数据值，若落在list的b中的某两个值之间，将该值两个值的较小值替换.
                    print("%d,%d:%s --> %s" % (j,i,data[j, i], b[q - 1]))
                    data[j, i] = b[q - 1]
        a.append(b)
    return data, a


if __name__=="__main__":
    pass