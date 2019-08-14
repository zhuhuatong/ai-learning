import numpy as np

def one_old(myTree,data,labels):
    if type(myTree) == int:
        return myTree
    print("---------------1")
    print(myTree)
    key = list(myTree.keys())[0]
    keys = key.split('<=')
    print("---------------2")
    if data[labels == keys[0]] <= float(keys[1]):
        return one(myTree[list(myTree.keys())[0]][1],data,labels)
    else:
        return one(myTree[list(myTree.keys())[0]][0],data,labels)



{'petal width (cm)': {
                        0.2: 0,
                        1.2: 1,
                        1.8: 2,
                        1.4: {'sepal length (cm)': {
                                                    4.9: 2,
                                                    5.2: 1,
                                                    6.0: 1,
                                                    6.7:{'petal length (cm)': {
                                                                        5.0: 1,
                                                                        4.0: 1,
                                                                        4.5: 1,
                                                                        5.6: 2}
                                                           },
                                                    6.3: 1,
                                                    5.6: 1
                                                    }
                               },
                        0.4: {'sepal width (cm)': {
                                                    3.2: 0,
                                                    2.0: 1,
                                                    3.5: 0,
                                                    2.6: 1
                                                   }
                             },
                        2.1: 2,
                        0.1: 0
                     }
}


def one(myTree,data,labels):
    ''' 这里输入的data 是 类型为 numpy.ndarray 的一行数据（4个特征） '''
    #print(myTree)
    #keys = myTree.keys()
    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    for (lab, sub_tree) in myTree.items():  # lab 是特征（列）名称，sub_tree是子节点也是一个树.
        if type(sub_tree)!=type({}):  # 最佳值不是字典，说明已经是叶子节点了. 返回匹配值
            return sub_tree
        axis = None
        for  kk in  range(len(labels)):  # 本循环,获取本节点的label对应的列序号
            if labels[kk]==lab:
                axis = kk
        # 下面将本列特征的值与 子节点树sub_tree 的key做最佳匹配(相减后绝对值最小者最佳)
        best_key = -1
        best_val = -1
        best_distinct = 9999999999
        for (k,v) in sub_tree.items():
            print(axis,data[axis])
            dist = np.abs(float(data[axis])-float(k))
            if dist<best_distinct:  # 取绝对值最小者.
                best_key = k
                best_distinct = dist
                best_val = v
        if type(best_val)!=type({}):  # 最佳值不是字典，说明已经是叶子节点了. 返回匹配值
            return best_val
        else:
            return one(sub_tree,data,labels)


def getResult(myTree,data,labels):
    result = []
    print("myTree:-------------------------")
    print(myTree)
    print("================================")
    for ii,elem in enumerate(data):
        print("[%s]elem:%s" % (labels,elem))
        res = one(myTree,elem,labels)
        print("elem-result:%s" % res)
        result.append(res)
    return result

def yanzheng(myTree,data,labels,target):
    count = 0
    result = getResult(myTree,data,labels)
    for i in range(len(result)):
        if(result[i] == target[i]):
            count += 1
    return count

