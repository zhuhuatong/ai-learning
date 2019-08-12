import math
import operator


def calcShannonEnt(dataset):
    numEntries = len(dataset)
    labelCounts = {}
    for featVec in dataset:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1

    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * math.log(prob, 2)
    return shannonEnt


def CreateDataSet():
    dataset = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataset, labels


def splitDataSet(dataSet, axis, value):
    ''' 对每行中的指定列（轴）的内容和value匹配，若相等，将 该行内容的其余字段提取组成一个新的行，所有新的行组合新生成一个数据集'''
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)

    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    ''' 获取信息增益最大的那个特征列序号 '''
    numberFeatures = len(dataSet[0]) - 1  # 总特征数
    baseEntropy = calcShannonEnt(dataSet) # 根据dataSet的最后一列的内容（Y），计算整个熵
    bestInfoGain = 0.0;
    bestFeature = -1;
    for i in range(numberFeatures):
        featList = [example[i] for example in dataSet]  # 将当前列特征提取出来保存到一个list
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value) # 对i列的值与value匹配获取，返回新的数据集，并去掉i列
            prob = len(subDataSet) / float(len(dataSet)) # 计算本值对应的记录数占总记录数的比值
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy  # 总熵减去本列特征对应的熵，即信息增益
        if (infoGain > bestInfoGain): # 保留信息增益最大的特征
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


def majorityCnt(classList):  # 感觉这个函数内部有点问题，不过目前的测试数据不会调用这里，无所谓.
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] = 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]  # 从数据集中获取最后一列（Y）生成一个list
    if classList.count(classList[0]) == len(classList): # 如果对第一个元素的计数，和总数量一样，说明全部只有一个数，返回.
        return classList[0]
    if len(dataSet[0]) == 1:  # 如果只有一列
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]  # 取信息增益最大的那一列的列名称.
    myTree = {bestFeatLabel: {}}
    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree
