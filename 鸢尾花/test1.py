'''
使用python实现的knn算法进行分类的一个实例，使用数据集依然是iris
'''
import csv
import random
import math
import operator
from sklearn import neighbors

#加载数据集
def loadDataset(filename, split, trainingSet=[], testSet=[]):
    #这里有个错误是此csv文件不是二进制文件，而是文本文件
    with open(filename,'rt') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        print(len(dataset))
        for x in range(len(dataset) - 1):
            for y in range(4):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[y])
        # for row in dataset:
            # print(', '.join(row))

#计算距离
def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance +=pow((instance1[x] - instance2[x]),2)
    return math.sqrt(distance)

#返回ｋ个最近邻
def getNeighbors(trainingSet, testInstance, k):
    distance = []
    length = len(testInstance) - 1
    #计算每一个测试实例到训练集实例的距离
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distance.append((trainingSet[x],dist))
    #对所有的距离进行排序
    distance.sort(key=operator.itemgetter(1))
    neighbors = []
    #返回ｋ个最近邻
    for x in range(k):
        neighbors.append(distance[x][0])
    return neighbors

#对ｋ个近邻进行合并，返回ｖａｌｕｅ最大的ｋｅｙ
def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    #排序
    sortedVotes = sorted(classVotes.items(),key=operator.itemgetter(1),reverse=True)
    return sortedVotes[0][0]

#计算准确率
def getAccuracy(testSet,predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0

def main():
    trainingSet = []   #训练数据集
    testSet = []        #测试数据集
    split = 0.67        #分割的比例
    loadDataset(r'iris.data',split,trainingSet,testSet)
    print('Train set:'+repr(len(trainingSet)))
    print('Test set:'+repr(len(testSet)))

    predictions = []
    k = 3
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
        print('>predicted = '+repr(result)+',actual = '+repr(testSet[x][-1]))
    accuacy = getAccuracy(testSet, predictions)
    print('Accuacy:'+repr(accuacy)+'%')

if __name__ =='__main__':
    main()

# loadDataset('iris.data',',',[],[])