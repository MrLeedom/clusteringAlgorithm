from numpy import *
import operator

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A', 'A', 'B','B']
    return group,labels

def classify(inX,dataSet,labels,k):
    #返回矩阵的行数，如果shape[1]返回的则是数组的列数
    dataSetSize = dataSet.shape[0]
    #两个矩阵相减，得到新的矩阵
    diffMat = tile(inX,(dataSetSize,1)) - dataSet
    #求平方
    sqDiffMat = diffMat**2
    #求和，返回的是一维数组
    sqDistances = sqDiffMat.sum(axis=1)
    #开方，即每个测试点到其余各个点的距离
    distances = sqDistances**0.5
    #排序，返回值是原数组从小到大排序的下标值
    sortedDistIndices = distances.argsort()
    #定义一个空的字典
    classCount = {}
    for i in range(k):
        #返回距离最近的ｋ个点所对应的标签值
        voteIlabel = labels[sortedDistIndices[i]]
        #存放到字典中,python字典中的ｇｅｔ方法返回函数指定键的值，如果值不在字典中，返回第二个参数默认值
        #将指定类标签的值累加，后面可以根据值的大小排序，也就会得到一个投票值，这相当于一个比较好的效果
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
    #排序classCount.items(),输出键值对ｋｅｙ代表排序的关键字，第二个域，Ｔｒｕｅ代表降序
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse = True)
    #返回距离最近的点对应的标签
    return sortedClassCount[0][0]

def main():
    group, labels = createDataSet()
    print(classify([0,0],group,labels,3))

if __name__ == '__main__':
    main()