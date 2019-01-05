'''
Ｓcikit-learn是一个开源的机器学习工具包，集成了各种常用的机器学习算法和预处理工具。
使用scikit-learn的ｋｎｎ算法进行分类的一个实例，使用的数据集依然是ｉｒｉｓ
'''
from sklearn.datasets import load_iris
from sklearn import neighbors
import sklearn

#查看ｉｒｉｓ数据集
iris = load_iris()
# print(iris)

knn = neighbors.KNeighborsClassifier()
#训练数据集
knn.fit(iris.data, iris.target)

#预测
predict = knn.predict([[7.8,3.0,5.1,2.1]])
print(predict)
print((iris.target_names[predict]))