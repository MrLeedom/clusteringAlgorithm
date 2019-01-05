import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sb
df = pd.read_csv('iris.csv')
df.columns = ['sepal_len','sepal_width','petal_len','petal_width','class']

df['class'] = df['class'].apply(lambda x:x.split('-')[1])
#查看数据信息
df.describe()
# print(df.head())
def scatter_plot_by_category(feat, x, y):
    alpha = 0.5
    gs = df.groupby(feat)
    cs = cm.rainbow(np.linspace(0,1,len(gs)))   #选择颜色，进行调节
    for g,c in zip(gs, cs):             #ｚｉｐ函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组
        plt.scatter(g[1][x],g[1][y],color=c,alpha=alpha)

plt.figure()
# plt.subplot(131)
scatter_plot_by_category('class','sepal_len','petal_len')
plt.xlabel('sepal_len')
plt.ylabel('petal_len')
plt.title('class')

plt.figure(figsize=(20,10))
for column_index,column in enumerate(df.columns):
    if column == 'class':
        continue
    plt.subplot(2,2,column_index+1)
    #小提琴图，它显示了定量数据在一个或多个分类变量的多个层次上的分布，小提琴绘图以基础分布的核密度估计为特征
    sb.violinplot(x='class',y=column,data=df)
plt.show()

from sklearn.model_selection import train_test_split

all_inputs = df[['sepal_len','sepal_width','petal_len','petal_width']].values
all_classes = df['class'].values

(X_train,X_test,Y_train,Y_test) = train_test_split(all_inputs,all_classes,train_size=0.8,random_state=1)

#使用决策树算法进行训练
from sklearn.tree import DecisionTreeClassifier

#定义一个决策树对象
decision_tree_classifier = DecisionTreeClassifier()

#训练模型
model = decision_tree_classifier.fit(X_train,Y_train)

#所得模型的准确性
print(decision_tree_classifier.score(X_test,Y_test))

#使用训练的模型进行预测，直接把测试集里面的数据拿出来三条
print(X_test[0:3])
print(Y_test[0:3])
print(model.predict(X_test[0:3]))
