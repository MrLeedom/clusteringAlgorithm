from sklearn.cross_validation import train_test_split

all_inputs = df[['sepal_len','sepal_width','petal_len','petal_width']].values
all_classes = df['class'].values

(X_train,X_test,X_train,Y_test) = train_test_split(all_inputs,all_classes,train_size=0.8,random_state=1)

#使用决策树算法进行训练
from sklearn.tree import DecisionTreeClassifier

#定义一个决策树对象
decision_tree_classifier = DecisionTreeClassifier()

#训练模型
model = decision_tree_classifier.fit(training_inputs,training_classes)

#所得模型的准确性
print(decision_tree_classifier.score(testing_inputs,testing_classes))

#使用训练的模型进行预测，直接把测试集里面的数据拿出来三条
print(X_test[0:3])
print(Y_test[0:3])
print(model.predict(X_test[0:3]))