
'''
集成学习分类
Author: james
Date: 2020-11-20 09:32:17
LastEditTime: 2020-11-20 10:10:40
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /PyCode/project_demo/Datafountain/Learn/集成学习/test1.py
'''

'''
其中，Iris鸢尾花数据集是一个经典数据集，在统计学习和机器学习领域都经常被用作
示例。数据集内包含3类共150条记录，每类各50个数据，每条记录都有 4项特征：
花萼长度、花萼宽度、花瓣长度、花瓣宽度，可以通过这4个特征预测鸢尾花卉属于（
iris-setosa, iris-versicolour, iris-virginica）
三个类别中的哪一品种。

'''
from sklearn.datasets import load_iris # 导入数据集
iris = load_iris() # 载入数据集
print('iris的所有数据')
print(iris.data)
print('iris数据集特征')
print(iris.data[:10])

print('iris数据集标签')
print(iris.target[:10])


from sklearn.ensemble import AdaBoostClassifier # 导入adaboost
clf = AdaBoostClassifier(n_estimators = 5)

clf.fit(iris.data[:120],iris.target[:120]) # 模型训练,这里取0.8比例的数据集作为训练集合对加载进来的数据进行训练

# 剩下0.2用来做预测集，使用训练好的决策树模型对其进行预测
predictions = clf.predict(iris.data[120:])
predictions[:10]

# 评估结果正确的数量占样本总数
from sklearn.metrics import accuracy_score # 准确率评价指标
print('Accuracy:%s'% accuracy_score(iris.target[120:],predictions))

# todo Adaboost参数调整 (n_estimators 代表基分类器数量)
