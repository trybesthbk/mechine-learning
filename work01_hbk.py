print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets

# import some data to play with
iris = datasets.load_iris()   #加载鸢尾花数据集
X = iris.data[:, :2]  #取数据前两列进行分类
Y = iris.target

#打印生成的数据集便于观察
print(X)
print(Y)

#设置核函数，下述方法等价于构造了一个线性核函数
def my_kernel(X, Y):
    """
    We create a custom kernel:

                 (1  0)
    k(X, Y) = X  (    ) Y.T
                 (0  1)
    """
    M = np.array([[1, 0], [0, 1.0]])
    return np.dot(np.dot(X, M), Y.T)


h = .02  # 设置网格步长

# 我们创建一个基于线性核函数的支持向量分类SVM并拟合数据。
clf=svm.SVC(kernel=my_kernel)  #kernel='linear'
# clf=svm.SVC(kernel="poly")   #多项式基SVC
# clf=svm.SVC(gamma=1)         #径向基SVC,gamma=1
# clf=svm.LinearSVC()                #径向基SVC
# clf=svm.NuSVC(kernel="linear")    #线性基NuSVC
# clf=svm.NuSVC(kernel="poly",gamma=1)   #多项式基NuSVC,gamma=1
# clf=svm.NuSVC()               #径向基NuSVC

clf.fit(X, Y)
#以上工作以完成支持向量机求解
#以下工作配置图样便于用于理解

# 绘制图像边界并将参数网格化
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1    #设置横轴范围
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1    #设置纵轴范围
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# 填充色彩
Z = Z.reshape(xx.shape)
plt.pcolormesh(xx, yy, Z)#, cmap=plt.cm.Paired
#色彩填充网格参考xx,yy，以Z值区分

# 配置图像信息
plt.scatter(X[:, 0], X[:, 1], c=Y,edgecolors='k')#,cmap=plt.cm.Paired
#将X第一列作为横轴，第二列作为纵轴，颜色由Y决定
plt.title('3-Class classification using Support Vector Machine with custom'
          ' kernel')
#设置标题
plt.axis('tight')
#坐标轴使用数据量
plt.show()
