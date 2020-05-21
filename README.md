# mechine-learning
&ensp;&ensp;&ensp;&ensp;本文通过支持向量机的理论对鸢尾花数据集分类以及数字图像识别两类经典的案例进行解析，加深对机器学习的理解与应用

## 1.鸢尾花数据集分类
代码如下：
```javascript
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

```
&ensp;&ensp;&ensp;&ensp;首先我们采用机器学习中经典的鸢尾花数据集（Iris），是一类多重变量分析的数据集。Iris数据集是一个150行4列的数据集，我们仅取其前两列分别作为横轴与纵轴，而这150个数据又被均分为三份，预测值分别设置为0，1，2。接下来只要通过sklearn导入的svm成员以及fit成员对数据进行求解并拟合即可完成支持向量机求解工作，这里我们可以看出python自带的包功能相当强大，上述所述的支持向量机求解工作已被极大的简化至仅需要调用相关的成员函数即可接触，而其中在运算则在下图classes.py的文件中实现。  
![内置支持向量机方法](https://github.com/trybesthbk/mechine-learning/blob/master/%E5%9B%BE%E7%89%87/%E5%86%85%E7%BD%AE%E6%94%AF%E6%8C%81%E5%90%91%E9%87%8F%E6%9C%BA%E6%96%B9%E6%B3%95.png)

&ensp;&ensp;&ensp;&ensp;为便于调试，上述工作我们在工作台（console）中完成，这样可以实时查看变量的值，以下为完成数学运算部分后的变量图，验证以上工作均正常完成。  
![相关变量参数](https://github.com/trybesthbk/mechine-learning/blob/master/%E5%9B%BE%E7%89%87/%E7%9B%B8%E5%85%B3%E5%8F%98%E9%87%8F%E5%8F%82%E6%95%B0.png)


&ensp;&ensp;&ensp;&ensp;接下来需要配置图像参数以便我们能够更好的查看分类的情况。我们主要需要对图像进行网格化并用不同的颜色区分不同的预测值以及相应分类的区间，最终以Iris数据集第一列作为横轴，第二列作为纵轴，预测值作为区分变量即可得到以下图像。

![鸢尾花数据集分类结果-线性核SVC](https://github.com/trybesthbk/mechine-learning/blob/master/%E5%9B%BE%E7%89%87/%E9%B8%A2%E5%B0%BE%E8%8A%B1%E6%95%B0%E6%8D%AE%E9%9B%86%E5%88%86%E7%B1%BB%E7%BB%93%E6%9E%9C-%E7%BA%BF%E6%80%A7%E6%A0%B8SVC.png)


&ensp;&ensp;&ensp;&ensp;从图中可以看出，通过线性划分我们很好的将这三类数据进行了分类，大部分点都处于他们应该位于其预测值的分类区间内。但我们也可以看到图中也有许多点进入了错误的区间，也就是第二章节感知机中的损失，显然我们上述的线性SVC方法不能说是最优的，为了确定更好的分类方法我们往往可以通过对多种分类模型进行对比，因此我们通过SVC/NuSVC函数对通过调节它的核函数及其相应参数即可得到不同的分类图像,如下图所示：

![鸢尾花数据集分类结果（多种方法）](https://github.com/trybesthbk/mechine-learning/blob/master/%E5%9B%BE%E7%89%87/%E9%B8%A2%E5%B0%BE%E8%8A%B1%E6%95%B0%E6%8D%AE%E9%9B%86%E5%88%86%E7%B1%BB%E7%BB%93%E6%9E%9C%EF%BC%88%E5%A4%9A%E7%A7%8D%E6%96%B9%E6%B3%95%EF%BC%89.png)


&ensp;&ensp;&ensp;&ensp;可以看到不同的分类方法都可以得到截然不同的结果，它们中的大多数都得到了很好的结果，但我们不可能说那种分类方法是最好的方法，而要根据用户的实际需求去选择更合适的方法。

## 2.数字图像识别
代码如下：
```javascript
print(__doc__)

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

# 导入digits手写字体数据集
digits = datasets.load_digits()

# 我们感兴趣的数据是由8x8的数字图像组成的，让我们看一下存储在数据集的images属性
# 中的前4张图像。 如果我们正在处理图像文件，则可以使用matplotlib.pyplot.imread
# 加载它们。 请注意，每个图像必须具有相同的大小。 对于这些图像，我们知道它们代表
# 哪个数字：它在数据集的“目标”中给出。

_, axes = plt.subplots(4, 5)      #生产一个图像包含2行4列的子图：_存图像,axes存子图
#_.show()

#将数字图像的数组与预测数字相匹配
images_and_labels = list(zip(digits.images, digits.target))
print(images_and_labels [0])

#按顺序显示数字图像及其训练值
for ax, (image, label) in zip(axes[0, :], images_and_labels[:5]):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='none')
    print(image)
    ax.set_title('Training: %i' % label)

for ax, (image, label) in zip(axes[1, :], images_and_labels[5:10]):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='none')
    ax.set_title('Training: %i' % label)

# 要对该数据应用分类器，我们需要对图像进行扁平化，将数据转化为一个(样本，特征)矩阵:

#将矩阵向量化，用于评价
n_samples = len(digits.images)  #一共有n_samples-1（1796）个images数据
data = digits.images.reshape((n_samples, -1)) #每个images数据向量化组成数组（1796行，64列）

# 创建分类器:支持向量分类器（SVC-rbf方法）
classifier = svm.SVC(gamma=0.001)
#classifier = svm.NuSVC(gamma=0.001)

# 将数据分成训练子集和测试子集
X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.5, shuffle=False)

# 我们在数字的前半部分学习数字
classifier.fit(X_train, y_train)

# 现在，预测下半部分的数字值：
predicted = classifier.predict(X_test)

#按顺序显示数字图像及其预测值
images_and_predictions = list(zip(digits.images[n_samples // 2:], predicted))
for ax, (image, prediction) in zip(axes[2, :], images_and_predictions[1:6]):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title('Prediction: %i' % prediction)

for ax, (image, prediction) in zip(axes[3, :], images_and_predictions[6:11]):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title('Prediction: %i' % prediction)

#打印分类器的参数（分类器类型，其中包括核函数，维度等）
#打印分类效果报告
print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(y_test, predicted)))
#创建混淆矩阵，将预测数据集的实际值与预测值进行显示对比
disp = metrics.plot_confusion_matrix(classifier, X_test, y_test)
#数字化显示
disp.figure_.suptitle("Confusion Matrix")
print("Confusion matrix:\n%s" % disp.confusion_matrix)
plt.show()
```
&ensp;&ensp;&ensp;&ensp;接下来我们来讨论另一个经典案例——数字图像识别，这里我们主要针对0-9十个数字进行分析。这可以认为是一种分类问题，我们可以把数字图像分解为相同尺寸的像素点，通过对图片扁平化，把像素的颜色深浅数字化，这样我们就可以把一个图片简化为一个二维数组，将数组作为变量，而图片代表的数字作为预测值，通过足够的样本量对其进行训练他就可以实现对数字图像的预测。
&ensp;&ensp;&ensp;&ensp;接下来我们进行数字图像识别的实现，数据集我们用sklearn模块中的datasets成员中的手写数据集，其中包含大量的数字图像并匹配有其对应的数字。首先我们通过程序将图片可视化显示出来，得到以下图像，他们分别对应阿拉伯数字0-9。
 
 ![原始数字图像](https://github.com/trybesthbk/mechine-learning/blob/master/%E5%9B%BE%E7%89%87/%E5%8E%9F%E5%A7%8B%E6%95%B0%E5%AD%97%E5%9B%BE%E5%83%8F.png)

&ensp;&ensp;&ensp;&ensp;可以看到每个数字都由一个8X8的像素矩阵组成，为了便于理解，我们通过打印功能将该图像数字化，可以看到，每一个数字图像都可以被量化为一个8X8的矩阵，并且数字由小到大对应颜色由蓝到黄。下图为数字0对应的矩阵以及其所代表的数字0。
 
 ![数字图像矩阵（0）](https://github.com/trybesthbk/mechine-learning/blob/master/%E5%9B%BE%E7%89%87/%E6%95%B0%E5%AD%97%E5%9B%BE%E5%83%8F%E7%9F%A9%E9%98%B5%EF%BC%880%EF%BC%89.png)


&ensp;&ensp;&ensp;&ensp;为了进行下一步的分析，复杂的矩阵显然是不方便的，我们将数字图像矩阵向量化，即生成一个n行64列的矩阵，其中n代表数据的数量。在机器学习的过程中我们先要通过一定的数据对模型进行训练以生成准确的模型，之后再用生成的模型对类似的数据进行检测，前者的数据我们成为训练集，后者的数据我们成为测试集。接下来我们就将手写数据集分为两个部分，本例中我们对半分取一半为训练集，一半为测试集。
&ensp;&ensp;&ensp;&ensp;对于训练集这里我们建立了一个基于径向基核函数的支持向量分类机进行训练，这在pycharm中通过调用模块可以很容易的实现。接再将测试集的观察值进行测试，这里为了便于观察，我们将训练情况和观察情况进行可视化。

![数字图像训练集测试情况](https://github.com/trybesthbk/mechine-learning/blob/master/%E5%9B%BE%E7%89%87/%E6%95%B0%E5%AD%97%E5%9B%BE%E5%83%8F%E8%AE%AD%E7%BB%83%E9%9B%86%E6%B5%8B%E8%AF%95%E6%83%85%E5%86%B5.png)


&ensp;&ensp;&ensp;&ensp;上图可以看出数字图像的测试效果较好，几个数字都取得了很好的匹配情况。但是接下来我们还需要讨论一个问题，如何评价数字图像的检测精度呢，显然可以将判断的正确率作为一个判定依据，这里我们通过混淆矩阵的方式将测试的效果可视化。下图为通过sklearn调用的metrics成员以得到其准确率的混淆矩阵图像如下：

![混淆矩阵图像](https://github.com/trybesthbk/mechine-learning/blob/master/%E5%9B%BE%E7%89%87/%E6%B7%B7%E6%B7%86%E7%9F%A9%E9%98%B5%E5%9B%BE%E5%83%8F.png)


&ensp;&ensp;&ensp;&ensp;上图中横轴表示上述的训练模型预测的数字，而纵轴表示其实际代表的数字，可以看出各个数字的准确率均达到了一个很高的水平，也反应了我们上述工作的成果是可信的，同样的我们可以得到一个评价参数报表，详细的反映了测试的情况。

![模型评价情况 ](https://github.com/trybesthbk/mechine-learning/blob/master/%E5%9B%BE%E7%89%87/%E6%A8%A1%E5%9E%8B%E8%AF%84%E4%BB%B7%E6%83%85%E5%86%B5.png)


以下对几个常见评价指标进行解释说明如下：
<center>评价指标说明</center>

评价指标     | 含义
-------- | -----
Precision（精度）  | 预测正确个数占所述实际值个数的比例
Recall（召回率） | 预测正确个数占所述实际值个数的比例
f1-score（f1值）  | 上述两值的算数平均数除以几何平均数
Support（数据量）  |相应数据个数
Accuracy（准确度） | 总体预测正确个数占总数据量比例
macro avg（宏平均）  | 按照不同预测值求平均
weighted avg（加权平均）  | 按照不同预测值数据量求加权平均
  
&ensp;&ensp;&ensp;&ensp;这里我们就完成了这样一个数字图像识别的流程，但同样我们还是要遇到一个问题，这样一个模型它是否是最优的呢，显然通过采用不同的分类方法，核函数，训练集数量都会对预测结果产生影响，上述工作我们是通过SVC分类方法，采用rbf核函数，测试集比例为0.5情况下测得的。接下来我们重复上述的工作，对比不同模型的预测情况：
<center>不同模型预测情况</center>

参数(默认分类方法=SVC,核函数=rbf，gamma=0.001,测试集比例=0.5)    | 准确度  | 测试数据量
-------- | -----  | -----
分类方法=LinearSVC  |0.91 |899
分类方法=NuSVC | 0.95|899
核函数=linear| 0.95|899
核函数=poly |0.96 |899
测试集比例=0.1 | 0.96|360
测试集比例=0.9| 0.85|1618、
  
&ensp;&ensp;&ensp;&ensp;以上我们主要提供各种模型建立及其效果，具体使用还是看具体的需求，总体而言说明我们此前建立的模型精度相对较高，可以采用。此外根据测试集比例我们还可以得出一个结论（上述仅列举两个数据，实际应有大量数据支撑），在测试集数据较少时数据的增加对预测的准确度很有帮助，但随着数据增加到一定程度，准确度基本维持在一个稳定的值，这和我们的学习过程也很相似。
&ensp;&ensp;&ensp;&ensp;这里我们就完成了这样一个数字图像识别的工作，但可能有的朋友会觉得上述的模型只是针对已有的数据完成分类的工作，并不具有时效性。实际上上述工作展示了解决问题的思路，对于实际的应用上，我们只需要增加一个图像输入的功能，并通过图像的数字化转化为8X8的矩阵，作为测试集即可完成实现识别工作，但外界的字体并不一定与训练集中的字体特征并不一定完全相同，这也是上述模型的局限性，实际模型的效果与训练集的数量与广度都有关系一系列问题可以通过分类算法的优化，数字图像矩阵的精度等方面就行改进。

## 3. 补充说明：

&ensp;&ensp;&ensp;&ensp;本文的应用实例均在pycharm环境中通过python3.7编译运行，案例及代码均参考sklearn官网，并经过了一定的修改与注释。
相关链接： [https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html#sphx-glr-auto-examples-classification-plot-digits-classification-py](https://mp.csdn.net).

 [https://scikit-learn.org/stable/auto_examples/linear_model/plot_iris_logistic.html#sphx-glr-auto-examples-linear-model-plot-iris-logistic-py](https://mp.csdn.net).
 
