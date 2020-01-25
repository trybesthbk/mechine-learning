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