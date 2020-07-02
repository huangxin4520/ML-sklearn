1. [AdaBoost是什么](https://github.com/huangxin4520/ML-sklearn/blob/master/ensembel/1.Adaboost/AdaBoost.md#1adaBoost是什么)
      1. [AdaBoost的定义](https://github.com/huangxin4520/ML-sklearn/blob/master/ensembel/1.Adaboost/AdaBoost.md#11adaBoost的定义)
      2. [AdaBoost的步骤](https://github.com/huangxin4520/ML-sklearn/blob/master/ensembel/1.Adaboost/AdaBoost.md#12adaBoost的步骤)
      3. [AdaBoost的具体案例](https://github.com/huangxin4520/ML-sklearn/blob/master/ensembel/1.Adaboost/AdaBoost.md#13adaBoost具体案例)
      4. [AdaBoost的两种权重](https://github.com/huangxin4520/ML-sklearn/blob/master/ensembel/1.Adaboost/AdaBoost.md#14adaBoost的两种权重)
      5. [AdaBoost的优缺点](https://github.com/huangxin4520/ML-sklearn/blob/master/ensembel/1.Adaboost/AdaBoost.md#15adaBoost的优缺点)
a
2. [sklearn里面的参数含义](https://github.com/huangxin4520/ML-sklearn/blob/master/ensembel/1.Adaboost/AdaBoost.md#2sklearn里面的参数含义)
3. [代码实现](https://github.com/huangxin4520/ML-sklearn/blob/master/ensembel/1.Adaboost/AdaBoost.md#3代码实现)
4. [参考文献](https://github.com/huangxin4520/ML-sklearn/blob/master/ensembel/1.Adaboost/AdaBoost.md#4参考文献)



#### **1.AdaBoost是什么**

**AdaBoost是Boosting的一种算法。**[Boosting是什么，点击此处。]()

​		Yoav Freund和Robert Schapire在1995年提出的AdaBoost算法。

##### **1.1AdaBoost的定义**

​		AdaBoost是英文"Adaptive Boosting"（自适应增强）的缩写，它的自适应在于：前一个基本分类器被错误分类的样本的权值会增大，而正确分类的样本的权值会减小，并再次用来训练下一个基本分类器。同时，在每一轮迭代中，加入一个新的弱分类器，直到达到某个预定的足够小的错误率或达到预先指定的最大迭代次数才确定最终的强分类器。

##### **1.2AdaBoost的步骤**

具体步骤可以看王喆的《百面机器学习》P282

**（1）首先，是初始化训练数据的权值分布D1。**

​		假设有N个训练样本数据，则每一个训练样本最开始时，都被赋予相同的权值：**D1=1/N**。
**（2）然后，训练弱分类器ht。**

​		具体训练过程中是：如果某个训练样本点，被弱分类器hi准确地分类，那么在构造下一个训练集中，它对应的权值要减小；相反，如果某个训练样本点被错误分类，那么它的权值就应该增大。权值更新过的样本集被用于训练下一个分类器，整个训练过程如此迭代地进行下去。

- 采样出子集，用子集训练ht。

- 计算hi的错误率：

   <img src="http://chart.googleapis.com/chart?cht=tx&chl=\\varepsilon_t=\\frac{\\sum_{i=1}^{N_t}{I[h_t(x_i)\\neq y_i]D_t}}{N_t}" style="border:none;">

- 计算基分类器的权重：
<img src="http://chart.googleapis.com/chart?cht=tx&chl=\varepsilon_t=a_t=log\frac{(1-\varepsilon_t)}{\varepsilon_t}" style="border:none;">

- 设置下一次采样的权重：
<img src="http://chart.googleapis.com/chart?cht=tx&chl=D（t+1)=(\begin{cases} \frac{D_t(i)（1-\varepsilon_t)}{\varepsilon_t}，h_t(x_i)\neq y_i\\ \frac{D_t(i)（\varepsilon_t)}{1-\varepsilon_t}，h_t(x_i)= y_i\end{cases})" style="border:none;">

**（3）最后，将各个训练得到的弱分类器组合成一个强分类器。**

​		各个弱分类器的训练过程结束后，加大分类误差率小的弱分类器的权重，使其在最终的分类函数中起着较大的决定作用，而降低分类误差率大的弱分类器的权重，使其在最终的分类函数中起着较小的决定作用。
换而言之，**误差率低的弱分类器在最终分类器中占的权重较大，否则较小。**
$$
sign(\sum_{t=1}^{T}h_t(z)a_t)
$$


##### **1.3AdaBoost具体案例**

​		举个**3层迭代**的例子，各个样本权值和误差率的变化，如下所示（其中，样本权值D中加了下划线的表示在上一轮中被分错的样本的新权值）：

![img](https://img-blog.csdn.net/20141103002143995)

训练之前，各个样本的权值被初始化为D1 = (0.1, 0.1,0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)；

- 第一轮迭代中，样本“6 7 8”被分错，对应的误差率为e1=P(G1(xi)≠yi) = 3*0.1 = 0.3，此第一个基本分类器在最终的分类器中所占的权重为a1 = 0.4236。第一轮迭代过后，样本新的权值为D2 = (0.0715, 0.0715, 0.0715, 0.0715, 0.0715,  0.0715, 0.1666, 0.1666, 0.1666, 0.0715)；
- 第二轮迭代中，样本“3 4 5”被分错，对应的误差率为e2=P(G2(xi)≠yi) = 0.0715 * 3 = 0.2143，此第二个基本分类器在最终的分类器中所占的权重为a2 = 0.6496。第二轮迭代过后，样本新的权值为D3 = (0.0455, 0.0455, 0.0455, 0.1667, 0.1667,  0.01667, 0.1060, 0.1060, 0.1060, 0.0455)；
- 第三轮迭代中，样本“0 1 2 9”被分错，对应的误差率为e3 = P(G3(xi)≠yi) = 0.0455*4 = 0.1820，此第三个基本分类器在最终的分类器中所占的权重为a3 = 0.7514。第三轮迭代过后，样本新的权值为D4 = (0.125, 0.125, 0.125, 0.102, 0.102,  0.102, 0.065, 0.065, 0.065, 0.125)。
- 从上述过程中可以发现，如果某些个样本被分错，它们在下一轮迭代中的权值将被增大，同时，其它被分对的样本在下一轮迭代中的权值将被减小**。就这样，分错样本权值增大，分对样本权值变小，而在下一轮迭代中，总是选取让误差率最低的阈值来设计基本分类器，所以误差率e（所有被Gm(x)误分类样本的权值之和）不断降低。**

综上，将上面计算得到的a1、a2、a3各值代入G(x)中，G(x) = sign[f3(x)] = sign[ a1 * G1(x) + a2 * G2(x) + a3 * G3(x) ]，得到最终的分类器为：
$$
G(x) = sign[f3(x)] = sign[ 0.4236G1(x) + 0.6496G2(x)+0.7514G3(x) ]。
$$


##### **1.4AdaBoost的两种权重**

一种为数据权重、一种为分类器权重
**数据权重：**用于确定分类器权重（弱分类器寻找其分类最小的决策点，找到之后用这个最小的误差计算出弱分类器的权重）
**分类器权重：**说明了弱分类器在最终决策中拥有发言权的大小
**数据权重**
		最开始每个点的权重都相同，错误就会增加权重。每训练一个弱分类器就会调整每个店的权重，上一轮训练中被错误分类点的权重增加，促使下一轮着分析错误分类点，达到“你分不对我来分的”效果。
由于每个分类器都会关注上个分错的点，那么也就是说每个分类器都有侧重。
**分类器权重**
		每个分类器都有可能分对其上一个分类器没分对的数据，同时针对上一个分类器分队的数据也可能没有分队。这就导致了分类器中都有各自最关注的点，这就说明每一个分类器都只关注训练数据中的一部分数据，全部分类器组合起来才能发挥作用，那么最终的结果是通过加权“投票“决定的，权重的大小是根据弱分类器的分类错误率计算出来的。

##### **1.5AdaBoost的优缺点**

**优点：**

- 分类精度高，构造简单，结果可理解。
- 可以使用各种回归分类模型来构建弱学习器，非常灵活。
- 不容易过拟合。

**缺点：**

- 训练时会过于偏向分类困难的数据，导致Adaboost容易受噪声数据干扰。
- 依赖于弱分类器，训练时间可能比较长。

**Adaboost算法的某些特性是非常好的，这里主要介绍Adaboost的两个特性**。

（1）是训练的错误率上界，随着迭代次数的增加，会逐渐下降；

（2）是Adaboost算法即使训练次数很多，也不会出现过拟合的问题。

#### **2.sklearn里面的参数含义**

**class sklearn.ensemble.AdaBoostClassifier(base_estimator=None, *, n_estimators=50, learning_rate=1.0, algorithm='SAMME.R', random_state=None)**

| 参数                    | 默认           |                                                              |
| ----------------------- | -------------- | ------------------------------------------------------------ |
| **base_estimator**     | 默认=None      | 基分类器模型，一般是决策树模型。如果是None，则基本估计量为DecisionTreeClassifier(max_depth=1)。 |
| **n_estimators int**    | 默认= 50       | 终止增强的估计器的最大数量。                                 |
| **learning_rate float** | 默认= 1        | learning_rate和 之间需要权衡n_estimators。                   |
| **algorithm**           | 默认='SAMME.R' | **取值为{'SAMME'，'SAMME.R'}**如果为“ SAMME.R”，则使用SAMME.R真正的增强算法。 base_estimator必须支持类概率的计算。如果为“ SAMME”，则使用SAMME离散提升算法。SAMME.R算法通常比SAMME收敛更快，从而以更少的提升迭代次数实现了更低的测试误差。 |
| **random_state**       | 默认=None      | 随机种子                                                     |

#### **3.代码实现**

https://github.com/huangxin4520/ML-sklearn/tree/master/ensembel/1.Adaboost

#### **4.参考文献**

[（十三）通俗易懂理解——Adaboost算法原理](https://zhuanlan.zhihu.com/p/41536315)

[scikit-learn Adaboost类库使用小结](https://www.cnblogs.com/pinard/p/6136914.html)

[Adaboost 算法的原理与推导](https://blog.csdn.net/v_july_v/article/details/40718799)

https://scikit-learn.org

