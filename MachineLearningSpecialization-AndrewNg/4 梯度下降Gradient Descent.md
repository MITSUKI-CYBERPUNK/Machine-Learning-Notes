# 4 梯度下降Gradient Descent

## 4.1 数学定义

梯度下降是一个用来**求函数最小值**的算法，我们将使用梯度下降算法来求出代价函数$J(\theta_{0}, \theta_{1})$ 的最小值。

梯度下降背后的思想是：开始时我们随机选择一个参数的组合$\left(  {\theta_{0}},{\theta_{1}},......,{\theta_{n}}  \right)$，计算代价函数，然后我们寻找下一个**能让代价函数值下降最多的参数组合**。我们持续这么做直到找到一个局部最小值（**local minimum**），因为我们并没有尝试完所有的参数组合，所以不能确定我们得到的局部最小值是否便是全局最小值（**global minimum**），选择不同的初始参数组合，可能会找到不同的局部最小值。

![image-20240130120707450](https://aquazone.oss-cn-guangzhou.aliyuncs.com/image-20240130120707450.png)

想象一下你正站立在山的这一点上，站立在你想象的公园这座红色山上，在梯度下降算法中，我们要做的就是旋转360度，看看我们的周围，并问自己要在某个方向上，用小碎步尽快下山。这些小碎步需要朝什么方向？如果我们站在山坡上的这一点，你看一下周围，你会发现最佳的下山方向，你再看看周围，然后再一次想想，我应该从什么方向迈着小碎步下山？然后你按照自己的判断又迈出一步，重复上面的步骤，从这个新的点，你环顾四周，并决定从什么方向将会最快下山，然后又迈进了一小步，并依此类推，直到你接近局部最低点的位置。

**批量梯度下降（batch gradient descent）算法的公式为：**

![image-20240130120804477](https://aquazone.oss-cn-guangzhou.aliyuncs.com/image-20240130120804477.png)

其中$a$是**学习率（learning rate）**，通常是0到1之间的一个小正数。它**决定了我们沿着能让代价函数下降程度最大的方向向下迈出的步子有多大**，在批量梯度下降中，我们每一次都同时让所有的参数减去学习速率乘以代价函数的导数(进行少量调整，决定了方向以及步数大小)。 ^ed1145
学习速率乘以代价函数的导数在梯度下降算法中起着关键作用。这个值代表了**在当前参数值处，代价函数变化最快的方向和速率**。换句话说，它表示了在当前参数值处，代价函数的**斜率或梯度**。

通过将学习速率乘以代价函数的导数，我们可以确定在每次迭代中应该更新参数的方向和步长。如果导数值为正，说明应该向参数减小的方向移动以减小代价函数；如果导数值为负，说明应该向参数增加的方向移动。

**等号可表示赋值或者真值断言，此处是赋值。**

![image-20240130120820778](https://aquazone.oss-cn-guangzhou.aliyuncs.com/image-20240130120820778.png)

在梯度下降算法中，还有一个更微妙的问题，梯度下降中，我们要更新${\theta_{0}}$和${\theta_{1}}$ ，当 $j = 0$和$j=1$时，会产生更新，所以你将更新$J\left( {\theta_{0}} \right)$和$J\left(  {\theta_{1}}  \right)$。实现梯度下降算法的微妙之处是，在这个表达式中，如果你要更新这个等式，你需要同时更新${\theta_{0}}$和${\theta_{1}}$，我的意思是在这个等式中，我们要这样更新：

${\theta_{0}}$:= ${\theta_{0}}$

并更新

${\theta_{1}}$:= ${\theta_{1}}$

实现方法是：**你应该计算公式右边的部分，通过那一部分计算出${\theta_{0}}$和${\theta_{1}}$的值，然后同时更新${\theta_{0}}$和${\theta_{1}}$。**

进一步阐述这个过程：

![image-20240130121059884](https://aquazone.oss-cn-guangzhou.aliyuncs.com/image-20240130121059884.png)

在梯度下降算法中，这是正确实现同时更新的方法。千万不要带入！我不打算解释为什么你需要**同时更新**，同时更新是梯度下降中的一种常用方法。我们之后会讲到，**同步更新是更自然的实现方法。当人们谈到梯度下降时，他们的意思就是同步更新**。

在接下来的视频中，我们要进入这个微分项的细节之中。我已经写了出来但没有真正定义，如果你已经修过微积分课程，如果你熟悉偏导数和导数，这其实就是这个微分项：

**$\alpha \frac{\partial }{\partial {{\theta }*{0}}}J({{\theta }*{0}},{{\theta }*{1}})$，$\alpha \frac{\partial }{\partial {{\theta }*{1}}}J({{\theta }*{0}},{{\theta }*{1}})$**



## 4.2 梯度下降的直观理解

在之前的视频中，我们给出了一个数学上关于梯度下降的定义，本次视频我们更深入研究一下，更直观地感受一下这个算法是做什么的，以及梯度下降算法的更新过程有什么意义。梯度下降算法如下：

**${\theta_{j}}:={\theta_{j}}-\alpha \frac{\partial }{\partial {\theta_{j}}}J\left(\theta \right)$**

描述：对 $\theta$ $赋值，使得$$J\left( \theta  \right)$ 
按梯度下降最快方向进行，一直迭代下去，最终得到局部最小值。其中$a$是**学习率（learning rate）**，它**决定了我们沿着能让代价函数下降程度最大的方向向下迈出的步子有多大**。

**请注意**：我们的目的是**找到让代价函数J取得最小值的参数$\theta$** ,通过不断更新它的值来实现，当代价函数收敛，即代价函数趋于最小值，所获得的参数就是我们的最优参数。

![image-20240130122145562](https://aquazone.oss-cn-guangzhou.aliyuncs.com/image-20240130122145562.png)

对于这个问题，求导的目的，基本上可以说取这个红点的切线，就是这样一条红色的直线，刚好与函数相切于这一点，让我们看看这条红色直线的斜率，就是这条刚好与函数曲线相切的这条直线，这条直线的斜率正好是这个三角形的高度除以这个水平长度，现在，这条线有一个正斜率，也就是说它有正导数，因此，我得到的新的${\theta_{1}}$，${\theta_{1}}$更新后等于${\theta_{1}}$减去一个正数乘以$a$。如果在左边则为负数，仍然在向最小值处下降。当我们接近最小值时，导数会自动变小，尽管变化率固定，也会使步子变小。

这就是我梯度下降法的更新规则：${\theta_{j}}:={\theta_{j}}-\alpha \frac{\partial }{\partial {\theta_{j}}}J\left( \theta  \right)$

让我们来看看如果$a$太小或$a$太大会出现什么情况：

如果$a$太小了，即我的学习速率太小，结果就是只能这样像小宝宝一样一点点地挪动，去努力接近最低点，这样就需要很多步才能到达最低点，所以如果$a$太小的话，可能会很慢，因为它会一点点挪动，它会需要很多步才能到达全局最低点。

如果$a$太大，那么梯度下降法可能会越过最低点，甚至可能无法收敛，下一次迭代又移动了一大步，越过一次，又越过一次，一次次越过最低点，直到你发现实际上离最低点越来越远，所以，如果$a$太大，它会导致无法收敛，甚至发散。

现在，我还有一个问题，当我第一次学习这个地方时，我花了很长一段时间才理解这个问题，如果我们预先把${\theta_{1}}$放在一个局部的最低点，你认为下一步梯度下降法会怎样工作？

假设你将${\theta_{1}}$初始化在局部最低点，在这儿，它已经在一个局部的最优处或局部最低点。结果是局部最优点的导数将等于零，因为它是那条切线的斜率。这意味着你已经在局部最优点，它使得${\theta_{1}}$不再改变，也就是新的${\theta_{1}}$等于原来的${\theta_{1}}$，因此，如果你的参数已经处于局部最低点，那么梯度下降法更新其实什么都没做，它不会改变参数的值。这也解释了为什么即使学习速率$a$保持不变时，梯度下降也可以收敛到局部最低点。

我们来看一个例子，这是代价函数$J\left( \theta  \right)$。

![image-20240130122232071](https://aquazone.oss-cn-guangzhou.aliyuncs.com/image-20240130122232071.png)

我想找到它的最小值，首先初始化我的梯度下降算法，在那个品红色的点初始化，如果我更新一步梯度下降，也许它会带我到这个点，因为这个点的导数是相当陡的。现在，在这个绿色的点，如果我再更新一步，你会发现我的导数，也即斜率，是没那么陡的。随着我接近最低点，我的导数越来越接近零，所以，梯度下降一步后，新的导数会变小一点点。然后我想再梯度下降一步，在这个绿点，我自然会用一个稍微跟刚才在那个品红点时比，再小一点的一步，到了新的红色点，更接近全局最低点了，因此这点的导数会比在绿点时更小。所以，我再进行一步梯度下降时，我的导数项是更小的，${\theta_{1}}$更新的幅度就会更小。所以随着梯度下降法的运行，你移动的幅度会自动变得越来越小，直到最终移动幅度非常小，你会发现，已经收敛到局部极小值。

回顾一下，在梯度下降法中，**当我们接近局部最低点时，梯度下降法会自动采取更小的幅度**，这是因为当我们接近局部最低点时，很显然在局部最低时导数等于零，所以**当我们接近局部最低时，导数值会自动变得越来越小，所以梯度下降将自动采取较小的幅度**，这就是梯度下降的做法。所以实际上没有必要再另外减小$a$。

这就是**梯度下降算法，可以用它来最小化任何代价函数$J$，不只是线性回归中的代价函数$J$。**

在接下来的视频中，我们要用代价函数$J$，回到它的本质，线性回归中的代价函数。也就是我们前面得出的平方误差函数，结合梯度下降法，以及平方代价函数，我们会得出第一个机器学习算法，即线性回归算法。



## 4.3 梯度下降的线性回归

在以前的视频中我们谈到关于梯度下降算法，梯度下降是很常用的算法，它不仅被用在线性回归上和线性回归模型、平方误差代价函数。在这段视频中，我们要将梯度下降和代价函数结合。我们将用到此算法，并将其应用于具体的拟合直线的线性回归算法里。

梯度下降算法和线性回归算法比较如图：

![image-20240130122430576](https://aquazone.oss-cn-guangzhou.aliyuncs.com/image-20240130122430576.png)

对我们之前的线性回归问题运用梯度下降法，关键在于求出代价函数的导数，即：

$\frac{\partial }{\partial {{\theta }_{j}}}J({{\theta }_{0}},{{\theta }_{1}})=\frac{\partial }{\partial {{\theta }_{j}}}\frac{1}{2m}{{\sum\limits_{i=1}^{m}{\left( {{h}_{\theta }}({{x}^{(i)}})-{{y}^{(i)}} \right)}}^{2}}$

$j=0$  时：$\frac{\partial }{\partial {{\theta }_{0}}}J({{\theta }_{0}},{{\theta }_{1}})=\frac{1}{m}{{\sum\limits_{i=1}^{m}{\left( {{h}_{\theta }}({{x}^{(i)}})-{{y}^{(i)}} \right)}}}$

$j=1$  时：$\frac{\partial }{\partial {{\theta }_{1}}}J({{\theta }_{0}},{{\theta }_{1}})=\frac{1}{m}\sum\limits_{i=1}^{m}{\left( \left( {{h}_{\theta }}({{x}^{(i)}})-{{y}^{(i)}} \right)\cdot {{x}^{(i)}} \right)}$

**则算法改写成：**

**Repeat {**
${\theta_{0}}:={\theta_{0}}-a\frac{1}{m}\sum\limits_{i=1}^{m}{ \left({{h}_{\theta }}({{x}^{(i)}})-{{y}^{(i)}} \right)}$

 ${\theta_{1}}:={\theta_{1}}-a\frac{1}{m}\sum\limits_{i=1}^{m}{\left( \left({{h}_{\theta }}({{x}^{(i)}})-{{y}^{(i)}} \right)\cdot {{x}^{(i)}} \right)}$

**}**

我们刚刚使用的算法，有时也称为**批量梯度下降**。实际上，在机器学习中，通常不太会给算法起名字，但这个名字”**批量梯度下降**”，指的是**在梯度下降的每一步中，我们都用到了所有的训练样本，在梯度下降中，在计算微分求导项时，我们需要进行求和运算，所以，在每一个单独的梯度下降中，我们最终都要计算这样一个东西，这个项需要对所有$m$个训练样本求和。因此，批量梯度下降法这个名字说明了我们需要考虑所有这一"批"训练样本，而事实上，有时也有其他类型的梯度下降法，不是这种"批量"型的，不考虑整个的训练集，而是每次只关注训练集中的一些小的子集。在后面的课程中，我们也将介绍这些方法。

**此函数是凸函数，碗形函数，除了单个全局最小值以外，它不能有任何局部最小值。**

但就目前而言，应用刚刚学到的算法，你应该已经掌握了批量梯度算法，并且能把它应用到线性回归中了，这就是**用于线性回归的梯度下降法**。

如果你之前学过线性代数，有些同学之前可能已经学过高等线性代数，你应该知道有一种计算代价函数$J$最小值的数值解法，不需要梯度下降这种迭代算法。在后面的课程中，我们也会谈到这个方法，它可以在不需要多步梯度下降的情况下，也能解出代价函数$J$的最小值，这是另一种称为**正规方程(normal equations)** 的方法。实际上**在数据量较大的情况下，梯度下降法比正规方程要更适用一些。**