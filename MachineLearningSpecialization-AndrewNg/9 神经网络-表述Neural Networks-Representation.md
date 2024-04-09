# 9 神经网络-表述Neural Networks-Representation

## 9.1 非线性假设Non-linear Hypotheses

我们之前学的，无论是线性回归还是逻辑回归都有这样一个缺点，即：当特征太多时，计算的负荷会非常大。

下面是一个例子：

![image-20240207163703582](https://aquazone.oss-cn-guangzhou.aliyuncs.com/image-20240207163703582.png)

当我们使用x~1~,x~2~  的多次项式进行预测时，我们可以应用的很好。 之前我们已经看到过，使用非线性的多项式项，能够帮助我们建立更好的分类模型。假设我们有非常多的特征，例如大于100个变量，我们希望用这100个特征来构建一个非线性的多项式模型，结果将是数量非常惊人的特征组合，即便我们只采用两两特征的组合(x~1~x~2~+x~1~x~3~+......x~2~x~3~+......+x~99~x~100~)，我们也会有接近5000个组合而成的特征。这对于一般的逻辑回归来说需要计算的特征太多了。

假设我们希望训练一个模型来识别视觉对象（例如识别一张图片上是否是一辆汽车），我们怎样才能这么做呢？一种方法是我们利用很多汽车的图片和很多非汽车的图片，然后利用这些图片上一个个像素的值（饱和度或亮度）来作为特征。

假如我们只选用灰度图片，每个像素则只有一个值（而非 RGB值），我们可以选取图片上的两个不同位置上的两个像素，然后训练一个逻辑回归算法利用这两个像素的值来判断图片上是否是汽车：

![image-20240207164000409](https://aquazone.oss-cn-guangzhou.aliyuncs.com/image-20240207164000409.png)

假使我们采用的都是50x50像素的小图片，并且我们将所有的像素视为特征，则会有 2500个特征，如果我们要进一步将两两特征组合构成一个多项式模型，则会有约2500^2^/2个（接近3百万个）特征。普通的逻辑回归模型，不能有效地处理这么多的特征，这时候我们需要神经网络。



## 9.2 神经元和大脑

我将向你们介绍神经网络。因为它能很好地解决不同的机器学习问题。而不只因为它们在逻辑上行得通，在这段视频中，我想告诉你们一些神经网络的背景知识，由此我们能知道可以用它们来做什么。不管是将其应用到现代的机器学习问题上，还是应用到那些你可能会感兴趣的问题中。也许，这一伟大的人工智能梦想在未来能制造出真正的智能机器。另外，我们还将讲解神经网络是怎么涉及这些问题的神经网络产生的原因是人们想尝试设计出模仿大脑的算法，从某种意义上说如果我们想要建立学习系统，那为什么不去模仿我们所认识的最神奇的学习机器——人类的大脑呢？

神经网络逐渐兴起于二十世纪八九十年代，应用得非常广泛。但由于各种原因，在90年代的后期应用减少了。但是最近，神经网络又东山再起了。其中一个原因是：神经网络是计算量有些偏大的算法。然而大概由于近些年计算机的运行速度变快，才足以真正运行起大规模的神经网络。正是由于这个原因和其他一些我们后面会讨论到的技术因素，如今的神经网络对于许多应用来说是最先进的技术。当你想模拟大脑时，是指想制造出与人类大脑作用效果相同的机器。大脑可以学会去以看而不是听的方式处理图像，学会处理我们的触觉。

我们能学习数学，学着做微积分，而且大脑能处理各种不同的令人惊奇的事情。似乎如果你想要模仿它，你得写很多不同的软件来模拟所有这些五花八门的奇妙的事情。不过能不能假设大脑做所有这些，不同事情的方法，不需要用上千个不同的程序去实现。相反的，大脑处理的方法，只需要一个单一的学习算法就可以了？尽管这只是一个假设，不过让我和你分享一些这方面的证据。

兴衰史： 

![image-20240207162153487](https://aquazone.oss-cn-guangzhou.aliyuncs.com/image-20240207162153487.png)

神经网络可能为我们打开一扇进入遥远的人工智能梦的窗户，但我在这节课中讲授神经网络的原因，主要是对于现代机器学习应用。它是最有效的技术方法。因此在接下来的一些课程中，我们将开始深入到神经网络的技术细节。



## 9.3 模型表示

仿生类比：

![image-20240207162455774](https://aquazone.oss-cn-guangzhou.aliyuncs.com/image-20240207162455774.png)

**警告：尽管这里做了一个松散的类比，但今天我们仍然几乎不知道人脑是如何工作的，但即使有了我们将要讨论的这些极其简化的神经元模型，我们也将能够构建真正强大的深度学习算法**

神经网络模型建立在很多神经元之上，每一个神经元又是一个个学习模型。这些**神经元（也叫激活单元，activation unit）**采纳一些特征作为输出，并且根据本身的模型提供一个输出。下图是一个以逻辑回归模型作为自身学习模型的神经元示例，在神经网络中，**参数又可被成为权重（weight）**。

举例:需求预测

![image-20240207165147357](https://aquazone.oss-cn-guangzhou.aliyuncs.com/image-20240207165147357.png)

神经网络架构问题：有多少个隐藏层以及每个隐藏层有多少个神经元等

![image-20240207165625938](https://aquazone.oss-cn-guangzhou.aliyuncs.com/image-20240207165628215.png)



我们设计出了类似于神经元的神经网络，效果如下：

![image-20240207165830720](https://aquazone.oss-cn-guangzhou.aliyuncs.com/image-20240207165830720.png)

其中x~1~,x~2~,x~3~是**输入单元（input units）**，我们将原始数据输入给它们。a~1~,a~2~.a~3~是中间单元，它们负责将数据进行处理，然后呈递到下一层。 最后是输出单元，它负责计算${h_\theta}\left( x \right)$。

神经网络模型是许多逻辑单元按照不同层级组织起来的网络，每一层的输出变量都是下一层的输入变量。下图为一个3层的神经网络，第一层成为**输入层（Input Layer）**，最后一层称为**输出层（Output Layer）**，中间一层成为**隐藏层（Hidden Layers）**。我们为每一层都增加一个**偏差单位（bias unit）**：



### 9.3.1 神经网络中的网络层

第一层:

![image-20240212103859082](https://aquazone.oss-cn-guangzhou.aliyuncs.com/image-20240212103859082.png)

第二层：(后一层的输入就是前一层的输出)

![image-20240212104108920](https://aquazone.oss-cn-guangzhou.aliyuncs.com/image-20240212104108920.png)

可选：二进制判断：

![image-20240212104211394](https://aquazone.oss-cn-guangzhou.aliyuncs.com/image-20240212104211394.png)



### 9.3.2 更复杂的神经网络

![image-20240212111055523](https://aquazone.oss-cn-guangzhou.aliyuncs.com/image-20240212111055523.png)

当l=0时，公式也同样适用



### 9.3.3 描述模型

![image-20240207170947046](https://aquazone.oss-cn-guangzhou.aliyuncs.com/image-20240207170947046.png)

下面引入一些标记法来帮助描述模型：
$a_{i}^{\left( j \right)}$ 代表第$j$ 层的第 $i$ 个激活单元。${{\theta }^{\left( j \right)}}$代表从第 $j$ 层映射到第$ j+1$ 层时的权重的矩阵，例如${{\theta }^{\left( 1 \right)}}$代表从第一层映射到第二层的权重的矩阵（或许方括号，表示与层相关的量）。**其尺寸为：以第 $j+1$层的激活单元数量为行数，以第 $j$ 层的激活单元数加一为列数的矩阵**。例如：上图所示的神经网络中${{\theta }^{\left( 1 \right)}}$的尺寸为 3*4。

对于上图所示的模型，激活单元和输出分别表达为：？

$a_{1}^{(2)}=g(\Theta _{10}^{(1)}{{x}_{0}}+\Theta _{11}^{(1)}{{x}_{1}}+\Theta _{12}^{(1)}{{x}_{2}}+\Theta _{13}^{(1)}{{x}_{3}})$
$a_{2}^{(2)}=g(\Theta _{20}^{(1)}{{x}_{0}}+\Theta _{21}^{(1)}{{x}_{1}}+\Theta _{22}^{(1)}{{x}_{2}}+\Theta _{23}^{(1)}{{x}_{3}})$
$a_{3}^{(2)}=g(\Theta _{30}^{(1)}{{x}_{0}}+\Theta _{31}^{(1)}{{x}_{1}}+\Theta _{32}^{(1)}{{x}_{2}}+\Theta _{33}^{(1)}{{x}_{3}})$
${{h}_{\Theta }}(x)=g(\Theta _{10}^{(2)}a_{0}^{(2)}+\Theta _{11}^{(2)}a_{1}^{(2)}+\Theta _{12}^{(2)}a_{2}^{(2)}+\Theta _{13}^{(2)}a_{3}^{(2)})$

上面进行的讨论中只是将特征矩阵中的一行（一个训练实例）喂给了神经网络，我们需要将整个训练集都喂给我们的神经网络算法来学习模型。

我们可以知道：**每一个$a$都是由上一层所有的$x$和每一个$x$所对应？的决定的。**

（我们把这样从左到右的算法称为**前向传播算法( FORWARD PROPAGATION )**，与反向传播相对应）

把$x$, $\theta$, $a$ 分别用矩阵表示：

![image-20240207171239583](C:\Users\x\AppData\Roaming\Typora\typora-user-images\image-20240207171239583.png)

我们可以得到**$\theta \cdot X=a$**




相对于使用循环来编码，利用**向量化**的方法会使得计算更为简便。以上面的神经网络为例，试着计算第二层的值：

![image-20240208110922717](https://aquazone.oss-cn-guangzhou.aliyuncs.com/image-20240208110922717.png)

![image-20240208110936038](https://aquazone.oss-cn-guangzhou.aliyuncs.com/image-20240208110936038.png)

我们令 ${{z}^{\left( 2 \right)}}={{\theta }^{\left( 1 \right)}}x$，则 ${{a}^{\left( 2 \right)}}=g({{z}^{\left( 2 \right)}})$ ，计算后添加 $a_{0}^{\left( 2 \right)}=1$。 计算输出的值为：

![image-20240208110953611](https://aquazone.oss-cn-guangzhou.aliyuncs.com/image-20240208110953611.png)

我们令 ${{z}^{\left( 3 \right)}}={{\theta }^{\left( 2 \right)}}{{a}^{\left( 2 \right)}}$，则 $h_\theta(x)={{a}^{\left( 3 \right)}}=g({{z}^{\left( 3 \right)}})$。
这只是针对训练集中一个训练实例所进行的计算。如果我们要对整个训练集进行计算，我们需要将训练集特征矩阵进行转置，使得同一个实例的特征都在同一列里。即：
${{z}^{\left( 2 \right)}}={{\Theta }^{\left( 1 \right)}}\times {{X}^{T}}$

 ${{a}^{\left( 2 \right)}}=g({{z}^{\left( 2 \right)}})$


为了更好了了解**Neuron Networks**的工作原理，我们先把左半部分遮住：

![image-20240208111001941](https://aquazone.oss-cn-guangzhou.aliyuncs.com/image-20240208111001941.png)

右半部分其实就是以$a_0, a_1, a_2, a_3$, 按照**Logistic Regression**的方式输出$h_\theta(x)$：

![image-20240208111009166](https://aquazone.oss-cn-guangzhou.aliyuncs.com/image-20240208111009166.png)

其实神经网络就像是**logistic regression**，只不过我们把**logistic regression**中的输入向量$\left[ x_1\sim {x_3} \right]$ 变成了中间层的$\left[ a_1^{(2)}\sim a_3^{(2)} \right]$, 即:  $h_\theta(x)=g\left( \Theta_0^{\left( 2 \right)}a_0^{\left( 2 \right)}+\Theta_1^{\left( 2 \right)}a_1^{\left( 2 \right)}+\Theta_{2}^{\left( 2 \right)}a_{2}^{\left( 2 \right)}+\Theta_{3}^{\left( 2 \right)}a_{3}^{\left( 2 \right)} \right)$ 
我们**可以把$a_0, a_1, a_2, a_3$看成更为高级的特征值，也就是$x_0, x_1, x_2, x_3$的进化体，并且它们是由 $x$与$\theta$决定的，因为是梯度下降的，所以$a$是变化的，并且变得越来越厉害，所以这些更高级的特征值远比仅仅将 $x$次方厉害，也能更好的预测新数据。**
这就是神经网络相比于逻辑回归和线性回归的优势。

## 9.4 Tensorflow起步

### 9.4.1 代码推理

![image.png](https://aquazone.oss-cn-guangzhou.aliyuncs.com/20240216092216.png)

![image.png](https://aquazone.oss-cn-guangzhou.aliyuncs.com/20240216092322.png)

### 9.4.2 Tensorflow数据表示

+ **Numpy表示：**

2x3矩阵:
```python
x = np.array([[1,2,3],
			 [4,5,6]])

"""
[[1,2,3],
[4,5,6]]
"""```

4x2矩阵：
```python
x = np.array([[1,2],
			 [4,5],
    		 [7,8],
			 [8,9]])

"""
[[1,2],
[4,5],
[7,8],
[8,9]]
"""
```

```python
x = np.array([[200,17]])
#[200 17] 1x2M

x= np.array([[200],
			[17]])
# 2x1M up200,down17

x = np.array([200,17])
# 1 dimension vector
```
Tensorflow旨在处理非常大的数据集，并通过在矩阵而非一维数组中表示数据，这使它更高效。
实际上a通常是矩阵，作为x的进化体。

![image.png](https://aquazone.oss-cn-guangzhou.aliyuncs.com/20240216094448.png)
将Numpy数组传递给Tensorflow时，它喜欢将其转换为自己的内部格式，以此提高运行效率。当读回数据，可保留为张量，或将其转换回Numpy数组。

### 9.4.3 搭建一个神经网络 
+ 显式前向传播
```python
x = np.array([[200,17]])# 定义了一个输入向量x，它是一个2维的NumPy数组，包含一个样本。该样本有两个特征，分别是200和17。

# 创建了一个Dense层(layer_1)，该层有3个神经元(units=3)。激活函数选择了sigmoid函数。然后，你将输入向量x传递给layer_1，得到输出a1。
layer_1 = Dense(units=3,activation="sigmoid")
a1 = layer_1(x)

layer_2 = Dense(units=1,activation="sigmoid")
a2 = layer_2(a1)

#这段代码构建了一个具有2个输入特征、一个隐藏层（3个神经元）和一个输出层（1个神经元）的神经网络模型。通过对输入进行前向传播计算，得到最终的输出a2。
```

+ （更常见的）让Tensorflow将第一层和第二层串在一起形成神经网络
```python
layer_1 = Dense(units=3,activation="sigmoid")
layer_2 = Dense(units=1,activation="sigmoid")

model = Sequential([layer_1,layer_2])# 顺序函数Sequential()将层逐个添加到模型中
```
使用`Sequential()`函数，你可以通过将层逐个添加到模型中来构建神经网络。

通过使用顺序模型，你可以很方便地定义和管理神经网络的结构。之后，你可以使用`model.compile()`方法来**配置模型的训练参数**，以及`model.fit()`方法来**训练模型**。

+ 最简神经网络
```python
model = Sequential([
	Dense(units=3,activation="sigmoid"),
	Dense(units=1,activation="sigmoid"),
	Dense(units=1,activation="sigmoid")])

model.compile(...)# 配置了模型的训练参数，比如选择优化器、损失函数和评估指标等

x = np.array(...)
y = np.array(...)

model.fit(x,y)# 使用`model.fit(x, y)`方法对模型进行训练，以使模型学习输入x与标签y之间的关系

model.predict(x_new)# 对新的输入数据x_new进行预测，得到模型的输出结果

```
这段代码定义了一个简单的神经网络模型结构，并通过训练和预测演示了神经网络的基本使用流程。

### 9.4.4 代码的底层原理？

![image.png](https://aquazone.oss-cn-guangzhou.aliyuncs.com/20240216113253.png)

![image.png](https://aquazone.oss-cn-guangzhou.aliyuncs.com/20240216114640.png)

![image.png](https://aquazone.oss-cn-guangzhou.aliyuncs.com/20240224113222.png)

## 9.5 特征和直观理解

从本质上讲，神经网络能够通过学习得出其自身的一系列特征。在普通的逻辑回归中，我们被限制为使用数据中的原始特征$x_1,x_2,...,{{x}_{n}}$，我们虽然可以使用一些二项式项来组合这些特征，但是我们仍然受到这些原始特征的限制。在神经网络中，原始特征只是输入层，在我们上面三层的神经网络例子中，第三层也就是输出层做出的预测利用的是第二层的特征，而非输入层中的原始特征，我们可以认为第二层中的特征是神经网络通过学习后自己得出的一系列用于预测输出变量的新特征。

神经网络中，单层神经元（无中间层）的计算可用来表示逻辑运算，比如**逻辑与(AND)、逻辑或(OR)**。

举例说明：逻辑与(**AND**)；**下图中左半部分是神经网络的设计与output层表达式，右边上部分是sigmod函数****，下半部分是真值表**。

我们可以用这样的一个神经网络表示**AND** 函数：

![image-20240208111040575](https://aquazone.oss-cn-guangzhou.aliyuncs.com/image-20240208111040575.png)

其中$\theta_0 = -30, \theta_1 = 20, \theta_2 = 20$
我们的输出函数$h_\theta(x)$即为：$h_\Theta(x)=g\left( -30+20x_1+20x_2 \right)$

我们知道$g(x)$的图像是：

![image-20240208111050370](https://aquazone.oss-cn-guangzhou.aliyuncs.com/image-20240208111050370.png)



![image-20240208111057718](https://aquazone.oss-cn-guangzhou.aliyuncs.com/image-20240208111057718.png)

所以我们有：$h_\Theta(x) \approx \text{x}_1 \text{AND} \, \text{x}_2$

所以我们的：$h_\Theta(x)$

这就是**AND**函数。

接下来再介绍一个**OR**函数：

![image-20240208111109326](https://aquazone.oss-cn-guangzhou.aliyuncs.com/image-20240208111109326.png)

**OR**与**AND**整体一样，区别只在于的取值不同。

### 9.5.1 与或非

**二元逻辑运算符（BINARY LOGICAL OPERATORS）**当输入特征为布尔值（0或1）时，我们可以用一个单一的激活层可以作为二元逻辑运算符，为了表示不同的运算符，我们只需要选择不同的权重即可。

下图的神经元（三个权重分别为-30，20，20）可以被视为作用同于逻辑与（**AND**）：

![image-20240208111134157](https://aquazone.oss-cn-guangzhou.aliyuncs.com/image-20240208111134157.png)

下图的神经元（三个权重分别为-10，20，20）可以被视为作用等同于逻辑或（**OR**）：

![image-20240208111140736](https://aquazone.oss-cn-guangzhou.aliyuncs.com/image-20240208111140736.png)

下图的神经元（两个权重分别为 10，-20）可以被视为作用等同于逻辑非（**NOT**）：

![image-20240208111148363](https://aquazone.oss-cn-guangzhou.aliyuncs.com/image-20240208111148363.png)

我们可以利用神经元来组合成更为复杂的神经网络以实现更复杂的运算。例如我们要实现**XNOR** 功能（输入的两个值必须一样，均为1或均为0），即 $\text{XNOR}=( \text{x}_1\, \text{AND}\, \text{x}_2 )\, \text{OR} \left( \left( \text{NOT}\, \text{x}_1 \right) \text{AND} \left( \text{NOT}\, \text{x}_2 \right) \right)$
首先构造一个能表达$\left( \text{NOT}\, \text{x}_1 \right) \text{AND} \left( \text{NOT}\, \text{x}_2 \right)$部分的神经元：

![image-20240208111159813](https://aquazone.oss-cn-guangzhou.aliyuncs.com/image-20240208111159813.png)

然后将表示 **AND** 的神经元和表示$\left( \text{NOT}\, \text{x}_1 \right) \text{AND} \left( \text{NOT}\, \text{x}_2 \right)$的神经元以及表示 OR 的神经元进行组合：

![image-20240208111209054](https://aquazone.oss-cn-guangzhou.aliyuncs.com/image-20240208111209054.png)

我们就得到了一个能实现 $\text{XNOR}$ 运算符功能的神经网络。

按这种方法我们可以逐渐构造出越来越复杂的函数，也能得到更加厉害的特征值。

这就是神经网络的厉害之处。

## 9.6 强人工智能

+ AI
	+ ANI(artificial narrow intelligence)
		+ E.g.,smart speaker,self-driving car,web search,AI in farming and factories(already sidely used)
	+ AGI(artificial general intelligence)
		+ E.g.,Do anything human can do(not so easy)

我们构建的神经网络比人的简单不少，更何况在今日我们对脑的了解仍然不足。仅仅试图模拟人脑作为通向AGI的途径将是一条极其困难的道路。

## 9.5 多类分类

当我们有不止两种分类时（也就是$y=1,2,3….$），比如以下这种情况，该怎么办？如果我们要训练一个神经网络算法来识别路人、汽车、摩托车和卡车，在输出层我们应该有4个值。例如，第一个值为1或0用于预测是否是行人，第二个值用于判断是否为汽车。

输入向量$x$有三个维度，两个中间层，输出层4个神经元分别用来表示4类，也就是每一个数据在输出层都会出现${{\left[ a\text{ }b\text{ }c\text{ }d \right]}^{T}}$，且$a,b,c,d$中仅有一个为1，表示当前类。下面是该神经网络的可能结构示例：

![image-20240208111244765](https://aquazone.oss-cn-guangzhou.aliyuncs.com/image-20240208111244765.png)

![image-20240208111252007](https://aquazone.oss-cn-guangzhou.aliyuncs.com/image-20240208111252007.png)

神经网络算法的输出结果为四种可能情形之一：

![image-20240208111259805](https://aquazone.oss-cn-guangzhou.aliyuncs.com/image-20240208111259805.png)
