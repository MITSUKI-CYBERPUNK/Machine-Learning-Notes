# 4 Python数据分析之Matplotlib

 [matplotlib文档](https://matplotlib.org/stable/api/index.html)

## 4.0 速查表

![](https://aquazone.oss-cn-guangzhou.aliyuncs.com/image-20240125203630947.png)

## 4.1 Matplotlib基础

**matplotlib**是一个**Python**的 2D 图形包。pyplot封装了很多画图的函数

导入相关的包：


```python
import matplotlib.pyplot as plt
import numpy as np
```

``matplotlib.pyplot``包含一系列类似**MATLAB**中绘图函数的相关函数。每个``matplotlib.pyplot``中的函数对当前的图像进行一些修改，例如：产生新的图像，在图像中产生新的绘图区域，在绘图区域中画线，给绘图加上标记，等等......``matplotlib.pyplot``会自动记住当前的图像和绘图区域，因此这些函数会直接作用在当前的图像上。

在实际的使用过程中，常常以``plt``作为``matplotlib.pyplot``的省略。

### 4.1.1 plt.show()函数 

默认情况下，``matplotlib.pyplot``不会直接显示图像，只有调用``plt.show()``函数时，图像才会显示出来。

**plt.show()``默认是在新窗口打开一幅图像，并且提供了对图像进行操作的按钮。**

不过在``ipython``命令中，我们可以将它插入``notebook``中，并且不需要调用``plt.show()``也可以显示：

* ``%matplotlib notebook``
* ``%matplotlib inline``

不过在实际写程序中，我们还是习惯调用``plt.show()``函数将图像显示出来。


```python
%matplotlib inline #魔术命令
```

### 4.1.2 plt.plot()函数 

#### 4.1.2.1 例子

``plt.plot()``函数可以用来绘线型图：


```python
plt.plot([1,2,3,4]) #默认以列表的索引作为x，输入的是y
plt.ylabel('y')
plt.xlabel("x轴") #设定标签，使用中文的话后面需要再设定
```


    Text(0.5, 0, 'x轴')


![image-20240123154546719](https://aquazone.oss-cn-guangzhou.aliyuncs.com/image-20240123154546719.png)
    

#### 4.1.2.2 基本用法

``plot``函数基本的用法：

指定x和y

* ``plt.plot(x,y)``

默认参数，x为0~N-1

* ``plt.plot(y)``

因此，在上面的例子中，我们没有给定``x``的值，所以其默认值为``[0,1,2,3]``

传入``x``和``y``：


```python
plt.plot([1,2,3,4],[1,4,9,16])
plt.show() #相当于打印的功能，下面不会再出现内存地址
```

![image-20240123155235072](https://aquazone.oss-cn-guangzhou.aliyuncs.com/image-20240123155235072.png)
    




#### 4.1.2.3 字符参数

和**MATLAB**中类似，我们还可以用字符来指定绘图的格式：

表示颜色的字符参数有：

|    字符 |          颜色 |
| ------: | ------------: |
| ``'b'`` |    蓝色，blue |
| ``'g'`` |   绿色，green |
| ``'r'`` |     红色，red |
| ``'c'`` |    青色，cyan |
| ``'m'`` | 品红，magenta |
| ``'y'`` |  黄色，yellow |
| ``'k'`` |   黑色，black |
| ``'w'`` |   白色，white |

表示类型的字符参数有：

|     字符 |       类型 |     字符 |      类型 |
| -------: | ---------: | -------: | --------: |
|  ``'-'`` |       实线 | ``'--'`` |      虚线 |
| ``'-'.`` |     虚点线 |  ``':'`` |      点线 |
|  ``'.'`` |         点 |  ``','`` |    像素点 |
|  ``'o'`` |       圆点 |  ``'v'`` |  下三角点 |
|  ``'^'`` |   上三角点 |  ``'<'`` |  左三角点 |
|  ``'>'`` |   右三角点 |  ``'1'`` |  下三叉点 |
|  ``'2'`` |   上三叉点 |  ``'3'`` |  左三叉点 |
|  ``'4'`` |   右三叉点 |  ``'s'`` |    正方点 |
|  ``'p'`` |     五角点 |  ``'*'`` |    星形点 |
|  ``'h'`` |  六边形点1 |  ``'H'`` | 六边形点2 |
|  ``'+'`` |     加号点 |  ``'x'`` |    乘号点 |
|  ``'D'`` | 实心菱形点 |  ``'d'`` |  瘦菱形点 |
|  ``'_'`` |     横线点 |          |           |

例如我们要画出红色圆点：


```python
plt.plot([1,2,3,4],[1,4,9,16],"ro") #也可以是or，没顺序要求
plt.show()
```


![image-20240123155542708](https://aquazone.oss-cn-guangzhou.aliyuncs.com/image-20240123155542708.png)
    

可以看出，有两个点在图像的边缘，因此，我们需要改变轴的显示范围。



#### 4.1.2.4 显示范围

与**MATLAB**类似，这里可以使用``axis``函数指定坐标轴显示的范围：
```python
plt.axis([xmin, xmax, ymin, ymax])
```


```python
plt.plot([1,2,3,4],[1,4,9,16],"g*")
plt.axis([0,6,0,20])
plt.show()
```

![image-20240123155631341](https://aquazone.oss-cn-guangzhou.aliyuncs.com/image-20240123155631341.png)
    




#### 4.1.2.5 传入``Numpy``数组 

之前我们传给``plot``的参数都是列表，事实上，向``plot``中传入``numpy``数组是更常用的做法。事实上，如果传入的是列表，``matplotlib``会在内部将它转化成数组再进行处理：

在一个图里面画多条线：


```python
t = np.arange(0.,5.,0.2) #左闭右开从0到5间隔0.2
plt.plot(t,t,"r--",
        t,t**2,"bs",
        t,t**3,"g^")
plt.show()
```

![image-20240123155802223](https://aquazone.oss-cn-guangzhou.aliyuncs.com/image-20240123155802223.png)
    




#### 4.1.2.6 传入多组数据 

事实上，在上面的例子中，我们不仅仅向``plot``函数传入了数组，还传入了多组``(x,y,format_str)``参数，它们在同一张图上显示。

这意味着我们不需要使用多个``plot``函数来画多组数组，只需要可以将这些组合放到一个``plot``函数中去即可。



#### 4.1.2.7 线条属性 

之前提到，我们可以用**字符串**来控制线条的属性，事实上还可以用**关键词**来改变线条的性质，例如``linewidth``可以改变线条的宽度，``color``可以改变线条的颜色：


```python
x = np.linspace(-np.pi,np.pi)
y = np.sin(x)
plt.plot(x,y,linewidth = 4.0,color = 'r') #细节调整的两个方式
plt.show()
```

![image-20240123160023088](https://aquazone.oss-cn-guangzhou.aliyuncs.com/image-20240123160023088.png)
    




#### 4.1.2.8 使用plt.plot()的返回值来设置线条属性

``plot``函数返回一个``Line2D``对象组成的列表，每个对象代表输入的一对组合，例如：

* line1,line2 为两个 Line2D 对象
```python
line1, line2 = plt.plot(x1, y1, x2, y2)
```
* 返回3个 Line2D 对象组成的列表
```python
lines = plt.plot(x1, y1, x2, y2, x3, y3)
```

我们可以使用这个返回值来对线条属性进行设置：


```python
line1,line2 = plt.plot(x,y,"r-",x,y+1,"g-")
line1.set_antialiased(False)  #抗锯齿
plt.show()
```


![image-20240123161045947](https://aquazone.oss-cn-guangzhou.aliyuncs.com/image-20240123161045947.png)
    



```python
line = plt.plot(x,y,"r-",x,y+1,"g-")
line[1].set_antialiased(False) #列表
plt.show()
```

![image-20240123161150448](https://aquazone.oss-cn-guangzhou.aliyuncs.com/image-20240123161150448.png)
    




#### 4.1.2.9 plt.setp() 修改线条性质

更方便的做法是使用``plt``的``setp``函数：

```python
line = plt.plot(x,y)
#plt.setp(line, color = 'g',linewidth = 4)
plt.setp(line,"color",'r',"linewidth",4) #matlab风格
```

![image-20240123161521568](https://aquazone.oss-cn-guangzhou.aliyuncs.com/image-20240123161521568.png)



#### 4.1.2.10 子图 

``figure()``函数会产生一个指定编号为``num``的图：

```python
plt.figure(num)
```
这里，``figure(1)``其实是可以省略的，因为默认情况下``plt``会自动产生一幅图像。

使用``subplot``可以在一幅图中生成多个子图，其参数为：
```python
plt.subplot(numrows, numcols, fignum)
```
当``numrows * numncols < 10``时，中间的逗号可以省略，因此``plt.subplot(211)``就相当于``plt.subplot(2,1,1)``。


```python
def f(t):
    return np.exp(-t)*np.cos(2*np.pi*t)

t1 = np.arange(0.0,5.0,0.1)
t2 = np.arange(0.0,4.0,0.02)

plt.figure(figsize = (10,6))
plt.subplot(211)
plt.plot(t1,f(t1),"bo",t2,f(t2),'k') #子图1上有两条线

plt.subplot(212)
plt.plot(t2,np.cos(2*np.pi*t2),"r--")
plt.show()
```

![image-20240123161658051](https://aquazone.oss-cn-guangzhou.aliyuncs.com/image-20240123161658051.png)
    

## 4.2 电影数据绘图

**在了解绘图的基础知识之后，我们可以对电影数据进行可视化分析。**


```python
import warnings
warnings.filterwarnings("ignore") #关闭一些可能出现但对数据分析并无影响的警告
```


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```


```python
plt.rcParams["font.sans-serif"] = ["SimHei"] #解决中文字符乱码的问题
plt.rcParams["axes.unicode_minus"] = False #正常显示负号
```


```python
df = pd.read_excel(r"C:\Users\Lovetianyi\Desktop\python\作业5\movie_data3.xlsx", index_col = 0)
```


```python
df[:5]
```



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>年代</th>
      <th>产地</th>
      <th>名字</th>
      <th>投票人数</th>
      <th>类型</th>
      <th>上映时间</th>
      <th>时长</th>
      <th>评分</th>
      <th>首映地点</th>
      <th>评分等级</th>
      <th>热门程度</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1994</td>
      <td>美国</td>
      <td>肖申克的救赎</td>
      <td>692795</td>
      <td>剧情/犯罪</td>
      <td>1994-09-10 00:00:00</td>
      <td>142</td>
      <td>9.6</td>
      <td>多伦多电影节</td>
      <td>A</td>
      <td>A</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1957</td>
      <td>美国</td>
      <td>控方证人</td>
      <td>42995</td>
      <td>剧情/悬疑/犯罪</td>
      <td>1957-12-17 00:00:00</td>
      <td>116</td>
      <td>9.5</td>
      <td>美国</td>
      <td>A</td>
      <td>A</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1997</td>
      <td>意大利</td>
      <td>美丽人生</td>
      <td>327855</td>
      <td>剧情/喜剧/爱情</td>
      <td>1997-12-20 00:00:00</td>
      <td>116</td>
      <td>9.5</td>
      <td>意大利</td>
      <td>A</td>
      <td>A</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1994</td>
      <td>美国</td>
      <td>阿甘正传</td>
      <td>580897</td>
      <td>剧情/爱情</td>
      <td>1994-06-23 00:00:00</td>
      <td>142</td>
      <td>9.4</td>
      <td>洛杉矶首映</td>
      <td>A</td>
      <td>A</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1993</td>
      <td>中国大陆</td>
      <td>霸王别姬</td>
      <td>478523</td>
      <td>剧情/爱情/同性</td>
      <td>1993-01-01 00:00:00</td>
      <td>171</td>
      <td>9.4</td>
      <td>香港</td>
      <td>A</td>
      <td>A</td>
    </tr>
  </tbody>
</table>


## 4.2.1 柱状图

绘制每个国家或地区的电影数量的柱状图：

**柱状图**(bar chart)，是一种以长方形的长度为变量的表达图形的统计报告图，由一系列高度不等的纵向条纹表示数据分布的情况，用来比较两个或以上的价值（不同时间或者不同条件），只有一个变量，通常利用较小的数据集分析。柱状图亦可横向排列，或用多维方式表达。


```python
data = df["产地"].value_counts()
data
```


    美国      11714
    日本       5006
    中国大陆     3791
    中国香港     2847
    法国       2787
    英国       2658
    其他       1883
    韩国       1342
    德国       1021
    意大利       741
    加拿大       709
    中国台湾      618
    俄罗斯       476
    西班牙       443
    印度        356
    澳大利亚      295
    泰国        294
    丹麦        197
    瑞典        187
    波兰        181
    荷兰        151
    比利时       137
    墨西哥       117
    阿根廷       113
    巴西         99
    Name: 产地, dtype: int64




```python
x = data.index
y = data.values

plt.figure(figsize = (10,6)) #设置图片大小
plt.bar(x,y,color = "g") #绘制柱状图，表格给的数据是怎样就怎样，不会自动排序

plt.title("各国家或地区电影数量", fontsize = 20) #设置标题
plt.xlabel("国家或地区",fontsize = 18) 
plt.ylabel("电影数量") #对横纵轴进行说明
plt.tick_params(labelsize = 14) #设置标签字体大小
plt.xticks(rotation = 90) #标签转90度

for a,b in zip(x,y): #数字直接显示在柱子上（添加文本）
    #a:x的位置，b:y的位置，加上10是为了展示位置高一点点不重合，
    #第二个b:显示的文本的内容,ha,va:格式设定,center居中,top&bottom在上或者在下,fontsize:字体指定
    plt.text(a,b+10,b,ha = "center",va = "bottom",fontsize = 10) 

#plt.grid() #画网格线，有失美观因而注释点

plt.show()
```

![image-20240123162855896](https://aquazone.oss-cn-guangzhou.aliyuncs.com/image-20240123162855896.png)



### 4.2.2 曲线图

绘制每年上映的电影数量的曲线图：

**曲线图**又称折线图，是利用曲线的升，降变化来表示被研究现象发展趋势的一种图形。它在分析研究社会经济现象的发展变化、依存关系等方面具有重要作用。

绘制曲线图时，如果是某一现象的时间指标，应将时间绘在坐标的横轴上，指标绘在坐标的纵轴上。如果是两个现象依存关系的显示，可以将表示原因的指标绘在横轴上，表示结果的指标绘在纵轴上，同时还应注意整个图形的长宽比例。

### 1888-2015年 


```python
data = df["年代"].value_counts()
data = data.sort_index()[:-2] #排除掉2016年以后的数据，共两条
data
```


    1888       2
    1890       1
    1892       1
    1894       3
    1895       8
            ... 
    2011    1845
    2012    2018
    2013    1977
    2014    1867
    2015    1569
    Name: 年代, Length: 125, dtype: int64




```python
x = data.index
y = data.values

plt.plot(x,y,color = 'b')
plt.title("每年电影数量",fontsize = 20)
plt.ylabel("电影数量",fontsize = 18)
plt.xlabel("年份",fontsize = 18)

for (a,b) in zip(x[::10],y[::10]): #每隔10年进行数量标记，防止过于密集
    plt.text(a,b+10,b,ha = "center", va = "bottom", fontsize = 10)
    
#标记特殊点如极值点，xy设置箭头尖的坐标，xytext注释内容起始位置，arrowprops对箭头设置，传字典，facecolor填充颜色，edgecolor边框颜色
plt.annotate("2012年达到最大值", xy = (2012,data[2012]), xytext = (2025,2100), arrowprops = dict(facecolor = "black",edgecolor = "red"))

#纯文本注释内容，例如注释增长最快的地方
plt.text(1980,1000,"电影数量开始快速增长")
plt.show()
```

![image-20240123163449116](https://aquazone.oss-cn-guangzhou.aliyuncs.com/image-20240123163449116.png)
    利用matplotlib绘图已经较为成熟了


对于这幅图形，我们使用``xlabel, ylabel, title, text``方法设置了文字，其中：

* ``xlabel``: x轴标注
* ``ylabel``: y轴标注
* ``title``: 图形标题
* ``text``: 在指定位置(坐标)放入文字

输入特殊符号支持使用``Tex``语法，用``$<some Text code>$``隔开。

除了使用``text``在指定位置标上文字之外，还可以使用``annotate``进行注释，``annotate``主要有两个参数：

* ``xy``: 注释位置
* ``xytext``: 注释文字位置



从上图可以看出，电影数量是逐年增加的，增长的趋势在2000年都变得飞快 



### 4.2.3 饼图 

根据电影的长度绘制饼图：

**饼图**英文学名为Sector Graph，又名Pie Graph。常用于统计学模块。2D饼图为圆形，手画时，常用圆规作图。

仅排列在工作表的一列或一行中的数据可以绘制到饼图中。饼图显示一个数据系列（数据系列：在图表中绘制的相关数据点，这些数据源自数据表的行或列。图表中的每个数据系列具有唯一的颜色或团并且在图表中的图例中表示。可以在图表中绘制一个或多个数据系列。饼图只有一个数据系列。）中各项的大小与各项总和的比例。饼图中的数据点（数据点：在图表中绘制的单个值，这些值由条形，柱形，折线，饼图或圆环图的扇面、圆点和其他被称为数据标记的图形表示。相同颜色的数据标记组成一个数据系列。）显示为整个饼图的百分比。

**函数原型：**

```python
pie(x, explode = None, labels = None, colors = None, autopct = None, pctdistance = 0.6,
    shadow = False, labeldistance = 1.1, startangle = None, radius = None)
```

**参数：**  
**x：** (每一块）的比例，如果sum(x)>1会使用sum(x)归一化  
**labels：** （每一块）饼图外侧显示的说明文字  
**explode：** （每一块）离开中心距离  
**startangle：** 起始绘制角度，默认图是从x轴正方向逆时针画起，如设定=90则从y轴正方向画起   
**shadow：** 是否阴影  
**labeldistance：** label绘制位置，相对于半径的比例，如<1则绘制在饼图内侧  
**autopct：** 控制饼图内百分比设置，可以使用format字符串或者format function  
**'%1.1f'：** 指小数点前后位数（没有用空格补齐）  
**pctdistance：** 类似于labeldistance，指定autopct的位置刻度  
**radius：** 控制饼图半径

**返回值：**  
如果没有设置autopct，返回(patches,texts)  
如果设置autopct，返回(patches,texts,autotexts)


```python
data = pd.cut(df["时长"], [0,60,90,110,1000]).value_counts() #数据离散化
data
```


    (90, 110]      13201
    (0, 60]         9884
    (60, 90]        7661
    (110, 1000]     7417
    Name: 时长, dtype: int64




```python
y = data.values
y = y/sum(y) #归一化，不进行的话系统会自动进行

plt.figure(figsize = (7,7))
plt.title("电影时长占比",fontsize = 15)
patches,l_text,p_text = plt.pie(y, labels = data.index, autopct = "%.1f %%", colors = "bygr", startangle = 90)

for i in p_text: #通过返回值设置饼图内部字体
    i.set_size(15)
    i.set_color('w')

for i in l_text: #通过返回值设置饼图外部字体
    i.set_size(15)
    i.set_color('r')
    
plt.legend() #图例
plt.show()
```

![image-20240123163810642](https://aquazone.oss-cn-guangzhou.aliyuncs.com/image-20240123163810642.png)
    

### 4.2.4 频率直方图

根据电影的评分绘制频率直方图：

**直方图**(Histogram)又称质量分布图。是一种统计报告图。由一系列高度不等的纵向条纹或线段表示数据分布的情况。一般用横轴表示数据类型，纵轴表示分布情况。

直方图是数值数据分布的精确图形表示。这是一个连续变量（定量变量）的概率分布的估计，并且被卡尔·皮尔逊(Karl Pearson)首先引入。它是一种条形图。为了构建直方图，第一步是将值的范围分段，即将整个值的范围分成一系列间隔，然后计算每个间隔中有多少值。这些值通常被指定为连续的，不重叠的变量间隔。间隔必须相邻，并且通常是（但不是必须的）相等的大小。

直方图也可以被归一化以显示“相对频率”。然后，它显示了属于几个类别中每个案例的比例，其高度等于1。


```python
plt.figure(figsize = (10,6))
plt.hist(df["评分"], bins = 20, edgecolor = 'k',alpha = 0.5)
plt.show()
```


![image-20240123164305903](https://aquazone.oss-cn-guangzhou.aliyuncs.com/image-20240123164305903.png)
    


hist的参数非常多，但常用的就这六个，只有第一个是必须的，后面可选

**arr**: 需要计算直方图的一维数组

**bins**: 直方图的柱数，可选项，默认为10

**normed**: 是否将得到的直方图向量归一化。默认为0

**facecolor**: 直方图颜色

**edgecolor**: 直方图边框颜色

**alpha**: 透明度

**histtype**: 直方图类型，"bar", "barstacked", "step", "stepfilled"

返回值：

**n**: 直方图向量，是否归一化由参数normed设定

**bins**: 返回各个bin的区间范围

**patches**: 返回每一个bin里面包含的数据，是一个list

从上图我们可以发现，电影的评分是服从一个右偏的正态分布的。 





### 4.2.5 双轴图

```python
from scipy.stats import norm #获取正态分布密度函数
```


```python
fig = plt.figure(figsize = (10,8))
ax1 = fig.add_subplot(111) #确认子图
n,bins,patches = ax1.hist(df["评分"],bins = 100, color = 'm') #bins默认是10

ax1.set_ylabel("电影数量",fontsize = 15)
ax1.set_xlabel("评分",fontsize = 15)
ax1.set_title("频率分布图",fontsize = 20)

#准备拟合
y = norm.pdf(bins,df["评分"].mean(),df["评分"].std()) #bins,mu,sigma
ax2 = ax1.twinx() #双轴
ax2.plot(bins,y,"b--")
ax2.set_ylabel("概率分布",fontsize = 15)
plt.show()
```

![image-20240123165422619](https://aquazone.oss-cn-guangzhou.aliyuncs.com/image-20240123165422619.png)



### 4.2.6 散点图

根据电影时长和电影评分绘制散点图：

用两组数据构成多个坐标点，考察坐标点的分布，判断两变量之间是否存在某种关联或总结坐标点的分布模式。散点图将序列显示为一组点。值由点在图表中的位置表示。类别由图表中的不同标记表示。散点图通常用于比较跨类别的聚合数据。


```python
x = df["时长"][::100]
y = df["评分"][::100] #解决数据冗杂的问题

plt.figure(figsize = (10,6))
plt.scatter(x,y,color = 'c',marker = 'p',label = "评分")
plt.legend() #图例
plt.title("电影时长与评分散点图",fontsize = 20)
plt.xlabel("时长",fontsize = 18)
plt.ylabel("评分",fontsize = 18)
plt.show()
```

![image-20240123165612585](https://aquazone.oss-cn-guangzhou.aliyuncs.com/image-20240123165612585.png)
    




由于我们的数据量过大，所以画出来的图非常冗杂

可以发现，大部分的电影时长还是集中在100附近，评分大多在7分左右



#### 4.2.6.1 marker属性

设置散点的形状

| **marker** | **description** | **描述**  |
| ---------- | --------------- | --------- |
| "."        | point           | 点        |
| ","        | pixel           | 像素      |
| "o"        | circle          | 圈        |
| "v"        | triangle_down   | 倒三角形  |
| "^"        | triangle_up     | 正三角形  |
| "<"        | triangle_left   | 左三角形  |
| ">"        | triangle_right  | 右三角形  |
| "1"        | tri_down        | tri_down  |
| "2"        | tri_up          | tri_up    |
| "3"        | tri_left        | tri_left  |
| "4"        | tri_right       | tri_right |
| "8"        | octagon         | 八角形    |
| "s"        | square          | 正方形    |
| "p"        | pentagon        | 五角      |
| "\*"       | star            | 星星      |
| "h"        | hexagon1        | 六角1     |
| "H"        | hexagon2        | 六角2     |
| "+"        | plus            | 加号      |
| "x"        | x               | x号       |
| "D"        | diamond         | 钻石      |
| "d"        | thin_diamon     | 细钻      |
| "\|"       | vline           | v线       |
| "\_"       | hline           | H线       |



### 4.2.7 箱型图

绘制各个地区的评分箱型图

**箱型图**（Box-plot）又称为盒须图，盒式图或箱型图，是一种用作显示一组数据分散情况资料的统计图。因形状如箱子而得名。在各种领域中也经常被使用，常见于品质管理。它主要用于反映原始数据分布的特征，还可以进行多组数据分布特征的比较。箱线图的绘制方法是：先找出一组数据的中位数，两个四分位数，上下边缘线；然后，连接两个四分位数画出箱子；再将上下边缘线与箱子相连接，中位数在箱子中间。

箱型图（Box-plot）又称为盒须图，盒式图或箱型图，是一种用作显示一组数据分散情况资料的统计图。因形状如箱子而得名。在各种领域中也经常被使用，常见于品质管理。它主要用于反映原始数据分布的特征，还可以进行多组数据分布特征的比较。箱线图的绘制方法是：先找出一组数据的中位数，两个四分位数，上下边缘线；然后，连接两个四分位数画出箱子；再将上下边缘线与箱子相连接，中位数在箱子中间。

箱型图（Box-plot）又称为盒须图，盒式图或箱型图，是一种用作显示一组数据分散情况资料的统计图。因形状如箱子而得名。在各种领域中也经常被使用，常见于品质管理。它主要用于反映原始数据分布的特征，还可以进行多组数据分布特征的比较。箱线图的绘制方法是：先找出一组数据的中位数，两个四分位数，上下边缘线；然后，连接两个四分位数画出箱子；再将上下边缘线与箱子相连接，中位数在箱子中间。

![375d3e018a4a35cde9d3cb178ce43ce](https://aquazone.oss-cn-guangzhou.aliyuncs.com/375d3e018a4a35cde9d3cb178ce43ce.jpg)



**一般计算过程**

（ 1 ）计算上四分位数（ Q3 ），中位数，下四分位数（ Q1 ）

（ 2 ）计算上四分位数和下四分位数之间的差值，即四分位数差（IQR, interquartile range）Q3-Q1

（ 3 ）绘制箱线图的上下范围，上限为上四分位数，下限为下四分位数。在箱子内部中位数的位置绘制横线

（ 4 ）大于上四分位数1.5倍四分位数差的值，或者小于下四分位数1.5倍四分位数差的值，划为异常值（outliers）

（ 5 ）异常值之外，最靠近上边缘和下边缘的两个值处，画横线，作为箱线图的触须

（ 6 ）极端异常值，即超出四分位数差3倍距离的异常值，用实心点表示；较为温和的异常值，即处于1.5倍-3倍四分位数差之间的异常值，用空心点表示

（ 7 ）为箱线图添加名称，数轴等

**参数详解**

```python
plt.boxplot(x,notch=None,sym=None,vert=None,
    whis=None,positions=None,widths=None,
    patch_artist=None,meanline=None,showmeans=None,
    showcaps=None,showbox=None,showfliers=None,
    boxprops=None,labels=None,flierprops=None,
    medianprops=None,meanprops=None,
    capprops=None,whiskerprops=None,)
```
**x: 指定要绘制箱线图的数据；**

**notch: 是否是凹口的形式展现箱线图，默认非凹口；**

**sym: 指定异常点的形状，默认为+号显示；**

**vert: 是否需要将箱线图垂直摆放，默认垂直摆放；**

**whis: 指定上下须与上下四分位的距离，默认为为1.5倍的四分位差；**

**positions: 指定箱线图的位置，默认为[0,1,2...]；**

**widths: 指定箱线图的宽度，默认为0.5；**

**patch_artist: 是否填充箱体的颜色；**

**meanline:是否用线的形式表示均值，默认用点来表示；**

**showmeans: 是否显示均值，默认不显示；**

**showcaps: 是否显示箱线图顶端和末端的两条线，默认显示；**

**showbox: 是否显示箱线图的箱体，默认显示；**

**showfliers: 是否显示异常值，默认显示；**

**boxprops: 设置箱体的属性，如边框色，填充色等；**

**labels: 为箱线图添加标签，类似于图例的作用；**

**filerprops: 设置异常值的属性，如异常点的形状、大小、填充色等；**

**medainprops: 设置中位数的属性，如线的类型、粗细等**

**meanprops: 设置均值的属性，如点的大小，颜色等；**

**capprops: 设置箱线图顶端和末端线条的属性，如颜色、粗细等；**

**whiskerprops: 设置须的属性，如颜色、粗细、线的类型等**



美国电影评分的箱线图:


```python
data = df[df.产地 == "美国"]["评分"]

plt.figure(figsize = (10,6))
plt.boxplot(data,whis = 2,flierprops = {"marker":'o',"markerfacecolor":"r","color":'k'}
           ,patch_artist = True, boxprops = {"color":'k',"facecolor":"#66ccff"})
plt.title("美国电影评分",fontsize = 20)
plt.show()
```

![image-20240123165958467](https://aquazone.oss-cn-guangzhou.aliyuncs.com/image-20240123165958467.png)
    



多组数据箱线图 


```python
data1 = df[df.产地 == "中国大陆"]["评分"]
data2 = df[df.产地 == "日本"]["评分"]
data3 = df[df.产地 == "中国香港"]["评分"]
data4 = df[df.产地 == "英国"]["评分"]
data5 = df[df.产地 == "法国"]["评分"]

plt.figure(figsize = (12,8))
plt.boxplot([data1,data2,data3,data4,data5],labels = ["中国大陆","日本","中国香港","英国","法国"],
           whis = 2,flierprops = {"marker":'o',"markerfacecolor":"r","color":'k'}
           ,patch_artist = True, boxprops = {"color":'k',"facecolor":"#66ccff"},
           vert = False)

ax = plt.gca() #获取当时的坐标系
ax.patch.set_facecolor("gray") #设置坐标系背景颜色
ax.patch.set_alpha(0.3) #设置背景透明度

plt.title("电影评分箱线图",fontsize = 20)
plt.show()
```

![image-20240123170040437](https://aquazone.oss-cn-guangzhou.aliyuncs.com/image-20240123170040437.png)

​    

**通过vert属性可以把图旋转过来**



### 4.2.9 热力图


```python
data = df[["投票人数","评分","时长"]]
data[:5]
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>投票人数</th>
      <th>评分</th>
      <th>时长</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>692795</td>
      <td>9.6</td>
      <td>142</td>
    </tr>
    <tr>
      <th>1</th>
      <td>42995</td>
      <td>9.5</td>
      <td>116</td>
    </tr>
    <tr>
      <th>2</th>
      <td>327855</td>
      <td>9.5</td>
      <td>116</td>
    </tr>
    <tr>
      <th>3</th>
      <td>580897</td>
      <td>9.4</td>
      <td>142</td>
    </tr>
    <tr>
      <th>4</th>
      <td>478523</td>
      <td>9.4</td>
      <td>171</td>
    </tr>
  </tbody>
</table>


pandas本身也封装了画图函数 

我们可以画出各个属性之间的散点图，对角线是分布图 


```python
%pylab inline 
#魔术命令，让图像直接展示在notebook里面
result = pd.plotting.scatter_matrix(data[::100],diagonal = "kde",color = 'k',alpha = 0.3,figsize = (10,10)) 
#diagonal = hist:对角线上显示的是数据集各个特征的直方图/kde:数据集各个特征的核密度估计
```

![image-20240123170255598](https://aquazone.oss-cn-guangzhou.aliyuncs.com/image-20240123170255598.png)
    



### 4.2.10 相关系数矩阵图

现在我们来画电影时长，投票人数，评分的一个相关系数矩阵图:

**seaborn**是一个精简的python库，可以创建具有统计意义的图表，能理解pandas的DataFrame类型


```python
import seaborn as sns

corr = data.corr() #获取相关系数
corr = abs(corr) #取绝对值

fig = plt.figure(figsize = (10,8))
ax = fig.add_subplot(111)

ax = sns.heatmap(corr,vmax = 1,vmin = 0,annot = True,annot_kws = {"size":13,"weight":"bold"},linewidths = 0.05)

plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)

plt.show()
```


![image-20240123170413299](https://aquazone.oss-cn-guangzhou.aliyuncs.com/image-20240123170413299.png)



#### 4.2.10.1 参数详解 

```python
sns.heatmap(data,vmin=None,vmax=None,cmap=None,center=None,robust=False,annot=None,fmt='.2g',annot_kws=None,linewidths=0,linecolor='white',cbar=True,cbar_kws=None,cbar_ax=None,square=False,xticklabels='auto',yticklabels='auto',mask=None,ax=None,**kwargs,)
```

（ 1 ）热力图输入数据参数：

data:矩阵数据集，可以是numpy的数组（array），也可以是pandas的DataFrame。如果是DataFrame，则df的index/column信息会分别对应到heatmap的columns和rows，即pt.index是热力图的行标，pt.columns是热力图的列标。

（ 2 ）热力图矩阵块颜色参数：

vmax,vmin:分别是热力图的颜色取值最大和最小范围，默认是根据data数据表里的取值确定。cmap:从数字到色彩空间的映射，取值是matplotlib包里的colormap名称或颜色对象，或者表示颜色的列表；改参数默认值：根据center参数设定。center:数据表取值有差异时，设置热力图的色彩中心对齐值；通过设置center值，可以调整生成的图像颜色的整体深浅；设置center数据时，如果有数据溢出，则手动设置的vmax、vmin会自动改变。robust:默认取值False，如果是False，且没设定vmin和vmax的值。

（ 3 ）热力图矩阵块注释参数：

annot(annotate的缩写):默认取值False；如果是True，在热力图每个方格写入数据；如果是矩阵，在热力图每个方格写入该矩阵对应位置数据。fmt:字符串格式代码，矩阵上标识数字的数据格式，比如保留小数点后几位数字。annot_kws:默认取值False；如果是True，设置热力图矩阵上数字的大小颜色字体，matplotlib包text类下的字体设置；

（ 4 ）热力图矩阵块之间间隔及间隔线参数：

linewidth:定义热力图里“表示两两特征关系的矩阵小块”之间的间隔大小。linecolor:切分热力图上每个矩阵小块的线的颜色，默认值是"white"。

（ 5 ）热力图颜色刻度条参数：

cbar:是否在热力图侧边绘制颜色进度条，默认值是True。cbar_kws:热力图侧边绘制颜色刻度条时，相关字体设置，默认值是None。cbar_ax：热力图侧边绘制颜色刻度条时，刻度条位置设置，默认值是None

（ 6 ）

square:设置热力图矩阵小块形状，默认值是False。xticklabels,yticklabels:xticklabels控制每列标签名的输出；yticklabels控制每行标签名的输出。默认值是auto。如果是True，则以DataFrame的列名作为标签名。如果是False，则不添加行标签名。如果是列表，则标签名改为列表中给的内容。如果是整数K，则在图上每隔K个标签进行一次标注。