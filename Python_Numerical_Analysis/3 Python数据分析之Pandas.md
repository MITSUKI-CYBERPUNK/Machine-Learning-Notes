# 3 Python数据分析之Pandas 

[pandas文档](https://pandas.pydata.org/docs/reference/index.html#api)
## 3.0 速查表

![image-20240125203538103.png](https://aquazone.oss-cn-guangzhou.aliyuncs.com/image-20240125203538103.png)

![image-20240125203630947.png](https://aquazone.oss-cn-guangzhou.aliyuncs.com/image-20240125203630947.png)

## 3.1 Panads基本介绍

Python Data Analysis Library 或 Pandas是基于Numpy的一种工具，该工具是为了解决数据分析任务而创建的。Pandas纳入了大量库和一些标准的数据模型，提供了高效地操作大型数据集所需的工具。pandas提供了大量能使我们快速便捷地处理数据的函数和方法。

```python
import pandas as pd
import numpy as np
```

### 3.2 Pandas 基本数据结构 

``pandas``有两种常用的基本结构： 

+ ``Series``
    + **一维数组**，与Numpy中的一维**array**类似。二者与Python基本的数据结构**List**也很接近。Series**能保存不同种数据类型**，字符串、boolean值、数字等都能保存在Series中。
+ ``DataFrame``
    + **二维的表格型数据结构**，**存储不同类型数据的二维数组**。很多功能与R中的**data.frame**类似。**可以将DataFrame理解为Series的容器。**以下的内容主要以DataFrame为主。

### 3.2.1 Pandas库的Series类型 
一维``Series``可以用一维列表初始化：
pd_Series.py

```python
s = pd.Series([1,3,5,np.nan,6,8])#index = ['a','b','c','d','x','y'])设置索引（替换原来的0，1，2，3），np.nan设置空值
print(s)

"""
    0    1.0
    1    3.0
    2    5.0
    3    NaN
    4    6.0
    5    8.0
    dtype: float64
默认情况下，``Series``的下标都是数字（可以使用额外参数指定），类型是统一的。
"""


#索引：数据的行标签
s.index #从0到6（不含），1为步长

#RangeIndex(start=0, stop=6, step=1)
#这里也是左闭右开


#值
s.values
#array([ 1.,  3.,  5., nan,  6.,  8.])

s[3]
#nan


#切片操作
s[2:5] #左闭右开
"""
    2    5.0
    3    NaN
    4    6.0
    dtype: float64
"""

s[::2]
"""
	0    1.0
    2    5.0
    4    6.0
    dtype: float64
"""


#索引赋值
s.index.name = '索引'
s
"""
	索引 #在此添加
    0    1.0
    1    3.0
    2    5.0
    3    NaN
    4    6.0
    5    8.0
    dtype: float64
"""

#把索引号换成abcdef
s.index = list('abcdef')
s
"""
    a    1.0
    b    3.0
    c    5.0
    d    NaN
    e    6.0
    f    8.0
    dtype: float64
"""

#依据自己定义的数据类型进行切片，不是左闭右开了
s['a':'c':2] 
"""
    a    1.0
    c    5.0
    dtype: float64
"""
```

### 3.2.2 Pandas库的Dataframe类型

#### 3.2.2.1 创建结构

``DataFrame``则是个二维结构，这里首先构造一组**时间序列**，作为我们第一维的下标：
pd_Dataframe.py

```python
date = pd.date_range("20180101", periods = 6)#持续6次
print(date)
"""
 DatetimeIndex(['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04','2018-01-05', '2018-01-06'], dtype='datetime64[ns]', freq='D')
 """


#然后创建一个``DataFrame``结构：
df = pd.DataFrame(np.random.randn(6,4), index = date, columns = list("ABCD"))#结合numpy中的随机数（六行四列）
df
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2018-01-01</th>
      <td>0.391943</td>
      <td>-1.252843</td>
      <td>-0.247644</td>
      <td>-0.320195</td>
    </tr>
    <tr>
      <th>2018-01-02</th>
      <td>0.845487</td>
      <td>0.208064</td>
      <td>-0.069838</td>
      <td>0.137163</td>
    </tr>
    <tr>
      <th>2018-01-03</th>
      <td>0.776754</td>
      <td>-2.215175</td>
      <td>-1.116371</td>
      <td>1.763836</td>
    </tr>
    <tr>
      <th>2018-01-04</th>
      <td>0.016040</td>
      <td>2.006192</td>
      <td>0.227209</td>
      <td>1.783695</td>
    </tr>
    <tr>
      <th>2018-01-05</th>
      <td>-0.006219</td>
      <td>0.592141</td>
      <td>0.462352</td>
      <td>0.993924</td>
    </tr>
    <tr>
      <th>2018-01-06</th>
      <td>1.112720</td>
      <td>-0.223669</td>
      <td>0.084223</td>
      <td>-0.550868</td>
    </tr>
  </tbody>
</table>


默认情况下，如果不指定``index``参数和``columns``，那么它们的值将从用0开始的数字替代。

```python
df = pd.DataFrame(np.random.randn(6,4))
df
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.284776</td>
      <td>0.568122</td>
      <td>-2.376747</td>
      <td>1.146297</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.534887</td>
      <td>0.142495</td>
      <td>0.628169</td>
      <td>-1.991141</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.569047</td>
      <td>0.255749</td>
      <td>0.139962</td>
      <td>-0.551621</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.334796</td>
      <td>1.025724</td>
      <td>-1.231977</td>
      <td>-0.656463</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.113152</td>
      <td>0.917217</td>
      <td>-0.747063</td>
      <td>-0.686428</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-2.078839</td>
      <td>-0.668506</td>
      <td>0.099214</td>
      <td>-0.038008</td>
    </tr>
  </tbody>
</table>




除了向``DataFrame``中传入二维数组，我们也可以使用字典传入数据：

```python
df2 = pd.DataFrame({'A':1.,'B':pd.Timestamp("20181001"),'C':pd.Series(1,index = list(range(4)),dtype = float),'D':np.array([3]*4, dtype = int),'E':pd.Categorical(["test","train","test","train"]),'F':"abc"}) #B:时间戳,E分类类型
df2
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>E</th>
      <th>F</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>2018-10-01</td>
      <td>1.0</td>
      <td>3</td>
      <td>test</td>
      <td>abc</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>2018-10-01</td>
      <td>1.0</td>
      <td>3</td>
      <td>train</td>
      <td>abc</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>2018-10-01</td>
      <td>1.0</td>
      <td>3</td>
      <td>test</td>
      <td>abc</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>2018-10-01</td>
      <td>1.0</td>
      <td>3</td>
      <td>train</td>
      <td>abc</td>
    </tr>
  </tbody>
</table>


```python
df2.dtypes #查看各个列的数据类型
```

​    A           float64
​    B    datetime64[ns]
​    C           float64
​    D             int32
​    E          category
​    F            object
​    dtype: object

字典的每个``key``代表一列，其``value``可以是各种能够转化为``Series``的对象。

与``Series``要求**所有的类型都一致**不同，``DataFrame``**只要求每一列**数据的格式相同。

#### 3.2.2.2 查看数据
头尾数据：
``head``和``tail``方法可以**分别查看最前面几行和最后面几行的数据（默认为5）**：

```python
df.head()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.284776</td>
      <td>0.568122</td>
      <td>-2.376747</td>
      <td>1.146297</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.534887</td>
      <td>0.142495</td>
      <td>0.628169</td>
      <td>-1.991141</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.569047</td>
      <td>0.255749</td>
      <td>0.139962</td>
      <td>-0.551621</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.334796</td>
      <td>1.025724</td>
      <td>-1.231977</td>
      <td>-0.656463</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.113152</td>
      <td>0.917217</td>
      <td>-0.747063</td>
      <td>-0.686428</td>
    </tr>
  </tbody>
</table>


最后3行：
```python
df.tail(3)
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>0.334796</td>
      <td>1.025724</td>
      <td>-1.231977</td>
      <td>-0.656463</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.113152</td>
      <td>0.917217</td>
      <td>-0.747063</td>
      <td>-0.686428</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-2.078839</td>
      <td>-0.668506</td>
      <td>0.099214</td>
      <td>-0.038008</td>
    </tr>
  </tbody>
</table>


#### 3.2.2.3 下标index，列标columns，数据values
pd_sign.py
```python
#下标使用``index``属性查看：
df.index
"""
DatetimeIndex(['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04','2018-01-05', '2018-01-06'],dtype='datetime64[ns]', freq='D')
"""

#列标使用``columns``属性查看：
df.columns
"""
Index(['A', 'B', 'C', 'D'], dtype='object')
"""

#数据值使用``values``查看：
df.values
"""
array([[ 0.39194344, -1.25284255, -0.24764423, -0.32019526],[ 0.84548738,  0.20806449, -0.06983781,  0.13716277],[ 0.7767544 , -2.21517465, -1.11637102,  1.76383631],[ 0.01603994,  2.00619213,  0.22720908,  1.78369472],[-0.00621932,  0.59214148,  0.46235154,  0.99392424],[ 1.11272049, -0.22366925,  0.08422338, -0.5508679 ]])
"""
```

## 3.3 Pandas读取数据及数据操作 
我们将以豆瓣的电影数据作为我们深入了解Pandas的一个示例。
```python
df = pd.read_excel(r"C:\Users\Lovetianyi\Desktop\python\作业3\豆瓣电影数据.xlsx",index_col = 0) 
#csv:read_csv;绝对路径或相对路径默认在当前文件夹下。r告诉编译器不需要转义
#具体其它参数可以去查帮助文档 ?pd.read_excel
```

```python
df.head()
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>名字</th>
      <th>投票人数</th>
      <th>类型</th>
      <th>产地</th>
      <th>上映时间</th>
      <th>时长</th>
      <th>年代</th>
      <th>评分</th>
      <th>首映地点</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>肖申克的救赎</td>
      <td>692795.0</td>
      <td>剧情/犯罪</td>
      <td>美国</td>
      <td>1994-09-10 00:00:00</td>
      <td>142</td>
      <td>1994</td>
      <td>9.6</td>
      <td>多伦多电影节</td>
    </tr>
    <tr>
      <th>1</th>
      <td>控方证人</td>
      <td>42995.0</td>
      <td>剧情/悬疑/犯罪</td>
      <td>美国</td>
      <td>1957-12-17 00:00:00</td>
      <td>116</td>
      <td>1957</td>
      <td>9.5</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>2</th>
      <td>美丽人生</td>
      <td>327855.0</td>
      <td>剧情/喜剧/爱情</td>
      <td>意大利</td>
      <td>1997-12-20 00:00:00</td>
      <td>116</td>
      <td>1997</td>
      <td>9.5</td>
      <td>意大利</td>
    </tr>
    <tr>
      <th>3</th>
      <td>阿甘正传</td>
      <td>580897.0</td>
      <td>剧情/爱情</td>
      <td>美国</td>
      <td>1994-06-23 00:00:00</td>
      <td>142</td>
      <td>1994</td>
      <td>9.4</td>
      <td>洛杉矶首映</td>
    </tr>
    <tr>
      <th>4</th>
      <td>霸王别姬</td>
      <td>478523.0</td>
      <td>剧情/爱情/同性</td>
      <td>中国大陆</td>
      <td>1993-01-01 00:00:00</td>
      <td>171</td>
      <td>1993</td>
      <td>9.4</td>
      <td>香港</td>
    </tr>
  </tbody>
</table>


### 3.3.1 行操作

```python
df.iloc[0]
```
    名字                   肖申克的救赎
    投票人数                 692795
    类型                    剧情/犯罪
    产地                       美国
    上映时间    1994-09-10 00:00:00
    时长                      142
    年代                     1994
    评分                      9.6
    首映地点                 多伦多电影节
    Name: 0, dtype: object

```python
df.iloc[0:5] #左闭右开
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>名字</th>
      <th>投票人数</th>
      <th>类型</th>
      <th>产地</th>
      <th>上映时间</th>
      <th>时长</th>
      <th>年代</th>
      <th>评分</th>
      <th>首映地点</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>肖申克的救赎</td>
      <td>692795.0</td>
      <td>剧情/犯罪</td>
      <td>美国</td>
      <td>1994-09-10 00:00:00</td>
      <td>142</td>
      <td>1994</td>
      <td>9.6</td>
      <td>多伦多电影节</td>
    </tr>
    <tr>
      <th>1</th>
      <td>控方证人</td>
      <td>42995.0</td>
      <td>剧情/悬疑/犯罪</td>
      <td>美国</td>
      <td>1957-12-17 00:00:00</td>
      <td>116</td>
      <td>1957</td>
      <td>9.5</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>2</th>
      <td>美丽人生</td>
      <td>327855.0</td>
      <td>剧情/喜剧/爱情</td>
      <td>意大利</td>
      <td>1997-12-20 00:00:00</td>
      <td>116</td>
      <td>1997</td>
      <td>9.5</td>
      <td>意大利</td>
    </tr>
    <tr>
      <th>3</th>
      <td>阿甘正传</td>
      <td>580897.0</td>
      <td>剧情/爱情</td>
      <td>美国</td>
      <td>1994-06-23 00:00:00</td>
      <td>142</td>
      <td>1994</td>
      <td>9.4</td>
      <td>洛杉矶首映</td>
    </tr>
    <tr>
      <th>4</th>
      <td>霸王别姬</td>
      <td>478523.0</td>
      <td>剧情/爱情/同性</td>
      <td>中国大陆</td>
      <td>1993-01-01 00:00:00</td>
      <td>171</td>
      <td>1993</td>
      <td>9.4</td>
      <td>香港</td>
    </tr>
  </tbody>
</table>

也可以使用loc

```python
df.loc[0:5] #不是左闭右开
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>名字</th>
      <th>投票人数</th>
      <th>类型</th>
      <th>产地</th>
      <th>上映时间</th>
      <th>时长</th>
      <th>年代</th>
      <th>评分</th>
      <th>首映地点</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>肖申克的救赎</td>
      <td>692795.0</td>
      <td>剧情/犯罪</td>
      <td>美国</td>
      <td>1994-09-10 00:00:00</td>
      <td>142</td>
      <td>1994</td>
      <td>9.6</td>
      <td>多伦多电影节</td>
    </tr>
    <tr>
      <th>1</th>
      <td>控方证人</td>
      <td>42995.0</td>
      <td>剧情/悬疑/犯罪</td>
      <td>美国</td>
      <td>1957-12-17 00:00:00</td>
      <td>116</td>
      <td>1957</td>
      <td>9.5</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>2</th>
      <td>美丽人生</td>
      <td>327855.0</td>
      <td>剧情/喜剧/爱情</td>
      <td>意大利</td>
      <td>1997-12-20 00:00:00</td>
      <td>116</td>
      <td>1997</td>
      <td>9.5</td>
      <td>意大利</td>
    </tr>
    <tr>
      <th>3</th>
      <td>阿甘正传</td>
      <td>580897.0</td>
      <td>剧情/爱情</td>
      <td>美国</td>
      <td>1994-06-23 00:00:00</td>
      <td>142</td>
      <td>1994</td>
      <td>9.4</td>
      <td>洛杉矶首映</td>
    </tr>
    <tr>
      <th>4</th>
      <td>霸王别姬</td>
      <td>478523.0</td>
      <td>剧情/爱情/同性</td>
      <td>中国大陆</td>
      <td>1993-01-01 00:00:00</td>
      <td>171</td>
      <td>1993</td>
      <td>9.4</td>
      <td>香港</td>
    </tr>
    <tr>
      <th>5</th>
      <td>泰坦尼克号</td>
      <td>157074.0</td>
      <td>剧情/爱情/灾难</td>
      <td>美国</td>
      <td>2012-04-10 00:00:00</td>
      <td>194</td>
      <td>2012</td>
      <td>9.4</td>
      <td>中国大陆</td>
    </tr>
  </tbody>
</table>


#### 3.3.1.1 添加一行 append

```python
dit = {"名字":"复仇者联盟3","投票人数":123456,"类型":"剧情/科幻","产地":"美国","上映时间":"2018-05-04 00:00:00","时长":142,"年代":2018,"评分":np.nan,"首映地点":"美国"}
s = pd.Series(dit)#进行字典数据的传入
s.name = 38738
s
```

    名字                   复仇者联盟3
    投票人数                 123456
    类型                    剧情/科幻
    产地                       美国
    上映时间    2018-05-04 00:00:00
    时长                      142
    年代                     2018
    评分                      8.7
    首映地点                     美国
    Name: 38738, dtype: object

```python
df = df.append(s) #覆盖掉原来的数据重新进行赋值
df[-5:]
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>名字</th>
      <th>投票人数</th>
      <th>类型</th>
      <th>产地</th>
      <th>上映时间</th>
      <th>时长</th>
      <th>年代</th>
      <th>评分</th>
      <th>首映地点</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>38734</th>
      <td>1935年</td>
      <td>57.0</td>
      <td>喜剧/歌舞</td>
      <td>美国</td>
      <td>1935-03-15 00:00:00</td>
      <td>98</td>
      <td>1935</td>
      <td>7.6</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>38735</th>
      <td>血溅画屏</td>
      <td>95.0</td>
      <td>剧情/悬疑/犯罪/武侠/古装</td>
      <td>中国大陆</td>
      <td>1905-06-08 00:00:00</td>
      <td>91</td>
      <td>1986</td>
      <td>7.1</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>38736</th>
      <td>魔窟中的幻想</td>
      <td>51.0</td>
      <td>惊悚/恐怖/儿童</td>
      <td>中国大陆</td>
      <td>1905-06-08 00:00:00</td>
      <td>78</td>
      <td>1986</td>
      <td>8.0</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>38737</th>
      <td>列宁格勒围困之星火战役 Блокада: Фильм 2: Ленинградский ме...</td>
      <td>32.0</td>
      <td>剧情/战争</td>
      <td>苏联</td>
      <td>1905-05-30 00:00:00</td>
      <td>97</td>
      <td>1977</td>
      <td>6.6</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>38738</th>
      <td>复仇者联盟3</td>
      <td>123456.0</td>
      <td>剧情/科幻</td>
      <td>美国</td>
      <td>2018-05-04 00:00:00</td>
      <td>142</td>
      <td>2018</td>
      <td>NaN</td>
      <td>美国</td>
    </tr>
  </tbody>
</table>
</div>

#### 3.3.1.2 删除一行 drop

```python
df = df.drop([38738])
df[-5:]
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>名字</th>
      <th>投票人数</th>
      <th>类型</th>
      <th>产地</th>
      <th>上映时间</th>
      <th>时长</th>
      <th>年代</th>
      <th>评分</th>
      <th>首映地点</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>38733</th>
      <td>神学院 S</td>
      <td>46.0</td>
      <td>Adult</td>
      <td>法国</td>
      <td>1905-06-05 00:00:00</td>
      <td>58</td>
      <td>1983</td>
      <td>8.6</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>38734</th>
      <td>1935年</td>
      <td>57.0</td>
      <td>喜剧/歌舞</td>
      <td>美国</td>
      <td>1935-03-15 00:00:00</td>
      <td>98</td>
      <td>1935</td>
      <td>7.6</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>38735</th>
      <td>血溅画屏</td>
      <td>95.0</td>
      <td>剧情/悬疑/犯罪/武侠/古装</td>
      <td>中国大陆</td>
      <td>1905-06-08 00:00:00</td>
      <td>91</td>
      <td>1986</td>
      <td>7.1</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>38736</th>
      <td>魔窟中的幻想</td>
      <td>51.0</td>
      <td>惊悚/恐怖/儿童</td>
      <td>中国大陆</td>
      <td>1905-06-08 00:00:00</td>
      <td>78</td>
      <td>1986</td>
      <td>8.0</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>38737</th>
      <td>列宁格勒围困之星火战役 Блокада: Фильм 2: Ленинградский ме...</td>
      <td>32.0</td>
      <td>剧情/战争</td>
      <td>苏联</td>
      <td>1905-05-30 00:00:00</td>
      <td>97</td>
      <td>1977</td>
      <td>6.6</td>
      <td>美国</td>
    </tr>
  </tbody>
</table>


### 3.3.2 列操作

```python
df.columns
```

    Index(['名字', '投票人数', '类型', '产地', '上映时间', '时长', '年代', '评分', '首映地点'], dtype='object')

```python
df["名字"][:5] #后面中括号表示只想看到的行数，下同
```
    0    肖申克的救赎
    1      控方证人
    2     美丽人生 
    3      阿甘正传
    4      霸王别姬
    Name: 名字, dtype: object

```python
df[["名字","类型"]][:5]#选取列 ？多重[]
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>名字</th>
      <th>类型</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>肖申克的救赎</td>
      <td>剧情/犯罪</td>
    </tr>
    <tr>
      <th>1</th>
      <td>控方证人</td>
      <td>剧情/悬疑/犯罪</td>
    </tr>
    <tr>
      <th>2</th>
      <td>美丽人生</td>
      <td>剧情/喜剧/爱情</td>
    </tr>
    <tr>
      <th>3</th>
      <td>阿甘正传</td>
      <td>剧情/爱情</td>
    </tr>
    <tr>
      <th>4</th>
      <td>霸王别姬</td>
      <td>剧情/爱情/同性</td>
    </tr>
  </tbody>
</table>


#### 3.3.2.1 增加一列 直接赋值

直接赋值

```python
df["序号"] = range(1,len(df)+1) #生成序号的基本方式 len(df)+1表示加一行以后的长度
df[:5]
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>名字</th>
      <th>投票人数</th>
      <th>类型</th>
      <th>产地</th>
      <th>上映时间</th>
      <th>时长</th>
      <th>年代</th>
      <th>评分</th>
      <th>首映地点</th>
      <th>序号</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>肖申克的救赎</td>
      <td>692795.0</td>
      <td>剧情/犯罪</td>
      <td>美国</td>
      <td>1994-09-10 00:00:00</td>
      <td>142</td>
      <td>1994</td>
      <td>9.6</td>
      <td>多伦多电影节</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>控方证人</td>
      <td>42995.0</td>
      <td>剧情/悬疑/犯罪</td>
      <td>美国</td>
      <td>1957-12-17 00:00:00</td>
      <td>116</td>
      <td>1957</td>
      <td>9.5</td>
      <td>美国</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>美丽人生</td>
      <td>327855.0</td>
      <td>剧情/喜剧/爱情</td>
      <td>意大利</td>
      <td>1997-12-20 00:00:00</td>
      <td>116</td>
      <td>1997</td>
      <td>9.5</td>
      <td>意大利</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>阿甘正传</td>
      <td>580897.0</td>
      <td>剧情/爱情</td>
      <td>美国</td>
      <td>1994-06-23 00:00:00</td>
      <td>142</td>
      <td>1994</td>
      <td>9.4</td>
      <td>洛杉矶首映</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>霸王别姬</td>
      <td>478523.0</td>
      <td>剧情/爱情/同性</td>
      <td>中国大陆</td>
      <td>1993-01-01 00:00:00</td>
      <td>171</td>
      <td>1993</td>
      <td>9.4</td>
      <td>香港</td>
      <td>5</td>
    </tr>
  </tbody>
</table>


#### 3.3.2.2 删除一列 drop axis

```python
df = df.drop("序号",axis = 1) #axis指定方向，0为行1为列，默认为0
df[:5]
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>名字</th>
      <th>投票人数</th>
      <th>类型</th>
      <th>产地</th>
      <th>上映时间</th>
      <th>时长</th>
      <th>年代</th>
      <th>评分</th>
      <th>首映地点</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>肖申克的救赎</td>
      <td>692795.0</td>
      <td>剧情/犯罪</td>
      <td>美国</td>
      <td>1994-09-10 00:00:00</td>
      <td>142</td>
      <td>1994</td>
      <td>9.6</td>
      <td>多伦多电影节</td>
    </tr>
    <tr>
      <th>1</th>
      <td>控方证人</td>
      <td>42995.0</td>
      <td>剧情/悬疑/犯罪</td>
      <td>美国</td>
      <td>1957-12-17 00:00:00</td>
      <td>116</td>
      <td>1957</td>
      <td>9.5</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>2</th>
      <td>美丽人生</td>
      <td>327855.0</td>
      <td>剧情/喜剧/爱情</td>
      <td>意大利</td>
      <td>1997-12-20 00:00:00</td>
      <td>116</td>
      <td>1997</td>
      <td>9.5</td>
      <td>意大利</td>
    </tr>
    <tr>
      <th>3</th>
      <td>阿甘正传</td>
      <td>580897.0</td>
      <td>剧情/爱情</td>
      <td>美国</td>
      <td>1994-06-23 00:00:00</td>
      <td>142</td>
      <td>1994</td>
      <td>9.4</td>
      <td>洛杉矶首映</td>
    </tr>
    <tr>
      <th>4</th>
      <td>霸王别姬</td>
      <td>478523.0</td>
      <td>剧情/爱情/同性</td>
      <td>中国大陆</td>
      <td>1993-01-01 00:00:00</td>
      <td>171</td>
      <td>1993</td>
      <td>9.4</td>
      <td>香港</td>
    </tr>
  </tbody>
</table>


#### 3.3.2.3 通过标签选择数据
``df.loc[[index],[colunm]]``通过标签选择数据

```python
df.loc[1,"名字"]
```

    '控方证人'

```python
df.loc[[1,3,5,7,9],["名字","评分"]] #多行跳行多列跳列选择
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>名字</th>
      <th>评分</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>控方证人</td>
      <td>9.5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>阿甘正传</td>
      <td>9.4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>泰坦尼克号</td>
      <td>9.4</td>
    </tr>
    <tr>
      <th>7</th>
      <td>新世纪福音战士剧场版：Air/真心为你 新世紀エヴァンゲリオン劇場版 Ai</td>
      <td>9.4</td>
    </tr>
    <tr>
      <th>9</th>
      <td>这个杀手不太冷</td>
      <td>9.4</td>
    </tr>
  </tbody>
</table>


### 3.3.3 条件选择
#### 3.3.3.1选取产地为美国的所有电影 

```python
df[df["产地"] == "美国"][:5] #内部为bool，需要再加上df[]来输出表格
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>名字</th>
      <th>投票人数</th>
      <th>类型</th>
      <th>产地</th>
      <th>上映时间</th>
      <th>时长</th>
      <th>年代</th>
      <th>评分</th>
      <th>首映地点</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>肖申克的救赎</td>
      <td>692795.0</td>
      <td>剧情/犯罪</td>
      <td>美国</td>
      <td>1994-09-10 00:00:00</td>
      <td>142</td>
      <td>1994</td>
      <td>9.6</td>
      <td>多伦多电影节</td>
    </tr>
    <tr>
      <th>1</th>
      <td>控方证人</td>
      <td>42995.0</td>
      <td>剧情/悬疑/犯罪</td>
      <td>美国</td>
      <td>1957-12-17 00:00:00</td>
      <td>116</td>
      <td>1957</td>
      <td>9.5</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>3</th>
      <td>阿甘正传</td>
      <td>580897.0</td>
      <td>剧情/爱情</td>
      <td>美国</td>
      <td>1994-06-23 00:00:00</td>
      <td>142</td>
      <td>1994</td>
      <td>9.4</td>
      <td>洛杉矶首映</td>
    </tr>
    <tr>
      <th>5</th>
      <td>泰坦尼克号</td>
      <td>157074.0</td>
      <td>剧情/爱情/灾难</td>
      <td>美国</td>
      <td>2012-04-10 00:00:00</td>
      <td>194</td>
      <td>2012</td>
      <td>9.4</td>
      <td>中国大陆</td>
    </tr>
    <tr>
      <th>6</th>
      <td>辛德勒的名单</td>
      <td>306904.0</td>
      <td>剧情/历史/战争</td>
      <td>美国</td>
      <td>1993-11-30 00:00:00</td>
      <td>195</td>
      <td>1993</td>
      <td>9.4</td>
      <td>华盛顿首映</td>
    </tr>
  </tbody>
</table>


#### 3.3.3.2 选取产地为美国的所有电影，并且评分大于9分的电影 &

```python
df[(df.产地 == "美国") & (df.评分 > 9)][:5] #df.标签:更简洁的写法
```



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>名字</th>
      <th>投票人数</th>
      <th>类型</th>
      <th>产地</th>
      <th>上映时间</th>
      <th>时长</th>
      <th>年代</th>
      <th>评分</th>
      <th>首映地点</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>肖申克的救赎</td>
      <td>692795.0</td>
      <td>剧情/犯罪</td>
      <td>美国</td>
      <td>1994-09-10 00:00:00</td>
      <td>142</td>
      <td>1994</td>
      <td>9.6</td>
      <td>多伦多电影节</td>
    </tr>
    <tr>
      <th>1</th>
      <td>控方证人</td>
      <td>42995.0</td>
      <td>剧情/悬疑/犯罪</td>
      <td>美国</td>
      <td>1957-12-17 00:00:00</td>
      <td>116</td>
      <td>1957</td>
      <td>9.5</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>3</th>
      <td>阿甘正传</td>
      <td>580897.0</td>
      <td>剧情/爱情</td>
      <td>美国</td>
      <td>1994-06-23 00:00:00</td>
      <td>142</td>
      <td>1994</td>
      <td>9.4</td>
      <td>洛杉矶首映</td>
    </tr>
    <tr>
      <th>5</th>
      <td>泰坦尼克号</td>
      <td>157074.0</td>
      <td>剧情/爱情/灾难</td>
      <td>美国</td>
      <td>2012-04-10 00:00:00</td>
      <td>194</td>
      <td>2012</td>
      <td>9.4</td>
      <td>中国大陆</td>
    </tr>
    <tr>
      <th>6</th>
      <td>辛德勒的名单</td>
      <td>306904.0</td>
      <td>剧情/历史/战争</td>
      <td>美国</td>
      <td>1993-11-30 00:00:00</td>
      <td>195</td>
      <td>1993</td>
      <td>9.4</td>
      <td>华盛顿首映</td>
    </tr>
  </tbody>
</table>


#### 3.3.3.3 选取产地为美国或中国大陆的所有电影，并且评分大于9分 & |

```python
df[((df.产地 == "美国") | (df.产地 == "中国大陆")) & (df.评分 > 9)][:5]
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>名字</th>
      <th>投票人数</th>
      <th>类型</th>
      <th>产地</th>
      <th>上映时间</th>
      <th>时长</th>
      <th>年代</th>
      <th>评分</th>
      <th>首映地点</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>肖申克的救赎</td>
      <td>692795.0</td>
      <td>剧情/犯罪</td>
      <td>美国</td>
      <td>1994-09-10 00:00:00</td>
      <td>142</td>
      <td>1994</td>
      <td>9.6</td>
      <td>多伦多电影节</td>
    </tr>
    <tr>
      <th>1</th>
      <td>控方证人</td>
      <td>42995.0</td>
      <td>剧情/悬疑/犯罪</td>
      <td>美国</td>
      <td>1957-12-17 00:00:00</td>
      <td>116</td>
      <td>1957</td>
      <td>9.5</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>3</th>
      <td>阿甘正传</td>
      <td>580897.0</td>
      <td>剧情/爱情</td>
      <td>美国</td>
      <td>1994-06-23 00:00:00</td>
      <td>142</td>
      <td>1994</td>
      <td>9.4</td>
      <td>洛杉矶首映</td>
    </tr>
    <tr>
      <th>4</th>
      <td>霸王别姬</td>
      <td>478523.0</td>
      <td>剧情/爱情/同性</td>
      <td>中国大陆</td>
      <td>1993-01-01 00:00:00</td>
      <td>171</td>
      <td>1993</td>
      <td>9.4</td>
      <td>香港</td>
    </tr>
    <tr>
      <th>5</th>
      <td>泰坦尼克号</td>
      <td>157074.0</td>
      <td>剧情/爱情/灾难</td>
      <td>美国</td>
      <td>2012-04-10 00:00:00</td>
      <td>194</td>
      <td>2012</td>
      <td>9.4</td>
      <td>中国大陆</td>
    </tr>
  </tbody>
</table>


## 3.4 缺失值及异常值处理 
### 3.4.1 缺失值处理方法：
|**方法**|**说明**|
|-:|-:|
|**dropna**|根据标签中的缺失值进行过滤，删除缺失值|
|**fillna**|对缺失值进行填充|
|**isnull**|返回一个布尔值对象，判断哪些值是缺失值|
|**notnull**|isnull的否定式|

### 3.4.2 判断缺失值
```python
df[df["名字"].isnull()][:10]#内部是布尔值，加上df[]就显示所有缺失的
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>名字</th>
      <th>投票人数</th>
      <th>类型</th>
      <th>产地</th>
      <th>上映时间</th>
      <th>时长</th>
      <th>年代</th>
      <th>评分</th>
      <th>首映地点</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>231</th>
      <td>NaN</td>
      <td>144.0</td>
      <td>纪录片/音乐</td>
      <td>韩国</td>
      <td>2011-02-02 00:00:00</td>
      <td>90</td>
      <td>2011</td>
      <td>9.7</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>361</th>
      <td>NaN</td>
      <td>80.0</td>
      <td>短片</td>
      <td>其他</td>
      <td>1905-05-17 00:00:00</td>
      <td>4</td>
      <td>1964</td>
      <td>5.7</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>369</th>
      <td>NaN</td>
      <td>5315.0</td>
      <td>剧情</td>
      <td>日本</td>
      <td>2004-07-10 00:00:00</td>
      <td>111</td>
      <td>2004</td>
      <td>7.5</td>
      <td>日本</td>
    </tr>
    <tr>
      <th>372</th>
      <td>NaN</td>
      <td>263.0</td>
      <td>短片/音乐</td>
      <td>英国</td>
      <td>1998-06-30 00:00:00</td>
      <td>34</td>
      <td>1998</td>
      <td>9.2</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>374</th>
      <td>NaN</td>
      <td>47.0</td>
      <td>短片</td>
      <td>其他</td>
      <td>1905-05-17 00:00:00</td>
      <td>3</td>
      <td>1964</td>
      <td>6.7</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>375</th>
      <td>NaN</td>
      <td>1193.0</td>
      <td>短片/音乐</td>
      <td>法国</td>
      <td>1905-07-01 00:00:00</td>
      <td>10</td>
      <td>2010</td>
      <td>7.7</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>411</th>
      <td>NaN</td>
      <td>32.0</td>
      <td>短片</td>
      <td>其他</td>
      <td>1905-05-17 00:00:00</td>
      <td>3</td>
      <td>1964</td>
      <td>7.0</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>432</th>
      <td>NaN</td>
      <td>1081.0</td>
      <td>剧情/动作/惊悚/犯罪</td>
      <td>美国</td>
      <td>2016-02-26 00:00:00</td>
      <td>115</td>
      <td>2016</td>
      <td>6.0</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>441</th>
      <td>NaN</td>
      <td>213.0</td>
      <td>恐怖</td>
      <td>美国</td>
      <td>2007-03-06 00:00:00</td>
      <td>83</td>
      <td>2007</td>
      <td>3.2</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>448</th>
      <td>NaN</td>
      <td>110.0</td>
      <td>纪录片</td>
      <td>荷兰</td>
      <td>2002-04-19 00:00:00</td>
      <td>48</td>
      <td>2000</td>
      <td>9.3</td>
      <td>美国</td>
    </tr>
  </tbody>
</table>


```python
df[df["名字"].notnull()][:5]#非缺失
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>名字</th>
      <th>投票人数</th>
      <th>类型</th>
      <th>产地</th>
      <th>上映时间</th>
      <th>时长</th>
      <th>年代</th>
      <th>评分</th>
      <th>首映地点</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>肖申克的救赎</td>
      <td>692795.0</td>
      <td>剧情/犯罪</td>
      <td>美国</td>
      <td>1994-09-10 00:00:00</td>
      <td>142</td>
      <td>1994</td>
      <td>9.6</td>
      <td>多伦多电影节</td>
    </tr>
    <tr>
      <th>1</th>
      <td>控方证人</td>
      <td>42995.0</td>
      <td>剧情/悬疑/犯罪</td>
      <td>美国</td>
      <td>1957-12-17 00:00:00</td>
      <td>116</td>
      <td>1957</td>
      <td>9.5</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>2</th>
      <td>美丽人生</td>
      <td>327855.0</td>
      <td>剧情/喜剧/爱情</td>
      <td>意大利</td>
      <td>1997-12-20 00:00:00</td>
      <td>116</td>
      <td>1997</td>
      <td>9.5</td>
      <td>意大利</td>
    </tr>
    <tr>
      <th>3</th>
      <td>阿甘正传</td>
      <td>580897.0</td>
      <td>剧情/爱情</td>
      <td>美国</td>
      <td>1994-06-23 00:00:00</td>
      <td>142</td>
      <td>1994</td>
      <td>9.4</td>
      <td>洛杉矶首映</td>
    </tr>
    <tr>
      <th>4</th>
      <td>霸王别姬</td>
      <td>478523.0</td>
      <td>剧情/爱情/同性</td>
      <td>中国大陆</td>
      <td>1993-01-01 00:00:00</td>
      <td>171</td>
      <td>1993</td>
      <td>9.4</td>
      <td>香港</td>
    </tr>
  </tbody>
</table>


### 3.4.3 填充缺失值
```python
df[df["评分"].isnull()][:10] #注意这里特地将前面加入的复仇者联盟令其评分缺失来举例
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>名字</th>
      <th>投票人数</th>
      <th>类型</th>
      <th>产地</th>
      <th>上映时间</th>
      <th>时长</th>
      <th>年代</th>
      <th>评分</th>
      <th>首映地点</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>38738</th>
      <td>复仇者联盟3</td>
      <td>123456.0</td>
      <td>剧情/科幻</td>
      <td>美国</td>
      <td>2018-05-04 00:00:00</td>
      <td>142</td>
      <td>2018</td>
      <td>NaN</td>
      <td>美国</td>
    </tr>
  </tbody>
</table>


```python
df["评分"].fillna(np.mean(df["评分"]), inplace = True) #使用均值来进行替代，inplace意为直接在原始数据中进行修改
df[-5:]
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>名字</th>
      <th>投票人数</th>
      <th>类型</th>
      <th>产地</th>
      <th>上映时间</th>
      <th>时长</th>
      <th>年代</th>
      <th>评分</th>
      <th>首映地点</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>38734</th>
      <td>1935年</td>
      <td>57.0</td>
      <td>喜剧/歌舞</td>
      <td>美国</td>
      <td>1935-03-15 00:00:00</td>
      <td>98</td>
      <td>1935</td>
      <td>7.600000</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>38735</th>
      <td>血溅画屏</td>
      <td>95.0</td>
      <td>剧情/悬疑/犯罪/武侠/古装</td>
      <td>中国大陆</td>
      <td>1905-06-08 00:00:00</td>
      <td>91</td>
      <td>1986</td>
      <td>7.100000</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>38736</th>
      <td>魔窟中的幻想</td>
      <td>51.0</td>
      <td>惊悚/恐怖/儿童</td>
      <td>中国大陆</td>
      <td>1905-06-08 00:00:00</td>
      <td>78</td>
      <td>1986</td>
      <td>8.000000</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>38737</th>
      <td>列宁格勒围困之星火战役 Блокада: Фильм 2: Ленинградский ме...</td>
      <td>32.0</td>
      <td>剧情/战争</td>
      <td>苏联</td>
      <td>1905-05-30 00:00:00</td>
      <td>97</td>
      <td>1977</td>
      <td>6.600000</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>38738</th>
      <td>复仇者联盟3</td>
      <td>123456.0</td>
      <td>剧情/科幻</td>
      <td>美国</td>
      <td>2018-05-04 00:00:00</td>
      <td>142</td>
      <td>2018</td>
      <td>6.935704</td>
      <td>美国</td>
    </tr>
  </tbody>
</table>


```python
df1 = df.fillna("未知电影") #谨慎使用，除非确定所有的空值都是在一列中，否则所有的空值都会填成这个
#不可采用df["名字"].fillna("未知电影")的形式，因为填写后数据格式就变了，变成Series了
```

```python
df1[df1["名字"].isnull()][:10]
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>名字</th>
      <th>投票人数</th>
      <th>类型</th>
      <th>产地</th>
      <th>上映时间</th>
      <th>时长</th>
      <th>年代</th>
      <th>评分</th>
      <th>首映地点</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>


### 3.4.4 删除缺失值
```python
df.dropna() 参数

how = 'all':删除全为空值的行或列
inplace = True: 覆盖之前的数据
axis = 0: 选择行或列，默认是行

len(df)
#38739

df2 = df.dropna()
len(df2)
#38176

df.dropna(inplace = True)
len(df) #inplace覆盖掉原来的值
#38176
```

### 3.4.5 处理异常值

异常值，即在数据集中存在不合理的值，又称离群点。比如年龄为-1，笔记本电脑重量为1吨等，都属于异常值的范围。

```python
df[df["投票人数"] < 0] #直接删除，或者找原始数据来修正都行
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>名字</th>
      <th>投票人数</th>
      <th>类型</th>
      <th>产地</th>
      <th>上映时间</th>
      <th>时长</th>
      <th>年代</th>
      <th>评分</th>
      <th>首映地点</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>19777</th>
      <td>皇家大贼 皇家大</td>
      <td>-80.0</td>
      <td>剧情/犯罪</td>
      <td>中国香港</td>
      <td>1985-05-31 00:00:00</td>
      <td>60</td>
      <td>1985</td>
      <td>6.3</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>19786</th>
      <td>日本的垃圾去中国大陆 にっぽんの“ゴミ” 大陆へ渡る ～中国式リサイクル錬</td>
      <td>-80.0</td>
      <td>纪录片</td>
      <td>日本</td>
      <td>1905-06-26 00:00:00</td>
      <td>60</td>
      <td>2004</td>
      <td>7.9</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>19797</th>
      <td>女教徒</td>
      <td>-118.0</td>
      <td>剧情</td>
      <td>法国</td>
      <td>1966-05-06 00:00:00</td>
      <td>135</td>
      <td>1966</td>
      <td>7.8</td>
      <td>美国</td>
    </tr>
  </tbody>
</table>


```python
df[df["投票人数"] % 1 != 0] #小数异常值
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>名字</th>
      <th>投票人数</th>
      <th>类型</th>
      <th>产地</th>
      <th>上映时间</th>
      <th>时长</th>
      <th>年代</th>
      <th>评分</th>
      <th>首映地点</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>19791</th>
      <td>女教师 女教</td>
      <td>8.30</td>
      <td>剧情/犯罪</td>
      <td>日本</td>
      <td>1977-10-29 00:00:00</td>
      <td>100</td>
      <td>1977</td>
      <td>6.6</td>
      <td>日本</td>
    </tr>
    <tr>
      <th>19804</th>
      <td>女郎漫游仙境 ドレミファ娘の血は騒</td>
      <td>5.90</td>
      <td>喜剧/歌舞</td>
      <td>日本</td>
      <td>1985-11-03 00:00:00</td>
      <td>80</td>
      <td>1985</td>
      <td>6.7</td>
      <td>日本</td>
    </tr>
    <tr>
      <th>19820</th>
      <td>女仆日记</td>
      <td>12.87</td>
      <td>剧情</td>
      <td>法国</td>
      <td>2015-04-01 00:00:00</td>
      <td>96</td>
      <td>2015</td>
      <td>5.7</td>
      <td>法国</td>
    </tr>
    <tr>
      <th>38055</th>
      <td>逃出亚卡拉</td>
      <td>12.87</td>
      <td>剧情/动作/惊悚/犯罪</td>
      <td>美国</td>
      <td>1979-09-20 00:00:00</td>
      <td>112</td>
      <td>1979</td>
      <td>7.8</td>
      <td>美国</td>
    </tr>
  </tbody>
</table>


**对于异常值，一般来说数量都会很少，在不影响整体数据分布的情况下，我们直接删除就可以了**

**其他属性的异常值处理，我们会在格式转换部分，进一步讨论**

```python
df = df[df.投票人数 > 0]
df = df[df["投票人数"] % 1 == 0]
```


## 3.5 数据保存与读入

数据处理之后，然后将数据重新保存到movie_data.xlsx 

```python
df.to_excel("movie_data.xlsx") #默认路径为现在文件夹所在的路径
```
读入已保存文件:

```python
df = pd.read_excel(r"C:\Users\Lovetianyi\Desktop\python\作业3\movie_data.xlsx",index_col = 0)

df[:5]
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>名字</th>
      <th>投票人数</th>
      <th>类型</th>
      <th>产地</th>
      <th>上映时间</th>
      <th>时长</th>
      <th>年代</th>
      <th>评分</th>
      <th>首映地点</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>肖申克的救赎</td>
      <td>692795</td>
      <td>剧情/犯罪</td>
      <td>美国</td>
      <td>1994-09-10 00:00:00</td>
      <td>142</td>
      <td>1994</td>
      <td>9.6</td>
      <td>多伦多电影节</td>
    </tr>
    <tr>
      <th>1</th>
      <td>控方证人</td>
      <td>42995</td>
      <td>剧情/悬疑/犯罪</td>
      <td>美国</td>
      <td>1957-12-17 00:00:00</td>
      <td>116</td>
      <td>1957</td>
      <td>9.5</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>2</th>
      <td>美丽人生</td>
      <td>327855</td>
      <td>剧情/喜剧/爱情</td>
      <td>意大利</td>
      <td>1997-12-20 00:00:00</td>
      <td>116</td>
      <td>1997</td>
      <td>9.5</td>
      <td>意大利</td>
    </tr>
    <tr>
      <th>3</th>
      <td>阿甘正传</td>
      <td>580897</td>
      <td>剧情/爱情</td>
      <td>美国</td>
      <td>1994-06-23 00:00:00</td>
      <td>142</td>
      <td>1994</td>
      <td>9.4</td>
      <td>洛杉矶首映</td>
    </tr>
    <tr>
      <th>4</th>
      <td>霸王别姬</td>
      <td>478523</td>
      <td>剧情/爱情/同性</td>
      <td>中国大陆</td>
      <td>1993-01-01 00:00:00</td>
      <td>171</td>
      <td>1993</td>
      <td>9.4</td>
      <td>香港</td>
    </tr>
  </tbody>
</table>


## 3.6 数据格式转换
在做数据分析的时候，原始数据往往会因为各种各样的原因产生各种数据格式的问题。  
数据格式是我们非常需要注意的一点，数据格式错误往往会造成很严重的后果。  
并且，很多异常值也是我们经过格式转换后才会发现，对我们规整数据，清洗数据有着重要的作用。

### 3.6.1 查看格式 

```python
df["投票人数"].dtype

#dtype('int64')

df["投票人数"] = df["投票人数"].astype("int") #转换格式
df[:5]
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>名字</th>
      <th>投票人数</th>
      <th>类型</th>
      <th>产地</th>
      <th>上映时间</th>
      <th>时长</th>
      <th>年代</th>
      <th>评分</th>
      <th>首映地点</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>肖申克的救赎</td>
      <td>692795</td>
      <td>剧情/犯罪</td>
      <td>美国</td>
      <td>1994-09-10 00:00:00</td>
      <td>142</td>
      <td>1994</td>
      <td>9.6</td>
      <td>多伦多电影节</td>
    </tr>
    <tr>
      <th>1</th>
      <td>控方证人</td>
      <td>42995</td>
      <td>剧情/悬疑/犯罪</td>
      <td>美国</td>
      <td>1957-12-17 00:00:00</td>
      <td>116</td>
      <td>1957</td>
      <td>9.5</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>2</th>
      <td>美丽人生</td>
      <td>327855</td>
      <td>剧情/喜剧/爱情</td>
      <td>意大利</td>
      <td>1997-12-20 00:00:00</td>
      <td>116</td>
      <td>1997</td>
      <td>9.5</td>
      <td>意大利</td>
    </tr>
    <tr>
      <th>3</th>
      <td>阿甘正传</td>
      <td>580897</td>
      <td>剧情/爱情</td>
      <td>美国</td>
      <td>1994-06-23 00:00:00</td>
      <td>142</td>
      <td>1994</td>
      <td>9.4</td>
      <td>洛杉矶首映</td>
    </tr>
    <tr>
      <th>4</th>
      <td>霸王别姬</td>
      <td>478523</td>
      <td>剧情/爱情/同性</td>
      <td>中国大陆</td>
      <td>1993-01-01 00:00:00</td>
      <td>171</td>
      <td>1993</td>
      <td>9.4</td>
      <td>香港</td>
    </tr>
  </tbody>
</table>


```python
df["产地"].dtype
#dtype('O')

df["产地"] = df["产地"].astype("str")
```

### 3.6.2 将年份转化为整数格式 

```python
df["年代"] = df["年代"].astype("int") #有异常值会报错
```

    ---------------------------------------------------------------------------
    
    ValueError                                Traceback (most recent call last)
    
    <ipython-input-9-aafea50a8773> in <module>
    ----> 1 df["年代"] = df["年代"].astype("int") #有异常值会报错


    ~\anaconda3\lib\site-packages\pandas\core\generic.py in astype(self, dtype, copy, errors)
       5696         else:
       5697             # else, only a single dtype is given
    -> 5698             new_data = self._data.astype(dtype=dtype, copy=copy, errors=errors)
       5699             return self._constructor(new_data).__finalize__(self)
       5700 


    ~\anaconda3\lib\site-packages\pandas\core\internals\managers.py in astype(self, dtype, copy, errors)
        580 
        581     def astype(self, dtype, copy: bool = False, errors: str = "raise"):
    --> 582         return self.apply("astype", dtype=dtype, copy=copy, errors=errors)
        583 
        584     def convert(self, **kwargs):


    ~\anaconda3\lib\site-packages\pandas\core\internals\managers.py in apply(self, f, filter, **kwargs)
        440                 applied = b.apply(f, **kwargs)
        441             else:
    --> 442                 applied = getattr(b, f)(**kwargs)
        443             result_blocks = _extend_blocks(applied, result_blocks)
        444 


    ~\anaconda3\lib\site-packages\pandas\core\internals\blocks.py in astype(self, dtype, copy, errors)
        623             vals1d = values.ravel()
        624             try:
    --> 625                 values = astype_nansafe(vals1d, dtype, copy=True)
        626             except (ValueError, TypeError):
        627                 # e.g. astype_nansafe can fail on object-dtype of strings


    ~\anaconda3\lib\site-packages\pandas\core\dtypes\cast.py in astype_nansafe(arr, dtype, copy, skipna)
        872         # work around NumPy brokenness, #1987
        873         if np.issubdtype(dtype.type, np.integer):
    --> 874             return lib.astype_intsafe(arr.ravel(), dtype).reshape(arr.shape)
        875 
        876         # if we have a datetime/timedelta array of objects


    pandas\_libs\lib.pyx in pandas._libs.lib.astype_intsafe()


    ValueError: invalid literal for int() with base 10: '2008\u200e'

```python
df[df.年代 == "2008\u200e"] #找到异常数据
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>名字</th>
      <th>投票人数</th>
      <th>类型</th>
      <th>产地</th>
      <th>上映时间</th>
      <th>时长</th>
      <th>年代</th>
      <th>评分</th>
      <th>首映地点</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>15205</th>
      <td>狂蟒惊魂</td>
      <td>544</td>
      <td>恐怖</td>
      <td>中国大陆</td>
      <td>2008-04-08 00:00:00</td>
      <td>93</td>
      <td>2008‎</td>
      <td>2.7</td>
      <td>美国</td>
    </tr>
  </tbody>
</table>


```python
df[df.年代 == "2008\u200e"]["年代"].values #后面是unicode的控制字符，使得其显示靠左，因此需要处理删除
```

    array(['2008\u200e'], dtype=object)

```python
df.loc[[14934,15205],"年代"] = 2008
```

```python
df.loc[14934]
```
    名字                    奶奶强盗团
    投票人数                  12591
    类型                 剧情/喜剧/动作
    产地                       韩国
    上映时间    2010-03-18 00:00:00
    时长                      107
    年代                     2008
    评分                      7.7
    首映地点                     韩国
    Name: 14934, dtype: object

```python
df["年代"] = df["年代"].astype("int")#此时无报错
```

```python
df["年代"][:5]
```
    0    1994
    1    1957
    2    1997
    3    1994
    4    1993
    Name: 年代, dtype: int32

### 3.6.3 将时长转化为整数格式 

```python
df["时长"] = df["时长"].astype("int")
```

    ---------------------------------------------------------------------------
    
    ValueError                                Traceback (most recent call last)
    
    <ipython-input-16-97b0e6bbe2ae> in <module>
    ----> 1 df["时长"] = df["时长"].astype("int")


    ~\anaconda3\lib\site-packages\pandas\core\generic.py in astype(self, dtype, copy, errors)
       5696         else:
       5697             # else, only a single dtype is given
    -> 5698             new_data = self._data.astype(dtype=dtype, copy=copy, errors=errors)
       5699             return self._constructor(new_data).__finalize__(self)
       5700 


    ~\anaconda3\lib\site-packages\pandas\core\internals\managers.py in astype(self, dtype, copy, errors)
        580 
        581     def astype(self, dtype, copy: bool = False, errors: str = "raise"):
    --> 582         return self.apply("astype", dtype=dtype, copy=copy, errors=errors)
        583 
        584     def convert(self, **kwargs):


    ~\anaconda3\lib\site-packages\pandas\core\internals\managers.py in apply(self, f, filter, **kwargs)
        440                 applied = b.apply(f, **kwargs)
        441             else:
    --> 442                 applied = getattr(b, f)(**kwargs)
        443             result_blocks = _extend_blocks(applied, result_blocks)
        444 


    ~\anaconda3\lib\site-packages\pandas\core\internals\blocks.py in astype(self, dtype, copy, errors)
        623             vals1d = values.ravel()
        624             try:
    --> 625                 values = astype_nansafe(vals1d, dtype, copy=True)
        626             except (ValueError, TypeError):
        627                 # e.g. astype_nansafe can fail on object-dtype of strings


    ~\anaconda3\lib\site-packages\pandas\core\dtypes\cast.py in astype_nansafe(arr, dtype, copy, skipna)
        872         # work around NumPy brokenness, #1987
        873         if np.issubdtype(dtype.type, np.integer):
    --> 874             return lib.astype_intsafe(arr.ravel(), dtype).reshape(arr.shape)
        875 
        876         # if we have a datetime/timedelta array of objects


    pandas\_libs\lib.pyx in pandas._libs.lib.astype_intsafe()


    ValueError: invalid literal for int() with base 10: '8U'

```python
df[df["时长"] == "8U"] #寻找异常值，不知道怎么改的话可以删除
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>名字</th>
      <th>投票人数</th>
      <th>类型</th>
      <th>产地</th>
      <th>上映时间</th>
      <th>时长</th>
      <th>年代</th>
      <th>评分</th>
      <th>首映地点</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>31644</th>
      <td>一个被隔绝的世界</td>
      <td>46</td>
      <td>纪录片/短片</td>
      <td>瑞典</td>
      <td>2001-10-25 00:00:00</td>
      <td>8U</td>
      <td>1948</td>
      <td>7.8</td>
      <td>美国</td>
    </tr>
  </tbody>
</table>


```python
df.drop([31644], inplace = True)
```

```python
df["时长"] = df["时长"].astype("int")
```

    ---------------------------------------------------------------------------
    
    ValueError                                Traceback (most recent call last)
    
    <ipython-input-19-97b0e6bbe2ae> in <module>
    ----> 1 df["时长"] = df["时长"].astype("int")


    ~\anaconda3\lib\site-packages\pandas\core\generic.py in astype(self, dtype, copy, errors)
       5696         else:
       5697             # else, only a single dtype is given
    -> 5698             new_data = self._data.astype(dtype=dtype, copy=copy, errors=errors)
       5699             return self._constructor(new_data).__finalize__(self)
       5700 


    ~\anaconda3\lib\site-packages\pandas\core\internals\managers.py in astype(self, dtype, copy, errors)
        580 
        581     def astype(self, dtype, copy: bool = False, errors: str = "raise"):
    --> 582         return self.apply("astype", dtype=dtype, copy=copy, errors=errors)
        583 
        584     def convert(self, **kwargs):


    ~\anaconda3\lib\site-packages\pandas\core\internals\managers.py in apply(self, f, filter, **kwargs)
        440                 applied = b.apply(f, **kwargs)
        441             else:
    --> 442                 applied = getattr(b, f)(**kwargs)
        443             result_blocks = _extend_blocks(applied, result_blocks)
        444 


    ~\anaconda3\lib\site-packages\pandas\core\internals\blocks.py in astype(self, dtype, copy, errors)
        623             vals1d = values.ravel()
        624             try:
    --> 625                 values = astype_nansafe(vals1d, dtype, copy=True)
        626             except (ValueError, TypeError):
        627                 # e.g. astype_nansafe can fail on object-dtype of strings


    ~\anaconda3\lib\site-packages\pandas\core\dtypes\cast.py in astype_nansafe(arr, dtype, copy, skipna)
        872         # work around NumPy brokenness, #1987
        873         if np.issubdtype(dtype.type, np.integer):
    --> 874             return lib.astype_intsafe(arr.ravel(), dtype).reshape(arr.shape)
        875 
        876         # if we have a datetime/timedelta array of objects


    pandas\_libs\lib.pyx in pandas._libs.lib.astype_intsafe()


    ValueError: invalid literal for int() with base 10: '12J'

```python
df[df["时长"] == "12J"]
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>名字</th>
      <th>投票人数</th>
      <th>类型</th>
      <th>产地</th>
      <th>上映时间</th>
      <th>时长</th>
      <th>年代</th>
      <th>评分</th>
      <th>首映地点</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>32949</th>
      <td>渔业危机</td>
      <td>41</td>
      <td>纪录片</td>
      <td>英国</td>
      <td>2009-06-19 00:00:00</td>
      <td>12J</td>
      <td>2008</td>
      <td>8.2</td>
      <td>USA</td>
    </tr>
  </tbody>
</table>


```python
df.drop([32949], inplace = True) #删数据，inplace替换原来数据
```

```python
df["时长"] = df["时长"].astype("int")
```

```python
df[:5]
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>名字</th>
      <th>投票人数</th>
      <th>类型</th>
      <th>产地</th>
      <th>上映时间</th>
      <th>时长</th>
      <th>年代</th>
      <th>评分</th>
      <th>首映地点</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>肖申克的救赎</td>
      <td>692795</td>
      <td>剧情/犯罪</td>
      <td>美国</td>
      <td>1994-09-10 00:00:00</td>
      <td>142</td>
      <td>1994</td>
      <td>9.6</td>
      <td>多伦多电影节</td>
    </tr>
    <tr>
      <th>1</th>
      <td>控方证人</td>
      <td>42995</td>
      <td>剧情/悬疑/犯罪</td>
      <td>美国</td>
      <td>1957-12-17 00:00:00</td>
      <td>116</td>
      <td>1957</td>
      <td>9.5</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>2</th>
      <td>美丽人生</td>
      <td>327855</td>
      <td>剧情/喜剧/爱情</td>
      <td>意大利</td>
      <td>1997-12-20 00:00:00</td>
      <td>116</td>
      <td>1997</td>
      <td>9.5</td>
      <td>意大利</td>
    </tr>
    <tr>
      <th>3</th>
      <td>阿甘正传</td>
      <td>580897</td>
      <td>剧情/爱情</td>
      <td>美国</td>
      <td>1994-06-23 00:00:00</td>
      <td>142</td>
      <td>1994</td>
      <td>9.4</td>
      <td>洛杉矶首映</td>
    </tr>
    <tr>
      <th>4</th>
      <td>霸王别姬</td>
      <td>478523</td>
      <td>剧情/爱情/同性</td>
      <td>中国大陆</td>
      <td>1993-01-01 00:00:00</td>
      <td>171</td>
      <td>1993</td>
      <td>9.4</td>
      <td>香港</td>
    </tr>
  </tbody>
</table>




## 3.7 排序

### 3.7.1 默认排序

```python
df[:10]#默认根据index进行排序
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>名字</th>
      <th>投票人数</th>
      <th>类型</th>
      <th>产地</th>
      <th>上映时间</th>
      <th>时长</th>
      <th>年代</th>
      <th>评分</th>
      <th>首映地点</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>肖申克的救赎</td>
      <td>692795</td>
      <td>剧情/犯罪</td>
      <td>美国</td>
      <td>1994-09-10 00:00:00</td>
      <td>142</td>
      <td>1994</td>
      <td>9.6</td>
      <td>多伦多电影节</td>
    </tr>
    <tr>
      <th>1</th>
      <td>控方证人</td>
      <td>42995</td>
      <td>剧情/悬疑/犯罪</td>
      <td>美国</td>
      <td>1957-12-17 00:00:00</td>
      <td>116</td>
      <td>1957</td>
      <td>9.5</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>2</th>
      <td>美丽人生</td>
      <td>327855</td>
      <td>剧情/喜剧/爱情</td>
      <td>意大利</td>
      <td>1997-12-20 00:00:00</td>
      <td>116</td>
      <td>1997</td>
      <td>9.5</td>
      <td>意大利</td>
    </tr>
    <tr>
      <th>3</th>
      <td>阿甘正传</td>
      <td>580897</td>
      <td>剧情/爱情</td>
      <td>美国</td>
      <td>1994-06-23 00:00:00</td>
      <td>142</td>
      <td>1994</td>
      <td>9.4</td>
      <td>洛杉矶首映</td>
    </tr>
    <tr>
      <th>4</th>
      <td>霸王别姬</td>
      <td>478523</td>
      <td>剧情/爱情/同性</td>
      <td>中国大陆</td>
      <td>1993-01-01 00:00:00</td>
      <td>171</td>
      <td>1993</td>
      <td>9.4</td>
      <td>香港</td>
    </tr>
    <tr>
      <th>5</th>
      <td>泰坦尼克号</td>
      <td>157074</td>
      <td>剧情/爱情/灾难</td>
      <td>美国</td>
      <td>2012-04-10 00:00:00</td>
      <td>194</td>
      <td>2012</td>
      <td>9.4</td>
      <td>中国大陆</td>
    </tr>
    <tr>
      <th>6</th>
      <td>辛德勒的名单</td>
      <td>306904</td>
      <td>剧情/历史/战争</td>
      <td>美国</td>
      <td>1993-11-30 00:00:00</td>
      <td>195</td>
      <td>1993</td>
      <td>9.4</td>
      <td>华盛顿首映</td>
    </tr>
    <tr>
      <th>7</th>
      <td>新世纪福音战士剧场版：Air/真心为你 新世紀エヴァンゲリオン劇場版 Ai</td>
      <td>24355</td>
      <td>剧情/动作/科幻/动画/奇幻</td>
      <td>日本</td>
      <td>1997-07-19 00:00:00</td>
      <td>87</td>
      <td>1997</td>
      <td>9.4</td>
      <td>日本</td>
    </tr>
    <tr>
      <th>8</th>
      <td>银魂完结篇：直到永远的万事屋 劇場版 銀魂 完結篇 万事屋よ</td>
      <td>21513</td>
      <td>剧情/动画</td>
      <td>日本</td>
      <td>2013-07-06 00:00:00</td>
      <td>110</td>
      <td>2013</td>
      <td>9.4</td>
      <td>日本</td>
    </tr>
    <tr>
      <th>9</th>
      <td>这个杀手不太冷</td>
      <td>662552</td>
      <td>剧情/动作/犯罪</td>
      <td>法国</td>
      <td>1994-09-14 00:00:00</td>
      <td>133</td>
      <td>1994</td>
      <td>9.4</td>
      <td>法国</td>
    </tr>
  </tbody>
</table>


### 3.7.2 按照投票人数进行排序  sort_values(by= , ascending = <Bool>)[:]


```python
df.sort_values(by = "投票人数", ascending = False)[:5] #默认从小到大
```

**by:指定某项** 

**ascending:**

​	**True:默认升序**

​	**False:降序**



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>名字</th>
      <th>投票人数</th>
      <th>类型</th>
      <th>产地</th>
      <th>上映时间</th>
      <th>时长</th>
      <th>年代</th>
      <th>评分</th>
      <th>首映地点</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>肖申克的救赎</td>
      <td>692795</td>
      <td>剧情/犯罪</td>
      <td>美国</td>
      <td>1994-09-10 00:00:00</td>
      <td>142</td>
      <td>1994</td>
      <td>9.6</td>
      <td>多伦多电影节</td>
    </tr>
    <tr>
      <th>9</th>
      <td>这个杀手不太冷</td>
      <td>662552</td>
      <td>剧情/动作/犯罪</td>
      <td>法国</td>
      <td>1994-09-14 00:00:00</td>
      <td>133</td>
      <td>1994</td>
      <td>9.4</td>
      <td>法国</td>
    </tr>
    <tr>
      <th>22</th>
      <td>盗梦空间</td>
      <td>642134</td>
      <td>剧情/动作/科幻/悬疑/冒险</td>
      <td>美国</td>
      <td>2010-09-01 00:00:00</td>
      <td>148</td>
      <td>2010</td>
      <td>9.2</td>
      <td>中国大陆</td>
    </tr>
    <tr>
      <th>3</th>
      <td>阿甘正传</td>
      <td>580897</td>
      <td>剧情/爱情</td>
      <td>美国</td>
      <td>1994-06-23 00:00:00</td>
      <td>142</td>
      <td>1994</td>
      <td>9.4</td>
      <td>洛杉矶首映</td>
    </tr>
    <tr>
      <th>99</th>
      <td>三傻大闹宝莱坞</td>
      <td>549808</td>
      <td>剧情/喜剧/爱情/歌舞</td>
      <td>印度</td>
      <td>2011-12-08 00:00:00</td>
      <td>171</td>
      <td>2009</td>
      <td>9.1</td>
      <td>中国大陆</td>
    </tr>
  </tbody>
</table>


### 3.7.3 按照年代进行排序


```python
df.sort_values(by = "年代")[:5]
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>名字</th>
      <th>投票人数</th>
      <th>类型</th>
      <th>产地</th>
      <th>上映时间</th>
      <th>时长</th>
      <th>年代</th>
      <th>评分</th>
      <th>首映地点</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1700</th>
      <td>朗德海花园场景</td>
      <td>650</td>
      <td>短片</td>
      <td>英国</td>
      <td>1888-10-14</td>
      <td>60</td>
      <td>1888</td>
      <td>8.7</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>14048</th>
      <td>利兹大桥</td>
      <td>126</td>
      <td>短片</td>
      <td>英国</td>
      <td>1888-10</td>
      <td>60</td>
      <td>1888</td>
      <td>7.2</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>26170</th>
      <td>恶作剧</td>
      <td>51</td>
      <td>短片</td>
      <td>美国</td>
      <td>1905-03-04 00:00:00</td>
      <td>60</td>
      <td>1890</td>
      <td>4.8</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>10627</th>
      <td>可怜的比埃洛</td>
      <td>176</td>
      <td>喜剧/爱情/动画/短片</td>
      <td>法国</td>
      <td>1892-10-28</td>
      <td>60</td>
      <td>1892</td>
      <td>7.5</td>
      <td>法国</td>
    </tr>
    <tr>
      <th>21765</th>
      <td>胚胎植入前遗传学筛查</td>
      <td>69</td>
      <td>纪录片/短片</td>
      <td>美国</td>
      <td>1894-05-18</td>
      <td>60</td>
      <td>1894</td>
      <td>5.7</td>
      <td>美国</td>
    </tr>
  </tbody>
</table>




### 3.7.4 多个值排序，先按照评分，再按照投票人数  sort_values(by = [" "," "],ascending = <Bool>)


```python
df.sort_values(by = ["评分","投票人数"], ascending = False) #列表中的顺序决定先后顺序
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>名字</th>
      <th>投票人数</th>
      <th>类型</th>
      <th>产地</th>
      <th>上映时间</th>
      <th>时长</th>
      <th>年代</th>
      <th>评分</th>
      <th>首映地点</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9278</th>
      <td>平安结祈 平安結</td>
      <td>208</td>
      <td>音乐</td>
      <td>日本</td>
      <td>2012-02-24 00:00:00</td>
      <td>60</td>
      <td>2012</td>
      <td>9.9</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>13882</th>
      <td>武之舞</td>
      <td>128</td>
      <td>纪录片</td>
      <td>中国大陆</td>
      <td>1997-02-01 00:00:00</td>
      <td>60</td>
      <td>34943</td>
      <td>9.9</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>23559</th>
      <td>未作回答的问题：伯恩斯坦哈佛六讲</td>
      <td>61</td>
      <td>纪录片</td>
      <td>美国</td>
      <td>1905-05-29 00:00:00</td>
      <td>60</td>
      <td>1973</td>
      <td>9.9</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>25273</th>
      <td>索科洛夫：巴黎现场</td>
      <td>43</td>
      <td>音乐</td>
      <td>法国</td>
      <td>2002-11-04 00:00:00</td>
      <td>127</td>
      <td>2002</td>
      <td>9.9</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>11479</th>
      <td>公园现场</td>
      <td>163</td>
      <td>音乐</td>
      <td>英国</td>
      <td>2012-12-03 00:00:00</td>
      <td>60</td>
      <td>2012</td>
      <td>9.8</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2790</th>
      <td>爸爸我来救你了</td>
      <td>278</td>
      <td>喜剧/动作/家庭/儿童/冒险</td>
      <td>中国大陆</td>
      <td>2016-01-22 00:00:00</td>
      <td>90</td>
      <td>2015</td>
      <td>2.2</td>
      <td>中国大陆</td>
    </tr>
    <tr>
      <th>10219</th>
      <td>大震撼</td>
      <td>185</td>
      <td>剧情</td>
      <td>中国大陆</td>
      <td>2011-05-19 00:00:00</td>
      <td>60</td>
      <td>2011</td>
      <td>2.2</td>
      <td>中国大陆</td>
    </tr>
    <tr>
      <th>9045</th>
      <td>钢管侠</td>
      <td>168</td>
      <td>动作</td>
      <td>中国大陆</td>
      <td>2015-07-28 00:00:00</td>
      <td>60</td>
      <td>2015</td>
      <td>2.2</td>
      <td>中国大陆</td>
    </tr>
    <tr>
      <th>13137</th>
      <td>嫁给大山的女人</td>
      <td>2690</td>
      <td>剧情</td>
      <td>中国大陆</td>
      <td>2009-04-22 00:00:00</td>
      <td>88</td>
      <td>2009</td>
      <td>2.1</td>
      <td>中国大陆</td>
    </tr>
    <tr>
      <th>29100</th>
      <td>都是手机惹的祸</td>
      <td>42</td>
      <td>喜剧</td>
      <td>中国大陆</td>
      <td>2013-01-18 00:00:00</td>
      <td>60</td>
      <td>2012</td>
      <td>2.0</td>
      <td>中国大陆</td>
    </tr>
  </tbody>
</table>
<p>38167 rows × 9 columns</p>




## 3.8 基本统计分析

### 3.8.1 描述性统计

**dataframe.describe()：对dataframe中的数值型数据进行描述性统计(概览)**


```python
df.describe()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>投票人数</th>
      <th>时长</th>
      <th>年代</th>
      <th>评分</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>38167.000000</td>
      <td>38167.000000</td>
      <td>38167.000000</td>
      <td>38167.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>6268.131291</td>
      <td>89.475594</td>
      <td>1998.805224</td>
      <td>6.922154</td>
    </tr>
    <tr>
      <th>std</th>
      <td>26298.331602</td>
      <td>83.763856</td>
      <td>255.065394</td>
      <td>1.263782</td>
    </tr>
    <tr>
      <th>min</th>
      <td>21.000000</td>
      <td>1.000000</td>
      <td>1888.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>101.000000</td>
      <td>60.000000</td>
      <td>1990.000000</td>
      <td>6.300000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>354.000000</td>
      <td>93.000000</td>
      <td>2005.000000</td>
      <td>7.100000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1798.500000</td>
      <td>106.000000</td>
      <td>2010.000000</td>
      <td>7.800000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>692795.000000</td>
      <td>11500.000000</td>
      <td>39180.000000</td>
      <td>9.900000</td>
    </tr>
  </tbody>
</table>




**通过描述性统计，可以发现一些异常值，很多异常值往往是需要我们逐步去发现的。** 


```python
df[df["年代"] > 2018] #异常值
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>名字</th>
      <th>投票人数</th>
      <th>类型</th>
      <th>产地</th>
      <th>上映时间</th>
      <th>时长</th>
      <th>年代</th>
      <th>评分</th>
      <th>首映地点</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>13882</th>
      <td>武之舞</td>
      <td>128</td>
      <td>纪录片</td>
      <td>中国大陆</td>
      <td>1997-02-01 00:00:00</td>
      <td>60</td>
      <td>34943</td>
      <td>9.9</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>17115</th>
      <td>妈妈回来吧-中国打工村的孩子</td>
      <td>49</td>
      <td>纪录片</td>
      <td>日本</td>
      <td>2007-04-08 00:00:00</td>
      <td>109</td>
      <td>39180</td>
      <td>8.9</td>
      <td>美国</td>
    </tr>
  </tbody>
</table>



```python
df[df["时长"] > 1000] #异常值
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>名字</th>
      <th>投票人数</th>
      <th>类型</th>
      <th>产地</th>
      <th>上映时间</th>
      <th>时长</th>
      <th>年代</th>
      <th>评分</th>
      <th>首映地点</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>19690</th>
      <td>怒海余生</td>
      <td>54</td>
      <td>剧情/家庭/冒险</td>
      <td>美国</td>
      <td>1937-09-01 00:00:00</td>
      <td>11500</td>
      <td>1937</td>
      <td>7.9</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>38730</th>
      <td>喧闹村的孩子们</td>
      <td>36</td>
      <td>家庭</td>
      <td>瑞典</td>
      <td>1986-12-06 00:00:00</td>
      <td>9200</td>
      <td>1986</td>
      <td>8.7</td>
      <td>瑞典</td>
    </tr>
  </tbody>
</table>



```python
df.drop(df[df["年代"] > 2018].index, inplace = True)
df.drop(df[df["时长"] > 1000].index, inplace = True) #删除异常数据
```


```python
df.index = range(len(df)) #解决删除后索引不连续的问题
```

不要忽视**索引不连续**！

### 3.8.2 最值 max()/min()

```python
df["投票人数"].max()
#692795

df["投票人数"].min()
#21

df["评分"].max()
#9.9

df["评分"].min()
#2.0

df["年代"].min()
#1888
```



### 3.8.3 均值和中值 mean()/median()

```python
df["投票人数"].mean()
#6268.7812802976705

df["投票人数"].median()
#354.0

df["评分"].mean()
#6.921951515969828

df["评分"].median()
#7.1
```



### 3.8.4 方差和标准差 var()/std()

```python
df["评分"].var()
#1.5968697056255758

df["评分"].std()
#1.263673100776295
```



### 3.8.5 求和 sum()

```python
df["投票人数"].sum()
#239235500
```



### 3.8.6 相关系数和协方差 corr()/cov()

```python
df[["投票人数", "评分"]].corr()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>投票人数</th>
      <th>评分</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>投票人数</th>
      <td>1.000000</td>
      <td>0.126953</td>
    </tr>
    <tr>
      <th>评分</th>
      <td>0.126953</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>



```python
df[["投票人数", "评分"]].cov()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>投票人数</th>
      <th>评分</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>投票人数</th>
      <td>6.916707e+08</td>
      <td>4219.174348</td>
    </tr>
    <tr>
      <th>评分</th>
      <td>4.219174e+03</td>
      <td>1.596870</td>
    </tr>
  </tbody>
</table>




### 3.8.7 计数

```python
len(df)
```


    38163


```python
df["产地"].unique() #指定唯一值
```


    array(['美国', '意大利', '中国大陆', '日本', '法国', '英国', '韩国', '中国香港', '阿根廷', '德国',
           '印度', '其他', '加拿大', '波兰', '泰国', '澳大利亚', '西班牙', '俄罗斯', '中国台湾', '荷兰',
           '丹麦', '比利时', '巴西', '瑞典', '墨西哥'], dtype=object)


```python
len(df["产地"].unique())
```


    25

产地中包含了一些重复的数据，比如美国和USA，德国和西德，俄罗斯和苏联
我们可以通过**数据替换**的方法将这些相同国家的电影数据合并一下。

**replace():替换**

**unique():独特**


```python
df["产地"].replace("USA","美国",inplace = True) #第一个参数是要替换的值，第二个参数是替换后的值
```


```python
df["产地"].replace(["西德","苏联"],["德国","俄罗斯"], inplace = True) #注意一一对应
```


```python
len(df["产地"].unique())
```


    25


```python
df["年代"].unique()
```


    array([1994, 1957, 1997, 1993, 2012, 2013, 2003, 2016, 2009, 2008, 2001,
           1931, 1961, 2010, 2004, 1998, 1972, 1939, 2015, 1946, 2011, 1982,
           1960, 2006, 1988, 2002, 1995, 1996, 1984, 2014, 1953, 2007, 2000,
           1967, 1983, 1963, 1977, 1966, 1971, 1974, 1985, 1987, 1973, 1962,
           1969, 1989, 1979, 1981, 1936, 1954, 1992, 1970, 1991, 2005, 1920,
           1933, 1990, 1999, 1896, 1965, 1921, 1947, 1975, 1964, 1943, 1928,
           1986, 1895, 1949, 1932, 1905, 1940, 1908, 1900, 1978, 1951, 1958,
           1898, 1976, 1938, 1907, 1948, 1952, 1926, 1955, 1906, 1959, 1934,
           1944, 1888, 1909, 1925, 1956, 1923, 1945, 1913, 1903, 1904, 1980,
           1968, 1917, 1935, 1942, 1950, 1902, 1941, 1930, 1937, 1922, 1916,
           1929, 1927, 1919, 1914, 1912, 1924, 1918, 1899, 1901, 1915, 1892,
           1894, 1910, 1897, 1911, 1890, 2018])


```python
len(df["年代"].unique())
```


    127

**计算每一年电影的数量：value_counts(ascending = <Bool>)[:]**


```python
df["年代"].value_counts(ascending = True)[:10] #默认从大到小
```


    1890    1
    2018    1
    1892    1
    1899    2
    1898    2
    1888    2
    1894    3
    1897    3
    1911    3
    1909    4
    Name: 年代, dtype: int64

电影产出前5的国家或地区：


```python
df["产地"].value_counts()[:5]
```


    美国      11714
    日本       5006
    中国大陆     3791
    中国香港     2847
    法国       2787
    Name: 产地, dtype: int64

保存数据


```python
df.to_excel("movie_data2.xlsx")
```


## 3.9 数据透视
Excel中数据透视表的使用非常广泛，其实Pandas也提供了一个类似的功能，名为**pivot_table**。

pivot_table非常有用，我们将重点解释pandas中的函数pivot_table。

使用pandas中的pivot_table的一个挑战是，你需要确保你理解你的数据，并清楚地知道你想通过透视表解决什么问题。虽然pivot_table看起来只是一个简单的函数，但是它能够快速地对数据进行强大的分析。

### 3.9.1 基础形式
```python
pd.set_option("max_columns",100) #设置可展示的行和列，让数据进行完整展示
pd.set_option("max_rows",500)
```

```python
pd.pivot_table(df, index = ["年代"]) #统计各个年代中所有数值型数据的均值（默认）
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>投票人数</th>
      <th>时长</th>
      <th>评分</th>
    </tr>
    <tr>
      <th>年代</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1888</th>
      <td>388.000000</td>
      <td>60.000000</td>
      <td>7.950000</td>
    </tr>
    <tr>
      <th>1890</th>
      <td>51.000000</td>
      <td>60.000000</td>
      <td>4.800000</td>
    </tr>
    <tr>
      <th>1892</th>
      <td>176.000000</td>
      <td>60.000000</td>
      <td>7.500000</td>
    </tr>
    <tr>
      <th>1894</th>
      <td>112.666667</td>
      <td>60.000000</td>
      <td>6.633333</td>
    </tr>
    <tr>
      <th>1895</th>
      <td>959.875000</td>
      <td>60.000000</td>
      <td>7.575000</td>
    </tr>
    <tr>
      <th>1896</th>
      <td>984.250000</td>
      <td>60.000000</td>
      <td>7.037500</td>
    </tr>
    <tr>
      <th>1897</th>
      <td>67.000000</td>
      <td>60.000000</td>
      <td>6.633333</td>
    </tr>
    <tr>
      <th>1898</th>
      <td>578.500000</td>
      <td>60.000000</td>
      <td>7.450000</td>
    </tr>
    <tr>
      <th>1899</th>
      <td>71.000000</td>
      <td>9.500000</td>
      <td>6.900000</td>
    </tr>
    <tr>
      <th>1900</th>
      <td>175.285714</td>
      <td>36.714286</td>
      <td>7.228571</td>
    </tr>
    <tr>
      <th>1901</th>
      <td>164.500000</td>
      <td>47.250000</td>
      <td>7.250000</td>
    </tr>
    <tr>
      <th>1902</th>
      <td>2309.600000</td>
      <td>29.600000</td>
      <td>7.680000</td>
    </tr>
    <tr>
      <th>1903</th>
      <td>349.846154</td>
      <td>27.000000</td>
      <td>7.015385</td>
    </tr>
    <tr>
      <th>1904</th>
      <td>249.166667</td>
      <td>35.000000</td>
      <td>7.616667</td>
    </tr>
    <tr>
      <th>1905</th>
      <td>332.600000</td>
      <td>43.800000</td>
      <td>6.820000</td>
    </tr>
    <tr>
      <th>1906</th>
      <td>189.857143</td>
      <td>30.571429</td>
      <td>7.342857</td>
    </tr>
    <tr>
      <th>1907</th>
      <td>213.600000</td>
      <td>31.800000</td>
      <td>7.020000</td>
    </tr>
    <tr>
      <th>1908</th>
      <td>368.750000</td>
      <td>35.000000</td>
      <td>7.425000</td>
    </tr>
    <tr>
      <th>1909</th>
      <td>62.000000</td>
      <td>10.750000</td>
      <td>7.650000</td>
    </tr>
    <tr>
      <th>1910</th>
      <td>105.200000</td>
      <td>41.800000</td>
      <td>6.940000</td>
    </tr>
    <tr>
      <th>1911</th>
      <td>308.000000</td>
      <td>28.666667</td>
      <td>7.500000</td>
    </tr>
    <tr>
      <th>1912</th>
      <td>181.000000</td>
      <td>18.000000</td>
      <td>7.925000</td>
    </tr>
    <tr>
      <th>1913</th>
      <td>62.285714</td>
      <td>64.571429</td>
      <td>6.671429</td>
    </tr>
    <tr>
      <th>1914</th>
      <td>104.923077</td>
      <td>25.923077</td>
      <td>6.473077</td>
    </tr>
    <tr>
      <th>1915</th>
      <td>314.900000</td>
      <td>56.800000</td>
      <td>7.260000</td>
    </tr>
    <tr>
      <th>1916</th>
      <td>666.636364</td>
      <td>42.363636</td>
      <td>7.690909</td>
    </tr>
    <tr>
      <th>1917</th>
      <td>124.416667</td>
      <td>31.333333</td>
      <td>7.075000</td>
    </tr>
    <tr>
      <th>1918</th>
      <td>357.083333</td>
      <td>35.166667</td>
      <td>7.200000</td>
    </tr>
    <tr>
      <th>1919</th>
      <td>194.777778</td>
      <td>64.611111</td>
      <td>7.494444</td>
    </tr>
    <tr>
      <th>1920</th>
      <td>636.500000</td>
      <td>59.357143</td>
      <td>7.492857</td>
    </tr>
    <tr>
      <th>1921</th>
      <td>729.818182</td>
      <td>57.363636</td>
      <td>7.750000</td>
    </tr>
    <tr>
      <th>1922</th>
      <td>767.090909</td>
      <td>66.363636</td>
      <td>7.804545</td>
    </tr>
    <tr>
      <th>1923</th>
      <td>447.705882</td>
      <td>74.882353</td>
      <td>7.811765</td>
    </tr>
    <tr>
      <th>1924</th>
      <td>384.518519</td>
      <td>81.962963</td>
      <td>8.059259</td>
    </tr>
    <tr>
      <th>1925</th>
      <td>1104.280000</td>
      <td>84.440000</td>
      <td>7.788000</td>
    </tr>
    <tr>
      <th>1926</th>
      <td>443.608696</td>
      <td>80.304348</td>
      <td>7.773913</td>
    </tr>
    <tr>
      <th>1927</th>
      <td>695.275862</td>
      <td>87.241379</td>
      <td>7.751724</td>
    </tr>
    <tr>
      <th>1928</th>
      <td>413.666667</td>
      <td>72.076923</td>
      <td>7.964103</td>
    </tr>
    <tr>
      <th>1929</th>
      <td>740.542857</td>
      <td>69.371429</td>
      <td>7.440000</td>
    </tr>
    <tr>
      <th>1930</th>
      <td>555.080000</td>
      <td>74.160000</td>
      <td>7.360000</td>
    </tr>
    <tr>
      <th>1931</th>
      <td>1468.666667</td>
      <td>78.523810</td>
      <td>7.483333</td>
    </tr>
    <tr>
      <th>1932</th>
      <td>600.081081</td>
      <td>77.540541</td>
      <td>7.294595</td>
    </tr>
    <tr>
      <th>1933</th>
      <td>756.020833</td>
      <td>79.187500</td>
      <td>7.420833</td>
    </tr>
    <tr>
      <th>1934</th>
      <td>791.460000</td>
      <td>83.260000</td>
      <td>7.536000</td>
    </tr>
    <tr>
      <th>1935</th>
      <td>887.695652</td>
      <td>73.673913</td>
      <td>7.515217</td>
    </tr>
    <tr>
      <th>1936</th>
      <td>1489.220339</td>
      <td>77.440678</td>
      <td>7.615254</td>
    </tr>
    <tr>
      <th>1937</th>
      <td>1612.104167</td>
      <td>87.187500</td>
      <td>7.568750</td>
    </tr>
    <tr>
      <th>1938</th>
      <td>552.000000</td>
      <td>85.973684</td>
      <td>7.736842</td>
    </tr>
    <tr>
      <th>1939</th>
      <td>5911.857143</td>
      <td>97.387755</td>
      <td>7.520408</td>
    </tr>
    <tr>
      <th>1940</th>
      <td>5689.815789</td>
      <td>93.684211</td>
      <td>7.544737</td>
    </tr>
    <tr>
      <th>1941</th>
      <td>1552.808511</td>
      <td>89.127660</td>
      <td>7.427660</td>
    </tr>
    <tr>
      <th>1942</th>
      <td>2607.754717</td>
      <td>78.264151</td>
      <td>7.554717</td>
    </tr>
    <tr>
      <th>1943</th>
      <td>755.357143</td>
      <td>79.714286</td>
      <td>7.605357</td>
    </tr>
    <tr>
      <th>1944</th>
      <td>1007.370370</td>
      <td>81.925926</td>
      <td>7.538889</td>
    </tr>
    <tr>
      <th>1945</th>
      <td>989.020408</td>
      <td>86.959184</td>
      <td>7.673469</td>
    </tr>
    <tr>
      <th>1946</th>
      <td>1034.457627</td>
      <td>85.016949</td>
      <td>7.606780</td>
    </tr>
    <tr>
      <th>1947</th>
      <td>443.702703</td>
      <td>87.486486</td>
      <td>7.502703</td>
    </tr>
    <tr>
      <th>1948</th>
      <td>1199.255814</td>
      <td>88.534884</td>
      <td>7.645349</td>
    </tr>
    <tr>
      <th>1949</th>
      <td>641.685393</td>
      <td>81.988764</td>
      <td>7.646067</td>
    </tr>
    <tr>
      <th>1950</th>
      <td>2235.026316</td>
      <td>80.157895</td>
      <td>7.655263</td>
    </tr>
    <tr>
      <th>1951</th>
      <td>967.884615</td>
      <td>86.653846</td>
      <td>7.637179</td>
    </tr>
    <tr>
      <th>1952</th>
      <td>1507.305882</td>
      <td>82.658824</td>
      <td>7.775294</td>
    </tr>
    <tr>
      <th>1953</th>
      <td>4840.620690</td>
      <td>84.448276</td>
      <td>7.579310</td>
    </tr>
    <tr>
      <th>1954</th>
      <td>2201.356436</td>
      <td>86.326733</td>
      <td>7.714851</td>
    </tr>
    <tr>
      <th>1955</th>
      <td>2000.491228</td>
      <td>82.912281</td>
      <td>7.567544</td>
    </tr>
    <tr>
      <th>1956</th>
      <td>1061.675862</td>
      <td>76.944828</td>
      <td>7.591724</td>
    </tr>
    <tr>
      <th>1957</th>
      <td>3057.586538</td>
      <td>88.884615</td>
      <td>7.622115</td>
    </tr>
    <tr>
      <th>1958</th>
      <td>886.196721</td>
      <td>82.975410</td>
      <td>7.536885</td>
    </tr>
    <tr>
      <th>1959</th>
      <td>1725.070312</td>
      <td>90.070312</td>
      <td>7.571875</td>
    </tr>
    <tr>
      <th>1960</th>
      <td>1457.107692</td>
      <td>101.530769</td>
      <td>7.580769</td>
    </tr>
    <tr>
      <th>1961</th>
      <td>3249.750000</td>
      <td>99.510000</td>
      <td>7.741000</td>
    </tr>
    <tr>
      <th>1962</th>
      <td>1985.661972</td>
      <td>92.225352</td>
      <td>7.707042</td>
    </tr>
    <tr>
      <th>1963</th>
      <td>1184.027972</td>
      <td>92.020979</td>
      <td>7.536364</td>
    </tr>
    <tr>
      <th>1964</th>
      <td>1125.441341</td>
      <td>91.162011</td>
      <td>7.540782</td>
    </tr>
    <tr>
      <th>1965</th>
      <td>2012.248447</td>
      <td>91.434783</td>
      <td>7.591304</td>
    </tr>
    <tr>
      <th>1966</th>
      <td>1326.640449</td>
      <td>89.651685</td>
      <td>7.521910</td>
    </tr>
    <tr>
      <th>1967</th>
      <td>1243.891429</td>
      <td>92.074286</td>
      <td>7.477143</td>
    </tr>
    <tr>
      <th>1968</th>
      <td>1107.841176</td>
      <td>92.594118</td>
      <td>7.324706</td>
    </tr>
    <tr>
      <th>1969</th>
      <td>601.279412</td>
      <td>99.475490</td>
      <td>7.367647</td>
    </tr>
    <tr>
      <th>1970</th>
      <td>683.468085</td>
      <td>97.484043</td>
      <td>7.294149</td>
    </tr>
    <tr>
      <th>1971</th>
      <td>1380.576037</td>
      <td>96.695853</td>
      <td>7.149309</td>
    </tr>
    <tr>
      <th>1972</th>
      <td>2332.132353</td>
      <td>96.372549</td>
      <td>7.253431</td>
    </tr>
    <tr>
      <th>1973</th>
      <td>810.809524</td>
      <td>95.471429</td>
      <td>7.238095</td>
    </tr>
    <tr>
      <th>1974</th>
      <td>1693.171171</td>
      <td>95.477477</td>
      <td>7.063964</td>
    </tr>
    <tr>
      <th>1975</th>
      <td>2256.704663</td>
      <td>97.813472</td>
      <td>7.056995</td>
    </tr>
    <tr>
      <th>1976</th>
      <td>1340.618026</td>
      <td>95.055794</td>
      <td>7.107725</td>
    </tr>
    <tr>
      <th>1977</th>
      <td>965.444954</td>
      <td>98.197248</td>
      <td>7.151376</td>
    </tr>
    <tr>
      <th>1978</th>
      <td>1001.820896</td>
      <td>94.467662</td>
      <td>7.096517</td>
    </tr>
    <tr>
      <th>1979</th>
      <td>1902.991071</td>
      <td>96.250000</td>
      <td>7.292857</td>
    </tr>
    <tr>
      <th>1980</th>
      <td>2417.040359</td>
      <td>94.443946</td>
      <td>7.182063</td>
    </tr>
    <tr>
      <th>1981</th>
      <td>1620.566176</td>
      <td>93.709559</td>
      <td>7.154044</td>
    </tr>
    <tr>
      <th>1982</th>
      <td>2174.809160</td>
      <td>92.889313</td>
      <td>7.286260</td>
    </tr>
    <tr>
      <th>1983</th>
      <td>1551.894545</td>
      <td>91.985455</td>
      <td>7.296727</td>
    </tr>
    <tr>
      <th>1984</th>
      <td>2676.129693</td>
      <td>91.023891</td>
      <td>7.380887</td>
    </tr>
    <tr>
      <th>1985</th>
      <td>1612.430380</td>
      <td>93.974684</td>
      <td>7.267722</td>
    </tr>
    <tr>
      <th>1986</th>
      <td>2836.777090</td>
      <td>89.235294</td>
      <td>7.249536</td>
    </tr>
    <tr>
      <th>1987</th>
      <td>3146.832845</td>
      <td>89.269795</td>
      <td>7.282111</td>
    </tr>
    <tr>
      <th>1988</th>
      <td>4859.352332</td>
      <td>89.292746</td>
      <td>7.265544</td>
    </tr>
    <tr>
      <th>1989</th>
      <td>3009.383632</td>
      <td>88.501279</td>
      <td>7.199233</td>
    </tr>
    <tr>
      <th>1990</th>
      <td>5003.312821</td>
      <td>91.976923</td>
      <td>7.156923</td>
    </tr>
    <tr>
      <th>1991</th>
      <td>4708.414216</td>
      <td>93.661765</td>
      <td>7.154412</td>
    </tr>
    <tr>
      <th>1992</th>
      <td>5573.535047</td>
      <td>92.670561</td>
      <td>7.190187</td>
    </tr>
    <tr>
      <th>1993</th>
      <td>8154.555809</td>
      <td>95.473804</td>
      <td>7.186560</td>
    </tr>
    <tr>
      <th>1994</th>
      <td>11591.339468</td>
      <td>91.554192</td>
      <td>7.257260</td>
    </tr>
    <tr>
      <th>1995</th>
      <td>8266.887734</td>
      <td>95.114345</td>
      <td>7.275052</td>
    </tr>
    <tr>
      <th>1996</th>
      <td>5154.665362</td>
      <td>95.819961</td>
      <td>7.249119</td>
    </tr>
    <tr>
      <th>1997</th>
      <td>8383.681818</td>
      <td>95.446970</td>
      <td>7.325758</td>
    </tr>
    <tr>
      <th>1998</th>
      <td>7587.203540</td>
      <td>92.021239</td>
      <td>7.223540</td>
    </tr>
    <tr>
      <th>1999</th>
      <td>7228.077796</td>
      <td>92.914100</td>
      <td>7.171151</td>
    </tr>
    <tr>
      <th>2000</th>
      <td>6022.013947</td>
      <td>92.917713</td>
      <td>7.112413</td>
    </tr>
    <tr>
      <th>2001</th>
      <td>8740.049751</td>
      <td>91.342040</td>
      <td>7.070647</td>
    </tr>
    <tr>
      <th>2002</th>
      <td>6705.952218</td>
      <td>92.268487</td>
      <td>7.045620</td>
    </tr>
    <tr>
      <th>2003</th>
      <td>8104.751896</td>
      <td>90.695558</td>
      <td>7.114410</td>
    </tr>
    <tr>
      <th>2004</th>
      <td>8758.644749</td>
      <td>91.376256</td>
      <td>6.999909</td>
    </tr>
    <tr>
      <th>2005</th>
      <td>6626.300242</td>
      <td>92.804681</td>
      <td>7.011864</td>
    </tr>
    <tr>
      <th>2006</th>
      <td>7407.740591</td>
      <td>90.619624</td>
      <td>6.901546</td>
    </tr>
    <tr>
      <th>2007</th>
      <td>6346.090801</td>
      <td>86.842136</td>
      <td>6.859703</td>
    </tr>
    <tr>
      <th>2008</th>
      <td>6876.232486</td>
      <td>85.589517</td>
      <td>6.892164</td>
    </tr>
    <tr>
      <th>2009</th>
      <td>8694.675558</td>
      <td>86.681546</td>
      <td>6.732825</td>
    </tr>
    <tr>
      <th>2010</th>
      <td>9691.044576</td>
      <td>83.785177</td>
      <td>6.752793</td>
    </tr>
    <tr>
      <th>2011</th>
      <td>8860.328455</td>
      <td>84.437398</td>
      <td>6.560325</td>
    </tr>
    <tr>
      <th>2012</th>
      <td>7165.750248</td>
      <td>85.321606</td>
      <td>6.444896</td>
    </tr>
    <tr>
      <th>2013</th>
      <td>7694.106727</td>
      <td>85.337380</td>
      <td>6.375974</td>
    </tr>
    <tr>
      <th>2014</th>
      <td>7803.983931</td>
      <td>86.354580</td>
      <td>6.249384</td>
    </tr>
    <tr>
      <th>2015</th>
      <td>7954.999363</td>
      <td>90.338432</td>
      <td>6.121925</td>
    </tr>
    <tr>
      <th>2016</th>
      <td>7341.388889</td>
      <td>91.646825</td>
      <td>5.834524</td>
    </tr>
    <tr>
      <th>2018</th>
      <td>123456.000000</td>
      <td>142.000000</td>
      <td>6.935704</td>
    </tr>
  </tbody>
</table>


### 3.9.2 多索引
也可以有多个索引。实际上，大多数的pivot_table参数可以通过列表获取多个值。

```python
pd.pivot_table(df, index = ["年代", "产地"]) #双索引
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>投票人数</th>
      <th>时长</th>
      <th>评分</th>
    </tr>
    <tr>
      <th>年代</th>
      <th>产地</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1888</th>
      <th>英国</th>
      <td>388.000000</td>
      <td>60.000000</td>
      <td>7.950000</td>
    </tr>
    <tr>
      <th>1890</th>
      <th>美国</th>
      <td>51.000000</td>
      <td>60.000000</td>
      <td>4.800000</td>
    </tr>
    <tr>
      <th>1892</th>
      <th>法国</th>
      <td>176.000000</td>
      <td>60.000000</td>
      <td>7.500000</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">1894</th>
      <th>法国</th>
      <td>148.000000</td>
      <td>60.000000</td>
      <td>7.000000</td>
    </tr>
    <tr>
      <th>美国</th>
      <td>95.000000</td>
      <td>60.000000</td>
      <td>6.450000</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">2016</th>
      <th>法国</th>
      <td>44.666667</td>
      <td>104.333333</td>
      <td>7.300000</td>
    </tr>
    <tr>
      <th>美国</th>
      <td>11224.225806</td>
      <td>93.161290</td>
      <td>6.522581</td>
    </tr>
    <tr>
      <th>英国</th>
      <td>14607.272727</td>
      <td>85.545455</td>
      <td>7.200000</td>
    </tr>
    <tr>
      <th>韩国</th>
      <td>1739.850000</td>
      <td>106.100000</td>
      <td>5.730000</td>
    </tr>
    <tr>
      <th>2018</th>
      <th>美国</th>
      <td>123456.000000</td>
      <td>142.000000</td>
      <td>6.935704</td>
    </tr>
  </tbody>
</table>
<p>1578 rows × 3 columns</p>


### 3.9.3 指定需要统计汇总的数据
```python
pd.pivot_table(df, index = ["年代", "产地"], values = ["评分"])
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>评分</th>
    </tr>
    <tr>
      <th>年代</th>
      <th>产地</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1888</th>
      <th>英国</th>
      <td>7.950000</td>
    </tr>
    <tr>
      <th>1890</th>
      <th>美国</th>
      <td>4.800000</td>
    </tr>
    <tr>
      <th>1892</th>
      <th>法国</th>
      <td>7.500000</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">1894</th>
      <th>法国</th>
      <td>7.000000</td>
    </tr>
    <tr>
      <th>美国</th>
      <td>6.450000</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">2016</th>
      <th>法国</th>
      <td>7.300000</td>
    </tr>
    <tr>
      <th>美国</th>
      <td>6.522581</td>
    </tr>
    <tr>
      <th>英国</th>
      <td>7.200000</td>
    </tr>
    <tr>
      <th>韩国</th>
      <td>5.730000</td>
    </tr>
    <tr>
      <th>2018</th>
      <th>美国</th>
      <td>6.935704</td>
    </tr>
  </tbody>
</table>
<p>1578 rows × 1 columns</p>


### 3.9.4 指定函数统计不同统计值
```python
pd.pivot_table(df, index = ["年代", "产地"], values = ["投票人数"], aggfunc = np.sum)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>投票人数</th>
    </tr>
    <tr>
      <th>年代</th>
      <th>产地</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1888</th>
      <th>英国</th>
      <td>776</td>
    </tr>
    <tr>
      <th>1890</th>
      <th>美国</th>
      <td>51</td>
    </tr>
    <tr>
      <th>1892</th>
      <th>法国</th>
      <td>176</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">1894</th>
      <th>法国</th>
      <td>148</td>
    </tr>
    <tr>
      <th>美国</th>
      <td>190</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">2016</th>
      <th>法国</th>
      <td>134</td>
    </tr>
    <tr>
      <th>美国</th>
      <td>695902</td>
    </tr>
    <tr>
      <th>英国</th>
      <td>160680</td>
    </tr>
    <tr>
      <th>韩国</th>
      <td>34797</td>
    </tr>
    <tr>
      <th>2018</th>
      <th>美国</th>
      <td>123456</td>
    </tr>
  </tbody>
</table>
<p>1578 rows × 1 columns</p>


通过将“投票人数”列和“评分”列进行对应分组，对“产地”实现数据聚合和总结。

```python
pd.pivot_table(df, index = ["产地"], values = ["投票人数", "评分"], aggfunc = [np.sum, np.mean])
```


<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="2" halign="left">sum</th>
      <th colspan="2" halign="left">mean</th>
    </tr>
    <tr>
      <th></th>
      <th>投票人数</th>
      <th>评分</th>
      <th>投票人数</th>
      <th>评分</th>
    </tr>
    <tr>
      <th>产地</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>中国台湾</th>
      <td>5237466</td>
      <td>4367.200000</td>
      <td>8474.864078</td>
      <td>7.066667</td>
    </tr>
    <tr>
      <th>中国大陆</th>
      <td>41380993</td>
      <td>22984.800000</td>
      <td>10915.587708</td>
      <td>6.062991</td>
    </tr>
    <tr>
      <th>中国香港</th>
      <td>23179448</td>
      <td>18430.200000</td>
      <td>8141.709870</td>
      <td>6.473551</td>
    </tr>
    <tr>
      <th>丹麦</th>
      <td>394745</td>
      <td>1427.600000</td>
      <td>2003.781726</td>
      <td>7.246701</td>
    </tr>
    <tr>
      <th>俄罗斯</th>
      <td>486082</td>
      <td>3597.200000</td>
      <td>1021.180672</td>
      <td>7.557143</td>
    </tr>
    <tr>
      <th>其他</th>
      <td>3048849</td>
      <td>13607.900000</td>
      <td>1619.144450</td>
      <td>7.226713</td>
    </tr>
    <tr>
      <th>加拿大</th>
      <td>1362581</td>
      <td>4769.600000</td>
      <td>1921.834979</td>
      <td>6.727221</td>
    </tr>
    <tr>
      <th>印度</th>
      <td>1146173</td>
      <td>2443.900000</td>
      <td>3219.587079</td>
      <td>6.864888</td>
    </tr>
    <tr>
      <th>墨西哥</th>
      <td>139462</td>
      <td>829.000000</td>
      <td>1191.982906</td>
      <td>7.085470</td>
    </tr>
    <tr>
      <th>巴西</th>
      <td>357027</td>
      <td>716.000000</td>
      <td>3606.333333</td>
      <td>7.232323</td>
    </tr>
    <tr>
      <th>德国</th>
      <td>2679856</td>
      <td>7338.300000</td>
      <td>2624.736533</td>
      <td>7.187365</td>
    </tr>
    <tr>
      <th>意大利</th>
      <td>2500842</td>
      <td>5322.700000</td>
      <td>3374.955466</td>
      <td>7.183131</td>
    </tr>
    <tr>
      <th>日本</th>
      <td>17981631</td>
      <td>36006.000000</td>
      <td>3592.015781</td>
      <td>7.192569</td>
    </tr>
    <tr>
      <th>比利时</th>
      <td>170449</td>
      <td>986.000000</td>
      <td>1244.153285</td>
      <td>7.197080</td>
    </tr>
    <tr>
      <th>法国</th>
      <td>10208966</td>
      <td>20186.500000</td>
      <td>3663.066380</td>
      <td>7.243093</td>
    </tr>
    <tr>
      <th>波兰</th>
      <td>159577</td>
      <td>1347.000000</td>
      <td>881.640884</td>
      <td>7.441989</td>
    </tr>
    <tr>
      <th>泰国</th>
      <td>1564881</td>
      <td>1796.100000</td>
      <td>5322.724490</td>
      <td>6.109184</td>
    </tr>
    <tr>
      <th>澳大利亚</th>
      <td>1415443</td>
      <td>2051.300000</td>
      <td>4798.111864</td>
      <td>6.953559</td>
    </tr>
    <tr>
      <th>瑞典</th>
      <td>289794</td>
      <td>1388.600000</td>
      <td>1549.700535</td>
      <td>7.425668</td>
    </tr>
    <tr>
      <th>美国</th>
      <td>101645832</td>
      <td>81100.135704</td>
      <td>8677.294861</td>
      <td>6.923351</td>
    </tr>
    <tr>
      <th>英国</th>
      <td>13236409</td>
      <td>19930.800000</td>
      <td>4979.837848</td>
      <td>7.498420</td>
    </tr>
    <tr>
      <th>荷兰</th>
      <td>144596</td>
      <td>1081.200000</td>
      <td>957.589404</td>
      <td>7.160265</td>
    </tr>
    <tr>
      <th>西班牙</th>
      <td>1486383</td>
      <td>3112.100000</td>
      <td>3355.266366</td>
      <td>7.025056</td>
    </tr>
    <tr>
      <th>阿根廷</th>
      <td>258085</td>
      <td>819.100000</td>
      <td>2283.938053</td>
      <td>7.248673</td>
    </tr>
    <tr>
      <th>韩国</th>
      <td>8759930</td>
      <td>8523.200000</td>
      <td>6527.518629</td>
      <td>6.351118</td>
    </tr>
  </tbody>
</table>




### 3.9.5 非数值(NaN)的处理
非数值（NaN）难以处理。如果想移除它们，可以使用“fill_value”将其设置为0。
```python
pd.pivot_table(df, index = ["产地"], aggfunc = [np.sum, np.mean], fill_value = 0)
```


<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="4" halign="left">sum</th>
      <th colspan="4" halign="left">mean</th>
    </tr>
    <tr>
      <th></th>
      <th>年代</th>
      <th>投票人数</th>
      <th>时长</th>
      <th>评分</th>
      <th>年代</th>
      <th>投票人数</th>
      <th>时长</th>
      <th>评分</th>
    </tr>
    <tr>
      <th>产地</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>中国台湾</th>
      <td>1235388</td>
      <td>5237466</td>
      <td>53925</td>
      <td>4367.200000</td>
      <td>1999.009709</td>
      <td>8474.864078</td>
      <td>87.257282</td>
      <td>7.066667</td>
    </tr>
    <tr>
      <th>中国大陆</th>
      <td>7599372</td>
      <td>41380993</td>
      <td>309031</td>
      <td>22984.800000</td>
      <td>2004.582432</td>
      <td>10915.587708</td>
      <td>81.517014</td>
      <td>6.062991</td>
    </tr>
    <tr>
      <th>中国香港</th>
      <td>5668630</td>
      <td>23179448</td>
      <td>252111</td>
      <td>18430.200000</td>
      <td>1991.088865</td>
      <td>8141.709870</td>
      <td>88.553214</td>
      <td>6.473551</td>
    </tr>
    <tr>
      <th>丹麦</th>
      <td>393821</td>
      <td>394745</td>
      <td>17436</td>
      <td>1427.600000</td>
      <td>1999.091371</td>
      <td>2003.781726</td>
      <td>88.507614</td>
      <td>7.246701</td>
    </tr>
    <tr>
      <th>俄罗斯</th>
      <td>944809</td>
      <td>486082</td>
      <td>45744</td>
      <td>3597.200000</td>
      <td>1984.892857</td>
      <td>1021.180672</td>
      <td>96.100840</td>
      <td>7.557143</td>
    </tr>
    <tr>
      <th>其他</th>
      <td>3763593</td>
      <td>3048849</td>
      <td>165057</td>
      <td>13607.900000</td>
      <td>1998.721721</td>
      <td>1619.144450</td>
      <td>87.656399</td>
      <td>7.226713</td>
    </tr>
    <tr>
      <th>加拿大</th>
      <td>1419787</td>
      <td>1362581</td>
      <td>57140</td>
      <td>4769.600000</td>
      <td>2002.520451</td>
      <td>1921.834979</td>
      <td>80.592384</td>
      <td>6.727221</td>
    </tr>
    <tr>
      <th>印度</th>
      <td>714150</td>
      <td>1146173</td>
      <td>43058</td>
      <td>2443.900000</td>
      <td>2006.039326</td>
      <td>3219.587079</td>
      <td>120.949438</td>
      <td>6.864888</td>
    </tr>
    <tr>
      <th>墨西哥</th>
      <td>233156</td>
      <td>139462</td>
      <td>10839</td>
      <td>829.000000</td>
      <td>1992.786325</td>
      <td>1191.982906</td>
      <td>92.641026</td>
      <td>7.085470</td>
    </tr>
    <tr>
      <th>巴西</th>
      <td>197989</td>
      <td>357027</td>
      <td>8749</td>
      <td>716.000000</td>
      <td>1999.888889</td>
      <td>3606.333333</td>
      <td>88.373737</td>
      <td>7.232323</td>
    </tr>
    <tr>
      <th>德国</th>
      <td>2037971</td>
      <td>2679856</td>
      <td>94196</td>
      <td>7338.300000</td>
      <td>1996.053869</td>
      <td>2624.736533</td>
      <td>92.258570</td>
      <td>7.187365</td>
    </tr>
    <tr>
      <th>意大利</th>
      <td>1471329</td>
      <td>2500842</td>
      <td>77311</td>
      <td>5322.700000</td>
      <td>1985.599190</td>
      <td>3374.955466</td>
      <td>104.333333</td>
      <td>7.183131</td>
    </tr>
    <tr>
      <th>日本</th>
      <td>10011432</td>
      <td>17981631</td>
      <td>425563</td>
      <td>36006.000000</td>
      <td>1999.886536</td>
      <td>3592.015781</td>
      <td>85.010587</td>
      <td>7.192569</td>
    </tr>
    <tr>
      <th>比利时</th>
      <td>273932</td>
      <td>170449</td>
      <td>11380</td>
      <td>986.000000</td>
      <td>1999.503650</td>
      <td>1244.153285</td>
      <td>83.065693</td>
      <td>7.197080</td>
    </tr>
    <tr>
      <th>法国</th>
      <td>5551130</td>
      <td>10208966</td>
      <td>251524</td>
      <td>20186.500000</td>
      <td>1991.794044</td>
      <td>3663.066380</td>
      <td>90.249013</td>
      <td>7.243093</td>
    </tr>
    <tr>
      <th>波兰</th>
      <td>359652</td>
      <td>159577</td>
      <td>14613</td>
      <td>1347.000000</td>
      <td>1987.027624</td>
      <td>881.640884</td>
      <td>80.734807</td>
      <td>7.441989</td>
    </tr>
    <tr>
      <th>泰国</th>
      <td>590684</td>
      <td>1564881</td>
      <td>26002</td>
      <td>1796.100000</td>
      <td>2009.129252</td>
      <td>5322.724490</td>
      <td>88.442177</td>
      <td>6.109184</td>
    </tr>
    <tr>
      <th>澳大利亚</th>
      <td>590875</td>
      <td>1415443</td>
      <td>25250</td>
      <td>2051.300000</td>
      <td>2002.966102</td>
      <td>4798.111864</td>
      <td>85.593220</td>
      <td>6.953559</td>
    </tr>
    <tr>
      <th>瑞典</th>
      <td>371589</td>
      <td>289794</td>
      <td>17695</td>
      <td>1388.600000</td>
      <td>1987.106952</td>
      <td>1549.700535</td>
      <td>94.625668</td>
      <td>7.425668</td>
    </tr>
    <tr>
      <th>美国</th>
      <td>23363806</td>
      <td>101645832</td>
      <td>1053980</td>
      <td>81100.135704</td>
      <td>1994.519891</td>
      <td>8677.294861</td>
      <td>89.976097</td>
      <td>6.923351</td>
    </tr>
    <tr>
      <th>英国</th>
      <td>5307045</td>
      <td>13236409</td>
      <td>237129</td>
      <td>19930.800000</td>
      <td>1996.630926</td>
      <td>4979.837848</td>
      <td>89.213318</td>
      <td>7.498420</td>
    </tr>
    <tr>
      <th>荷兰</th>
      <td>302181</td>
      <td>144596</td>
      <td>11459</td>
      <td>1081.200000</td>
      <td>2001.198675</td>
      <td>957.589404</td>
      <td>75.887417</td>
      <td>7.160265</td>
    </tr>
    <tr>
      <th>西班牙</th>
      <td>886685</td>
      <td>1486383</td>
      <td>40271</td>
      <td>3112.100000</td>
      <td>2001.546275</td>
      <td>3355.266366</td>
      <td>90.905192</td>
      <td>7.025056</td>
    </tr>
    <tr>
      <th>阿根廷</th>
      <td>226476</td>
      <td>258085</td>
      <td>10458</td>
      <td>819.100000</td>
      <td>2004.212389</td>
      <td>2283.938053</td>
      <td>92.548673</td>
      <td>7.248673</td>
    </tr>
    <tr>
      <th>韩国</th>
      <td>2694871</td>
      <td>8759930</td>
      <td>134225</td>
      <td>8523.200000</td>
      <td>2008.100596</td>
      <td>6527.518629</td>
      <td>100.018629</td>
      <td>6.351118</td>
    </tr>
  </tbody>
</table>


### 3.9.6 显示总和数据
加入margins = True，可以在下方显示一些总和数据。

```python
pd.pivot_table(df, index = ["产地"], aggfunc = [np.sum, np.mean], fill_value = 0, margins = True)
```



<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="4" halign="left">sum</th>
      <th colspan="4" halign="left">mean</th>
    </tr>
    <tr>
      <th></th>
      <th>年代</th>
      <th>投票人数</th>
      <th>时长</th>
      <th>评分</th>
      <th>年代</th>
      <th>投票人数</th>
      <th>时长</th>
      <th>评分</th>
    </tr>
    <tr>
      <th>产地</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>中国台湾</th>
      <td>1235388</td>
      <td>5237466</td>
      <td>53925</td>
      <td>4367.200000</td>
      <td>1999.009709</td>
      <td>8474.864078</td>
      <td>87.257282</td>
      <td>7.066667</td>
    </tr>
    <tr>
      <th>中国大陆</th>
      <td>7599372</td>
      <td>41380993</td>
      <td>309031</td>
      <td>22984.800000</td>
      <td>2004.582432</td>
      <td>10915.587708</td>
      <td>81.517014</td>
      <td>6.062991</td>
    </tr>
    <tr>
      <th>中国香港</th>
      <td>5668630</td>
      <td>23179448</td>
      <td>252111</td>
      <td>18430.200000</td>
      <td>1991.088865</td>
      <td>8141.709870</td>
      <td>88.553214</td>
      <td>6.473551</td>
    </tr>
    <tr>
      <th>丹麦</th>
      <td>393821</td>
      <td>394745</td>
      <td>17436</td>
      <td>1427.600000</td>
      <td>1999.091371</td>
      <td>2003.781726</td>
      <td>88.507614</td>
      <td>7.246701</td>
    </tr>
    <tr>
      <th>俄罗斯</th>
      <td>944809</td>
      <td>486082</td>
      <td>45744</td>
      <td>3597.200000</td>
      <td>1984.892857</td>
      <td>1021.180672</td>
      <td>96.100840</td>
      <td>7.557143</td>
    </tr>
    <tr>
      <th>其他</th>
      <td>3763593</td>
      <td>3048849</td>
      <td>165057</td>
      <td>13607.900000</td>
      <td>1998.721721</td>
      <td>1619.144450</td>
      <td>87.656399</td>
      <td>7.226713</td>
    </tr>
    <tr>
      <th>加拿大</th>
      <td>1419787</td>
      <td>1362581</td>
      <td>57140</td>
      <td>4769.600000</td>
      <td>2002.520451</td>
      <td>1921.834979</td>
      <td>80.592384</td>
      <td>6.727221</td>
    </tr>
    <tr>
      <th>印度</th>
      <td>714150</td>
      <td>1146173</td>
      <td>43058</td>
      <td>2443.900000</td>
      <td>2006.039326</td>
      <td>3219.587079</td>
      <td>120.949438</td>
      <td>6.864888</td>
    </tr>
    <tr>
      <th>墨西哥</th>
      <td>233156</td>
      <td>139462</td>
      <td>10839</td>
      <td>829.000000</td>
      <td>1992.786325</td>
      <td>1191.982906</td>
      <td>92.641026</td>
      <td>7.085470</td>
    </tr>
    <tr>
      <th>巴西</th>
      <td>197989</td>
      <td>357027</td>
      <td>8749</td>
      <td>716.000000</td>
      <td>1999.888889</td>
      <td>3606.333333</td>
      <td>88.373737</td>
      <td>7.232323</td>
    </tr>
    <tr>
      <th>德国</th>
      <td>2037971</td>
      <td>2679856</td>
      <td>94196</td>
      <td>7338.300000</td>
      <td>1996.053869</td>
      <td>2624.736533</td>
      <td>92.258570</td>
      <td>7.187365</td>
    </tr>
    <tr>
      <th>意大利</th>
      <td>1471329</td>
      <td>2500842</td>
      <td>77311</td>
      <td>5322.700000</td>
      <td>1985.599190</td>
      <td>3374.955466</td>
      <td>104.333333</td>
      <td>7.183131</td>
    </tr>
    <tr>
      <th>日本</th>
      <td>10011432</td>
      <td>17981631</td>
      <td>425563</td>
      <td>36006.000000</td>
      <td>1999.886536</td>
      <td>3592.015781</td>
      <td>85.010587</td>
      <td>7.192569</td>
    </tr>
    <tr>
      <th>比利时</th>
      <td>273932</td>
      <td>170449</td>
      <td>11380</td>
      <td>986.000000</td>
      <td>1999.503650</td>
      <td>1244.153285</td>
      <td>83.065693</td>
      <td>7.197080</td>
    </tr>
    <tr>
      <th>法国</th>
      <td>5551130</td>
      <td>10208966</td>
      <td>251524</td>
      <td>20186.500000</td>
      <td>1991.794044</td>
      <td>3663.066380</td>
      <td>90.249013</td>
      <td>7.243093</td>
    </tr>
    <tr>
      <th>波兰</th>
      <td>359652</td>
      <td>159577</td>
      <td>14613</td>
      <td>1347.000000</td>
      <td>1987.027624</td>
      <td>881.640884</td>
      <td>80.734807</td>
      <td>7.441989</td>
    </tr>
    <tr>
      <th>泰国</th>
      <td>590684</td>
      <td>1564881</td>
      <td>26002</td>
      <td>1796.100000</td>
      <td>2009.129252</td>
      <td>5322.724490</td>
      <td>88.442177</td>
      <td>6.109184</td>
    </tr>
    <tr>
      <th>澳大利亚</th>
      <td>590875</td>
      <td>1415443</td>
      <td>25250</td>
      <td>2051.300000</td>
      <td>2002.966102</td>
      <td>4798.111864</td>
      <td>85.593220</td>
      <td>6.953559</td>
    </tr>
    <tr>
      <th>瑞典</th>
      <td>371589</td>
      <td>289794</td>
      <td>17695</td>
      <td>1388.600000</td>
      <td>1987.106952</td>
      <td>1549.700535</td>
      <td>94.625668</td>
      <td>7.425668</td>
    </tr>
    <tr>
      <th>美国</th>
      <td>23363806</td>
      <td>101645832</td>
      <td>1053980</td>
      <td>81100.135704</td>
      <td>1994.519891</td>
      <td>8677.294861</td>
      <td>89.976097</td>
      <td>6.923351</td>
    </tr>
    <tr>
      <th>英国</th>
      <td>5307045</td>
      <td>13236409</td>
      <td>237129</td>
      <td>19930.800000</td>
      <td>1996.630926</td>
      <td>4979.837848</td>
      <td>89.213318</td>
      <td>7.498420</td>
    </tr>
    <tr>
      <th>荷兰</th>
      <td>302181</td>
      <td>144596</td>
      <td>11459</td>
      <td>1081.200000</td>
      <td>2001.198675</td>
      <td>957.589404</td>
      <td>75.887417</td>
      <td>7.160265</td>
    </tr>
    <tr>
      <th>西班牙</th>
      <td>886685</td>
      <td>1486383</td>
      <td>40271</td>
      <td>3112.100000</td>
      <td>2001.546275</td>
      <td>3355.266366</td>
      <td>90.905192</td>
      <td>7.025056</td>
    </tr>
    <tr>
      <th>阿根廷</th>
      <td>226476</td>
      <td>258085</td>
      <td>10458</td>
      <td>819.100000</td>
      <td>2004.212389</td>
      <td>2283.938053</td>
      <td>92.548673</td>
      <td>7.248673</td>
    </tr>
    <tr>
      <th>韩国</th>
      <td>2694871</td>
      <td>8759930</td>
      <td>134225</td>
      <td>8523.200000</td>
      <td>2008.100596</td>
      <td>6527.518629</td>
      <td>100.018629</td>
      <td>6.351118</td>
    </tr>
    <tr>
      <th>All</th>
      <td>76210353</td>
      <td>239235500</td>
      <td>3394146</td>
      <td>264162.435704</td>
      <td>1996.969656</td>
      <td>6268.781280</td>
      <td>88.938134</td>
      <td>6.921952</td>
    </tr>
  </tbody>
</table>
此处的All便是总和数据

### 3.9.7 对不同值执行不同函数：传递字典
**对不同值执行不同的函数：可以向aggfunc传递一个字典。不过，这样做有一个副作用，那就是必须将标签做的更加整洁才行。**

```python
pd.pivot_table(df, index = ["产地"], values = ["投票人数", "评分"], aggfunc = {"投票人数":np.sum, "评分":np.mean}, fill_value = 0)
```



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>投票人数</th>
      <th>评分</th>
    </tr>
    <tr>
      <th>产地</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>中国台湾</th>
      <td>5237466</td>
      <td>7.066667</td>
    </tr>
    <tr>
      <th>中国大陆</th>
      <td>41380993</td>
      <td>6.062991</td>
    </tr>
    <tr>
      <th>中国香港</th>
      <td>23179448</td>
      <td>6.473551</td>
    </tr>
    <tr>
      <th>丹麦</th>
      <td>394745</td>
      <td>7.246701</td>
    </tr>
    <tr>
      <th>俄罗斯</th>
      <td>486082</td>
      <td>7.557143</td>
    </tr>
    <tr>
      <th>其他</th>
      <td>3048849</td>
      <td>7.226713</td>
    </tr>
    <tr>
      <th>加拿大</th>
      <td>1362581</td>
      <td>6.727221</td>
    </tr>
    <tr>
      <th>印度</th>
      <td>1146173</td>
      <td>6.864888</td>
    </tr>
    <tr>
      <th>墨西哥</th>
      <td>139462</td>
      <td>7.085470</td>
    </tr>
    <tr>
      <th>巴西</th>
      <td>357027</td>
      <td>7.232323</td>
    </tr>
    <tr>
      <th>德国</th>
      <td>2679856</td>
      <td>7.187365</td>
    </tr>
    <tr>
      <th>意大利</th>
      <td>2500842</td>
      <td>7.183131</td>
    </tr>
    <tr>
      <th>日本</th>
      <td>17981631</td>
      <td>7.192569</td>
    </tr>
    <tr>
      <th>比利时</th>
      <td>170449</td>
      <td>7.197080</td>
    </tr>
    <tr>
      <th>法国</th>
      <td>10208966</td>
      <td>7.243093</td>
    </tr>
    <tr>
      <th>波兰</th>
      <td>159577</td>
      <td>7.441989</td>
    </tr>
    <tr>
      <th>泰国</th>
      <td>1564881</td>
      <td>6.109184</td>
    </tr>
    <tr>
      <th>澳大利亚</th>
      <td>1415443</td>
      <td>6.953559</td>
    </tr>
    <tr>
      <th>瑞典</th>
      <td>289794</td>
      <td>7.425668</td>
    </tr>
    <tr>
      <th>美国</th>
      <td>101645832</td>
      <td>6.923351</td>
    </tr>
    <tr>
      <th>英国</th>
      <td>13236409</td>
      <td>7.498420</td>
    </tr>
    <tr>
      <th>荷兰</th>
      <td>144596</td>
      <td>7.160265</td>
    </tr>
    <tr>
      <th>西班牙</th>
      <td>1486383</td>
      <td>7.025056</td>
    </tr>
    <tr>
      <th>阿根廷</th>
      <td>258085</td>
      <td>7.248673</td>
    </tr>
    <tr>
      <th>韩国</th>
      <td>8759930</td>
      <td>6.351118</td>
    </tr>
  </tbody>
</table>


对各个年份的投票人数求和，对评分求均值


```python
pd.pivot_table(df, index = ["年代"], values = ["投票人数", "评分"], aggfunc = {"投票人数":np.sum, "评分":np.mean}, fill_value = 0, margins = True)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>投票人数</th>
      <th>评分</th>
    </tr>
    <tr>
      <th>年代</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1888</th>
      <td>776</td>
      <td>7.950000</td>
    </tr>
    <tr>
      <th>1890</th>
      <td>51</td>
      <td>4.800000</td>
    </tr>
    <tr>
      <th>1892</th>
      <td>176</td>
      <td>7.500000</td>
    </tr>
    <tr>
      <th>1894</th>
      <td>338</td>
      <td>6.633333</td>
    </tr>
    <tr>
      <th>1895</th>
      <td>7679</td>
      <td>7.575000</td>
    </tr>
    <tr>
      <th>1896</th>
      <td>7874</td>
      <td>7.037500</td>
    </tr>
    <tr>
      <th>1897</th>
      <td>201</td>
      <td>6.633333</td>
    </tr>
    <tr>
      <th>1898</th>
      <td>1157</td>
      <td>7.450000</td>
    </tr>
    <tr>
      <th>1899</th>
      <td>142</td>
      <td>6.900000</td>
    </tr>
    <tr>
      <th>1900</th>
      <td>1227</td>
      <td>7.228571</td>
    </tr>
    <tr>
      <th>1901</th>
      <td>658</td>
      <td>7.250000</td>
    </tr>
    <tr>
      <th>1902</th>
      <td>11548</td>
      <td>7.680000</td>
    </tr>
    <tr>
      <th>1903</th>
      <td>4548</td>
      <td>7.015385</td>
    </tr>
    <tr>
      <th>1904</th>
      <td>1495</td>
      <td>7.616667</td>
    </tr>
    <tr>
      <th>1905</th>
      <td>1663</td>
      <td>6.820000</td>
    </tr>
    <tr>
      <th>1906</th>
      <td>1329</td>
      <td>7.342857</td>
    </tr>
    <tr>
      <th>1907</th>
      <td>1068</td>
      <td>7.020000</td>
    </tr>
    <tr>
      <th>1908</th>
      <td>1475</td>
      <td>7.425000</td>
    </tr>
    <tr>
      <th>1909</th>
      <td>248</td>
      <td>7.650000</td>
    </tr>
    <tr>
      <th>1910</th>
      <td>526</td>
      <td>6.940000</td>
    </tr>
    <tr>
      <th>1911</th>
      <td>924</td>
      <td>7.500000</td>
    </tr>
    <tr>
      <th>1912</th>
      <td>724</td>
      <td>7.925000</td>
    </tr>
    <tr>
      <th>1913</th>
      <td>436</td>
      <td>6.671429</td>
    </tr>
    <tr>
      <th>1914</th>
      <td>2728</td>
      <td>6.473077</td>
    </tr>
    <tr>
      <th>1915</th>
      <td>6298</td>
      <td>7.260000</td>
    </tr>
    <tr>
      <th>1916</th>
      <td>7333</td>
      <td>7.690909</td>
    </tr>
    <tr>
      <th>1917</th>
      <td>1493</td>
      <td>7.075000</td>
    </tr>
    <tr>
      <th>1918</th>
      <td>4285</td>
      <td>7.200000</td>
    </tr>
    <tr>
      <th>1919</th>
      <td>3506</td>
      <td>7.494444</td>
    </tr>
    <tr>
      <th>1920</th>
      <td>8911</td>
      <td>7.492857</td>
    </tr>
    <tr>
      <th>1921</th>
      <td>16056</td>
      <td>7.750000</td>
    </tr>
    <tr>
      <th>1922</th>
      <td>16876</td>
      <td>7.804545</td>
    </tr>
    <tr>
      <th>1923</th>
      <td>7611</td>
      <td>7.811765</td>
    </tr>
    <tr>
      <th>1924</th>
      <td>10382</td>
      <td>8.059259</td>
    </tr>
    <tr>
      <th>1925</th>
      <td>27607</td>
      <td>7.788000</td>
    </tr>
    <tr>
      <th>1926</th>
      <td>10203</td>
      <td>7.773913</td>
    </tr>
    <tr>
      <th>1927</th>
      <td>20163</td>
      <td>7.751724</td>
    </tr>
    <tr>
      <th>1928</th>
      <td>16133</td>
      <td>7.964103</td>
    </tr>
    <tr>
      <th>1929</th>
      <td>25919</td>
      <td>7.440000</td>
    </tr>
    <tr>
      <th>1930</th>
      <td>13877</td>
      <td>7.360000</td>
    </tr>
    <tr>
      <th>1931</th>
      <td>61684</td>
      <td>7.483333</td>
    </tr>
    <tr>
      <th>1932</th>
      <td>22203</td>
      <td>7.294595</td>
    </tr>
    <tr>
      <th>1933</th>
      <td>36289</td>
      <td>7.420833</td>
    </tr>
    <tr>
      <th>1934</th>
      <td>39573</td>
      <td>7.536000</td>
    </tr>
    <tr>
      <th>1935</th>
      <td>40834</td>
      <td>7.515217</td>
    </tr>
    <tr>
      <th>1936</th>
      <td>87864</td>
      <td>7.615254</td>
    </tr>
    <tr>
      <th>1937</th>
      <td>77381</td>
      <td>7.568750</td>
    </tr>
    <tr>
      <th>1938</th>
      <td>20976</td>
      <td>7.736842</td>
    </tr>
    <tr>
      <th>1939</th>
      <td>289681</td>
      <td>7.520408</td>
    </tr>
    <tr>
      <th>1940</th>
      <td>216213</td>
      <td>7.544737</td>
    </tr>
    <tr>
      <th>1941</th>
      <td>72982</td>
      <td>7.427660</td>
    </tr>
    <tr>
      <th>1942</th>
      <td>138211</td>
      <td>7.554717</td>
    </tr>
    <tr>
      <th>1943</th>
      <td>42300</td>
      <td>7.605357</td>
    </tr>
    <tr>
      <th>1944</th>
      <td>54398</td>
      <td>7.538889</td>
    </tr>
    <tr>
      <th>1945</th>
      <td>48462</td>
      <td>7.673469</td>
    </tr>
    <tr>
      <th>1946</th>
      <td>61033</td>
      <td>7.606780</td>
    </tr>
    <tr>
      <th>1947</th>
      <td>32834</td>
      <td>7.502703</td>
    </tr>
    <tr>
      <th>1948</th>
      <td>103136</td>
      <td>7.645349</td>
    </tr>
    <tr>
      <th>1949</th>
      <td>57110</td>
      <td>7.646067</td>
    </tr>
    <tr>
      <th>1950</th>
      <td>169862</td>
      <td>7.655263</td>
    </tr>
    <tr>
      <th>1951</th>
      <td>75495</td>
      <td>7.637179</td>
    </tr>
    <tr>
      <th>1952</th>
      <td>128121</td>
      <td>7.775294</td>
    </tr>
    <tr>
      <th>1953</th>
      <td>421134</td>
      <td>7.579310</td>
    </tr>
    <tr>
      <th>1954</th>
      <td>222337</td>
      <td>7.714851</td>
    </tr>
    <tr>
      <th>1955</th>
      <td>228056</td>
      <td>7.567544</td>
    </tr>
    <tr>
      <th>1956</th>
      <td>153943</td>
      <td>7.591724</td>
    </tr>
    <tr>
      <th>1957</th>
      <td>317989</td>
      <td>7.622115</td>
    </tr>
    <tr>
      <th>1958</th>
      <td>108116</td>
      <td>7.536885</td>
    </tr>
    <tr>
      <th>1959</th>
      <td>220809</td>
      <td>7.571875</td>
    </tr>
    <tr>
      <th>1960</th>
      <td>189424</td>
      <td>7.580769</td>
    </tr>
    <tr>
      <th>1961</th>
      <td>324975</td>
      <td>7.741000</td>
    </tr>
    <tr>
      <th>1962</th>
      <td>281964</td>
      <td>7.707042</td>
    </tr>
    <tr>
      <th>1963</th>
      <td>169316</td>
      <td>7.536364</td>
    </tr>
    <tr>
      <th>1964</th>
      <td>201454</td>
      <td>7.540782</td>
    </tr>
    <tr>
      <th>1965</th>
      <td>323972</td>
      <td>7.591304</td>
    </tr>
    <tr>
      <th>1966</th>
      <td>236142</td>
      <td>7.521910</td>
    </tr>
    <tr>
      <th>1967</th>
      <td>217681</td>
      <td>7.477143</td>
    </tr>
    <tr>
      <th>1968</th>
      <td>188333</td>
      <td>7.324706</td>
    </tr>
    <tr>
      <th>1969</th>
      <td>122661</td>
      <td>7.367647</td>
    </tr>
    <tr>
      <th>1970</th>
      <td>128492</td>
      <td>7.294149</td>
    </tr>
    <tr>
      <th>1971</th>
      <td>299585</td>
      <td>7.149309</td>
    </tr>
    <tr>
      <th>1972</th>
      <td>475755</td>
      <td>7.253431</td>
    </tr>
    <tr>
      <th>1973</th>
      <td>170270</td>
      <td>7.238095</td>
    </tr>
    <tr>
      <th>1974</th>
      <td>375884</td>
      <td>7.063964</td>
    </tr>
    <tr>
      <th>1975</th>
      <td>435544</td>
      <td>7.056995</td>
    </tr>
    <tr>
      <th>1976</th>
      <td>312364</td>
      <td>7.107725</td>
    </tr>
    <tr>
      <th>1977</th>
      <td>210467</td>
      <td>7.151376</td>
    </tr>
    <tr>
      <th>1978</th>
      <td>201366</td>
      <td>7.096517</td>
    </tr>
    <tr>
      <th>1979</th>
      <td>426270</td>
      <td>7.292857</td>
    </tr>
    <tr>
      <th>1980</th>
      <td>539000</td>
      <td>7.182063</td>
    </tr>
    <tr>
      <th>1981</th>
      <td>440794</td>
      <td>7.154044</td>
    </tr>
    <tr>
      <th>1982</th>
      <td>569800</td>
      <td>7.286260</td>
    </tr>
    <tr>
      <th>1983</th>
      <td>426771</td>
      <td>7.296727</td>
    </tr>
    <tr>
      <th>1984</th>
      <td>784106</td>
      <td>7.380887</td>
    </tr>
    <tr>
      <th>1985</th>
      <td>509528</td>
      <td>7.267722</td>
    </tr>
    <tr>
      <th>1986</th>
      <td>916279</td>
      <td>7.249536</td>
    </tr>
    <tr>
      <th>1987</th>
      <td>1073070</td>
      <td>7.282111</td>
    </tr>
    <tr>
      <th>1988</th>
      <td>1875710</td>
      <td>7.265544</td>
    </tr>
    <tr>
      <th>1989</th>
      <td>1176669</td>
      <td>7.199233</td>
    </tr>
    <tr>
      <th>1990</th>
      <td>1951292</td>
      <td>7.156923</td>
    </tr>
    <tr>
      <th>1991</th>
      <td>1921033</td>
      <td>7.154412</td>
    </tr>
    <tr>
      <th>1992</th>
      <td>2385473</td>
      <td>7.190187</td>
    </tr>
    <tr>
      <th>1993</th>
      <td>3579850</td>
      <td>7.186560</td>
    </tr>
    <tr>
      <th>1994</th>
      <td>5668165</td>
      <td>7.257260</td>
    </tr>
    <tr>
      <th>1995</th>
      <td>3976373</td>
      <td>7.275052</td>
    </tr>
    <tr>
      <th>1996</th>
      <td>2634034</td>
      <td>7.249119</td>
    </tr>
    <tr>
      <th>1997</th>
      <td>4426584</td>
      <td>7.325758</td>
    </tr>
    <tr>
      <th>1998</th>
      <td>4286770</td>
      <td>7.223540</td>
    </tr>
    <tr>
      <th>1999</th>
      <td>4459724</td>
      <td>7.171151</td>
    </tr>
    <tr>
      <th>2000</th>
      <td>4317784</td>
      <td>7.112413</td>
    </tr>
    <tr>
      <th>2001</th>
      <td>7027000</td>
      <td>7.070647</td>
    </tr>
    <tr>
      <th>2002</th>
      <td>5894532</td>
      <td>7.045620</td>
    </tr>
    <tr>
      <th>2003</th>
      <td>7480686</td>
      <td>7.114410</td>
    </tr>
    <tr>
      <th>2004</th>
      <td>9590716</td>
      <td>6.999909</td>
    </tr>
    <tr>
      <th>2005</th>
      <td>8209986</td>
      <td>7.011864</td>
    </tr>
    <tr>
      <th>2006</th>
      <td>11022718</td>
      <td>6.901546</td>
    </tr>
    <tr>
      <th>2007</th>
      <td>10693163</td>
      <td>6.859703</td>
    </tr>
    <tr>
      <th>2008</th>
      <td>13250500</td>
      <td>6.892164</td>
    </tr>
    <tr>
      <th>2009</th>
      <td>15972119</td>
      <td>6.732825</td>
    </tr>
    <tr>
      <th>2010</th>
      <td>18044725</td>
      <td>6.752793</td>
    </tr>
    <tr>
      <th>2011</th>
      <td>16347306</td>
      <td>6.560325</td>
    </tr>
    <tr>
      <th>2012</th>
      <td>14460484</td>
      <td>6.444896</td>
    </tr>
    <tr>
      <th>2013</th>
      <td>15211249</td>
      <td>6.375974</td>
    </tr>
    <tr>
      <th>2014</th>
      <td>14570038</td>
      <td>6.249384</td>
    </tr>
    <tr>
      <th>2015</th>
      <td>12481394</td>
      <td>6.121925</td>
    </tr>
    <tr>
      <th>2016</th>
      <td>1850030</td>
      <td>5.834524</td>
    </tr>
    <tr>
      <th>2018</th>
      <td>123456</td>
      <td>6.935704</td>
    </tr>
    <tr>
      <th>All</th>
      <td>239235500</td>
      <td>6.921952</td>
    </tr>
  </tbody>
</table>


### 3.9.8 透视表过滤

```python
table = pd.pivot_table(df, index = ["年代"], values = ["投票人数", "评分"], aggfunc = {"投票人数":np.sum, "评分":np.mean}, fill_value = 0)
```


```python
type(table)
```
pandas.core.frame.DataFrame


```python
table[:5]
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>投票人数</th>
      <th>评分</th>
    </tr>
    <tr>
      <th>年代</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1888</th>
      <td>776</td>
      <td>7.950000</td>
    </tr>
    <tr>
      <th>1890</th>
      <td>51</td>
      <td>4.800000</td>
    </tr>
    <tr>
      <th>1892</th>
      <td>176</td>
      <td>7.500000</td>
    </tr>
    <tr>
      <th>1894</th>
      <td>338</td>
      <td>6.633333</td>
    </tr>
    <tr>
      <th>1895</th>
      <td>7679</td>
      <td>7.575000</td>
    </tr>
  </tbody>
</table>


**1994年被誉为电影史上最伟大的一年，但是通过数据我们可以发现，1994年的平均得分其实并不是很高。1924年的电影均分最高。**


```python
table[table.index == 1994]
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>投票人数</th>
      <th>评分</th>
    </tr>
    <tr>
      <th>年代</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1994</th>
      <td>5668165</td>
      <td>7.25726</td>
    </tr>
  </tbody>
</table>

```python
table.sort_values("评分", ascending = False)
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>投票人数</th>
      <th>评分</th>
    </tr>
    <tr>
      <th>年代</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1924</th>
      <td>10382</td>
      <td>8.059259</td>
    </tr>
    <tr>
      <th>1928</th>
      <td>16133</td>
      <td>7.964103</td>
    </tr>
    <tr>
      <th>1888</th>
      <td>776</td>
      <td>7.950000</td>
    </tr>
    <tr>
      <th>1912</th>
      <td>724</td>
      <td>7.925000</td>
    </tr>
    <tr>
      <th>1923</th>
      <td>7611</td>
      <td>7.811765</td>
    </tr>
    <tr>
      <th>1922</th>
      <td>16876</td>
      <td>7.804545</td>
    </tr>
    <tr>
      <th>1925</th>
      <td>27607</td>
      <td>7.788000</td>
    </tr>
    <tr>
      <th>1952</th>
      <td>128121</td>
      <td>7.775294</td>
    </tr>
    <tr>
      <th>1926</th>
      <td>10203</td>
      <td>7.773913</td>
    </tr>
    <tr>
      <th>1927</th>
      <td>20163</td>
      <td>7.751724</td>
    </tr>
    <tr>
      <th>1921</th>
      <td>16056</td>
      <td>7.750000</td>
    </tr>
    <tr>
      <th>1961</th>
      <td>324975</td>
      <td>7.741000</td>
    </tr>
    <tr>
      <th>1938</th>
      <td>20976</td>
      <td>7.736842</td>
    </tr>
    <tr>
      <th>1954</th>
      <td>222337</td>
      <td>7.714851</td>
    </tr>
    <tr>
      <th>1962</th>
      <td>281964</td>
      <td>7.707042</td>
    </tr>
    <tr>
      <th>1916</th>
      <td>7333</td>
      <td>7.690909</td>
    </tr>
    <tr>
      <th>1902</th>
      <td>11548</td>
      <td>7.680000</td>
    </tr>
    <tr>
      <th>1945</th>
      <td>48462</td>
      <td>7.673469</td>
    </tr>
    <tr>
      <th>1950</th>
      <td>169862</td>
      <td>7.655263</td>
    </tr>
    <tr>
      <th>1909</th>
      <td>248</td>
      <td>7.650000</td>
    </tr>
    <tr>
      <th>1949</th>
      <td>57110</td>
      <td>7.646067</td>
    </tr>
    <tr>
      <th>1948</th>
      <td>103136</td>
      <td>7.645349</td>
    </tr>
    <tr>
      <th>1951</th>
      <td>75495</td>
      <td>7.637179</td>
    </tr>
    <tr>
      <th>1957</th>
      <td>317989</td>
      <td>7.622115</td>
    </tr>
    <tr>
      <th>1904</th>
      <td>1495</td>
      <td>7.616667</td>
    </tr>
    <tr>
      <th>1936</th>
      <td>87864</td>
      <td>7.615254</td>
    </tr>
    <tr>
      <th>1946</th>
      <td>61033</td>
      <td>7.606780</td>
    </tr>
    <tr>
      <th>1943</th>
      <td>42300</td>
      <td>7.605357</td>
    </tr>
    <tr>
      <th>1956</th>
      <td>153943</td>
      <td>7.591724</td>
    </tr>
    <tr>
      <th>1965</th>
      <td>323972</td>
      <td>7.591304</td>
    </tr>
    <tr>
      <th>1960</th>
      <td>189424</td>
      <td>7.580769</td>
    </tr>
    <tr>
      <th>1953</th>
      <td>421134</td>
      <td>7.579310</td>
    </tr>
    <tr>
      <th>1895</th>
      <td>7679</td>
      <td>7.575000</td>
    </tr>
    <tr>
      <th>1959</th>
      <td>220809</td>
      <td>7.571875</td>
    </tr>
    <tr>
      <th>1937</th>
      <td>77381</td>
      <td>7.568750</td>
    </tr>
    <tr>
      <th>1955</th>
      <td>228056</td>
      <td>7.567544</td>
    </tr>
    <tr>
      <th>1942</th>
      <td>138211</td>
      <td>7.554717</td>
    </tr>
    <tr>
      <th>1940</th>
      <td>216213</td>
      <td>7.544737</td>
    </tr>
    <tr>
      <th>1964</th>
      <td>201454</td>
      <td>7.540782</td>
    </tr>
    <tr>
      <th>1944</th>
      <td>54398</td>
      <td>7.538889</td>
    </tr>
    <tr>
      <th>1958</th>
      <td>108116</td>
      <td>7.536885</td>
    </tr>
    <tr>
      <th>1963</th>
      <td>169316</td>
      <td>7.536364</td>
    </tr>
    <tr>
      <th>1934</th>
      <td>39573</td>
      <td>7.536000</td>
    </tr>
    <tr>
      <th>1966</th>
      <td>236142</td>
      <td>7.521910</td>
    </tr>
    <tr>
      <th>1939</th>
      <td>289681</td>
      <td>7.520408</td>
    </tr>
    <tr>
      <th>1935</th>
      <td>40834</td>
      <td>7.515217</td>
    </tr>
    <tr>
      <th>1947</th>
      <td>32834</td>
      <td>7.502703</td>
    </tr>
    <tr>
      <th>1892</th>
      <td>176</td>
      <td>7.500000</td>
    </tr>
    <tr>
      <th>1911</th>
      <td>924</td>
      <td>7.500000</td>
    </tr>
    <tr>
      <th>1919</th>
      <td>3506</td>
      <td>7.494444</td>
    </tr>
    <tr>
      <th>1920</th>
      <td>8911</td>
      <td>7.492857</td>
    </tr>
    <tr>
      <th>1931</th>
      <td>61684</td>
      <td>7.483333</td>
    </tr>
    <tr>
      <th>1967</th>
      <td>217681</td>
      <td>7.477143</td>
    </tr>
    <tr>
      <th>1898</th>
      <td>1157</td>
      <td>7.450000</td>
    </tr>
    <tr>
      <th>1929</th>
      <td>25919</td>
      <td>7.440000</td>
    </tr>
    <tr>
      <th>1941</th>
      <td>72982</td>
      <td>7.427660</td>
    </tr>
    <tr>
      <th>1908</th>
      <td>1475</td>
      <td>7.425000</td>
    </tr>
    <tr>
      <th>1933</th>
      <td>36289</td>
      <td>7.420833</td>
    </tr>
    <tr>
      <th>1984</th>
      <td>784106</td>
      <td>7.380887</td>
    </tr>
    <tr>
      <th>1969</th>
      <td>122661</td>
      <td>7.367647</td>
    </tr>
    <tr>
      <th>1930</th>
      <td>13877</td>
      <td>7.360000</td>
    </tr>
    <tr>
      <th>1906</th>
      <td>1329</td>
      <td>7.342857</td>
    </tr>
    <tr>
      <th>1997</th>
      <td>4426584</td>
      <td>7.325758</td>
    </tr>
    <tr>
      <th>1968</th>
      <td>188333</td>
      <td>7.324706</td>
    </tr>
    <tr>
      <th>1983</th>
      <td>426771</td>
      <td>7.296727</td>
    </tr>
    <tr>
      <th>1932</th>
      <td>22203</td>
      <td>7.294595</td>
    </tr>
    <tr>
      <th>1970</th>
      <td>128492</td>
      <td>7.294149</td>
    </tr>
    <tr>
      <th>1979</th>
      <td>426270</td>
      <td>7.292857</td>
    </tr>
    <tr>
      <th>1982</th>
      <td>569800</td>
      <td>7.286260</td>
    </tr>
    <tr>
      <th>1987</th>
      <td>1073070</td>
      <td>7.282111</td>
    </tr>
    <tr>
      <th>1995</th>
      <td>3976373</td>
      <td>7.275052</td>
    </tr>
    <tr>
      <th>1985</th>
      <td>509528</td>
      <td>7.267722</td>
    </tr>
    <tr>
      <th>1988</th>
      <td>1875710</td>
      <td>7.265544</td>
    </tr>
    <tr>
      <th>1915</th>
      <td>6298</td>
      <td>7.260000</td>
    </tr>
    <tr>
      <th>1994</th>
      <td>5668165</td>
      <td>7.257260</td>
    </tr>
    <tr>
      <th>1972</th>
      <td>475755</td>
      <td>7.253431</td>
    </tr>
    <tr>
      <th>1901</th>
      <td>658</td>
      <td>7.250000</td>
    </tr>
    <tr>
      <th>1986</th>
      <td>916279</td>
      <td>7.249536</td>
    </tr>
    <tr>
      <th>1996</th>
      <td>2634034</td>
      <td>7.249119</td>
    </tr>
    <tr>
      <th>1973</th>
      <td>170270</td>
      <td>7.238095</td>
    </tr>
    <tr>
      <th>1900</th>
      <td>1227</td>
      <td>7.228571</td>
    </tr>
    <tr>
      <th>1998</th>
      <td>4286770</td>
      <td>7.223540</td>
    </tr>
    <tr>
      <th>1918</th>
      <td>4285</td>
      <td>7.200000</td>
    </tr>
    <tr>
      <th>1989</th>
      <td>1176669</td>
      <td>7.199233</td>
    </tr>
    <tr>
      <th>1992</th>
      <td>2385473</td>
      <td>7.190187</td>
    </tr>
    <tr>
      <th>1993</th>
      <td>3579850</td>
      <td>7.186560</td>
    </tr>
    <tr>
      <th>1980</th>
      <td>539000</td>
      <td>7.182063</td>
    </tr>
    <tr>
      <th>1999</th>
      <td>4459724</td>
      <td>7.171151</td>
    </tr>
    <tr>
      <th>1990</th>
      <td>1951292</td>
      <td>7.156923</td>
    </tr>
    <tr>
      <th>1991</th>
      <td>1921033</td>
      <td>7.154412</td>
    </tr>
    <tr>
      <th>1981</th>
      <td>440794</td>
      <td>7.154044</td>
    </tr>
    <tr>
      <th>1977</th>
      <td>210467</td>
      <td>7.151376</td>
    </tr>
    <tr>
      <th>1971</th>
      <td>299585</td>
      <td>7.149309</td>
    </tr>
    <tr>
      <th>2003</th>
      <td>7480686</td>
      <td>7.114410</td>
    </tr>
    <tr>
      <th>2000</th>
      <td>4317784</td>
      <td>7.112413</td>
    </tr>
    <tr>
      <th>1976</th>
      <td>312364</td>
      <td>7.107725</td>
    </tr>
    <tr>
      <th>1978</th>
      <td>201366</td>
      <td>7.096517</td>
    </tr>
    <tr>
      <th>1917</th>
      <td>1493</td>
      <td>7.075000</td>
    </tr>
    <tr>
      <th>2001</th>
      <td>7027000</td>
      <td>7.070647</td>
    </tr>
    <tr>
      <th>1974</th>
      <td>375884</td>
      <td>7.063964</td>
    </tr>
    <tr>
      <th>1975</th>
      <td>435544</td>
      <td>7.056995</td>
    </tr>
    <tr>
      <th>2002</th>
      <td>5894532</td>
      <td>7.045620</td>
    </tr>
    <tr>
      <th>1896</th>
      <td>7874</td>
      <td>7.037500</td>
    </tr>
    <tr>
      <th>1907</th>
      <td>1068</td>
      <td>7.020000</td>
    </tr>
    <tr>
      <th>1903</th>
      <td>4548</td>
      <td>7.015385</td>
    </tr>
    <tr>
      <th>2005</th>
      <td>8209986</td>
      <td>7.011864</td>
    </tr>
    <tr>
      <th>2004</th>
      <td>9590716</td>
      <td>6.999909</td>
    </tr>
    <tr>
      <th>1910</th>
      <td>526</td>
      <td>6.940000</td>
    </tr>
    <tr>
      <th>2018</th>
      <td>123456</td>
      <td>6.935704</td>
    </tr>
    <tr>
      <th>2006</th>
      <td>11022718</td>
      <td>6.901546</td>
    </tr>
    <tr>
      <th>1899</th>
      <td>142</td>
      <td>6.900000</td>
    </tr>
    <tr>
      <th>2008</th>
      <td>13250500</td>
      <td>6.892164</td>
    </tr>
    <tr>
      <th>2007</th>
      <td>10693163</td>
      <td>6.859703</td>
    </tr>
    <tr>
      <th>1905</th>
      <td>1663</td>
      <td>6.820000</td>
    </tr>
    <tr>
      <th>2010</th>
      <td>18044725</td>
      <td>6.752793</td>
    </tr>
    <tr>
      <th>2009</th>
      <td>15972119</td>
      <td>6.732825</td>
    </tr>
    <tr>
      <th>1913</th>
      <td>436</td>
      <td>6.671429</td>
    </tr>
    <tr>
      <th>1894</th>
      <td>338</td>
      <td>6.633333</td>
    </tr>
    <tr>
      <th>1897</th>
      <td>201</td>
      <td>6.633333</td>
    </tr>
    <tr>
      <th>2011</th>
      <td>16347306</td>
      <td>6.560325</td>
    </tr>
    <tr>
      <th>1914</th>
      <td>2728</td>
      <td>6.473077</td>
    </tr>
    <tr>
      <th>2012</th>
      <td>14460484</td>
      <td>6.444896</td>
    </tr>
    <tr>
      <th>2013</th>
      <td>15211249</td>
      <td>6.375974</td>
    </tr>
    <tr>
      <th>2014</th>
      <td>14570038</td>
      <td>6.249384</td>
    </tr>
    <tr>
      <th>2015</th>
      <td>12481394</td>
      <td>6.121925</td>
    </tr>
    <tr>
      <th>2016</th>
      <td>1850030</td>
      <td>5.834524</td>
    </tr>
    <tr>
      <th>1890</th>
      <td>51</td>
      <td>4.800000</td>
    </tr>
  </tbody>
</table>


**同样的，我们也可以按照多个索引来进行汇总。**


```python
pd.pivot_table(df, index = ["产地", "年代"], values = ["投票人数", "评分"], aggfunc = {"投票人数":np.sum, "评分":np.mean}, fill_value = 0)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>投票人数</th>
      <th>评分</th>
    </tr>
    <tr>
      <th>产地</th>
      <th>年代</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">中国台湾</th>
      <th>1963</th>
      <td>121</td>
      <td>6.400000</td>
    </tr>
    <tr>
      <th>1965</th>
      <td>461</td>
      <td>6.800000</td>
    </tr>
    <tr>
      <th>1966</th>
      <td>51</td>
      <td>7.900000</td>
    </tr>
    <tr>
      <th>1967</th>
      <td>4444</td>
      <td>8.000000</td>
    </tr>
    <tr>
      <th>1968</th>
      <td>178</td>
      <td>7.400000</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">韩国</th>
      <th>2012</th>
      <td>610317</td>
      <td>6.035238</td>
    </tr>
    <tr>
      <th>2013</th>
      <td>1130800</td>
      <td>6.062037</td>
    </tr>
    <tr>
      <th>2014</th>
      <td>453152</td>
      <td>5.650833</td>
    </tr>
    <tr>
      <th>2015</th>
      <td>349808</td>
      <td>5.423853</td>
    </tr>
    <tr>
      <th>2016</th>
      <td>34797</td>
      <td>5.730000</td>
    </tr>
  </tbody>
</table>
<p>1578 rows × 2 columns</p>




## 3.10 数据重塑和轴向旋转

### 3.10.1 层次化索引

层次化索引是pandas的一项重要功能，它能使我们在一个轴上拥有多个索引。

#### 3.10.1.1 Series的层次化索引：

```python
s = pd.Series(np.arange(1,10), index = [['a','a','a','b','b','c','c','d','d'], [1,2,3,1,2,3,1,2,3]])
s #类似于合并单元格
```


    a  1    1
       2    2
       3    3
    b  1    4
       2    5
    c  3    6
       1    7
    d  2    8
       3    9
    dtype: int32


```python
s.index
```


    MultiIndex([('a', 1),
                ('a', 2),
                ('a', 3),
                ('b', 1),
                ('b', 2),
                ('c', 3),
                ('c', 1),
                ('d', 2),
                ('d', 3)],
               )




```python
s['a'] #外层索引
```


    1    1
    2    2
    3    3
    dtype: int32




```python
s['a':'c'] #切片
```


    a  1    1
       2    2
       3    3
    b  1    4
       2    5
    c  3    6
       1    7
    dtype: int32




```python
s[:,1] #内层索引
```


    a    1
    b    4
    c    7
    dtype: int32




```python
s['c',3] #提取具体的值
```


    6

#### 3.10.1.2 通过unstack方法将Series变成DataFrame

```python
s.unstack()
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>1.0</td>
      <td>2.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>b</th>
      <td>4.0</td>
      <td>5.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>c</th>
      <td>7.0</td>
      <td>NaN</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>d</th>
      <td>NaN</td>
      <td>8.0</td>
      <td>9.0</td>
    </tr>
  </tbody>
</table>



```python
s.unstack().stack() #形式上的相互转换 具体的应用？还是说整花活
```


    a  1    1.0
       2    2.0
       3    3.0
    b  1    4.0
       2    5.0
    c  1    7.0
       3    6.0
    d  2    8.0
       3    9.0
    dtype: float64



#### 3.10.1.3 Dataframe的层次化索引： 

对于DataFrame来说，行和列都能进行层次化索引。


```python
data = pd.DataFrame(np.arange(12).reshape(4,3), index = [['a','a','b','b'],[1,2,1,2]], columns = [['A','A','B'],['Z','X','C']])
data
```

<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th></th>
      <th colspan="2" halign="left">A</th>
      <th>B</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th>Z</th>
      <th>X</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">a</th>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>4</td>
      <td>5</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">b</th>
      <th>1</th>
      <td>6</td>
      <td>7</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
  </tbody>
</table>



```python
data['A']
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>Z</th>
      <th>X</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">a</th>
      <th>1</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">b</th>
      <th>1</th>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9</td>
      <td>10</td>
    </tr>
  </tbody>
</table>



```python
data.index.names = ["row1","row2"]
data.columns.names = ["col1", "col2"]
data
```

<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>col1</th>
      <th colspan="2" halign="left">A</th>
      <th>B</th>
    </tr>
    <tr>
      <th></th>
      <th>col2</th>
      <th>Z</th>
      <th>X</th>
      <th>C</th>
    </tr>
    <tr>
      <th>row1</th>
      <th>row2</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">a</th>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>4</td>
      <td>5</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">b</th>
      <th>1</th>
      <td>6</td>
      <td>7</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
这样位置就不对了，需要调整




```python
data.swaplevel("row1","row2") #位置调整，交换
```

<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>col1</th>
      <th colspan="2" halign="left">A</th>
      <th>B</th>
    </tr>
    <tr>
      <th></th>
      <th>col2</th>
      <th>Z</th>
      <th>X</th>
      <th>C</th>
    </tr>
    <tr>
      <th>row2</th>
      <th>row1</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <th>a</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <th>a</th>
      <td>3</td>
      <td>4</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <th>b</th>
      <td>6</td>
      <td>7</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <th>b</th>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
  </tbody>
</table>




#### 3.10.1.4 将电影数据处理成多层索引的结构

```python
df.index #默认索引
```


    Int64Index([    0,     1,     2,     3,     4,     5,     6,     7,     8,
                    9,
                ...
                38153, 38154, 38155, 38156, 38157, 38158, 38159, 38160, 38161,
                38162],
               dtype='int64', length=38163)

**把产地和年代同时设成索引，产地是外层索引，年代为内层索引。**

**set_index可以把列变成索引**

**reset_index是把索引变成列** 

```python
df = df.set_index(["产地", "年代"])
df
```



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>名字</th>
      <th>投票人数</th>
      <th>类型</th>
      <th>上映时间</th>
      <th>时长</th>
      <th>评分</th>
      <th>首映地点</th>
    </tr>
    <tr>
      <th>产地</th>
      <th>年代</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">美国</th>
      <th>1994</th>
      <td>肖申克的救赎</td>
      <td>692795</td>
      <td>剧情/犯罪</td>
      <td>1994-09-10 00:00:00</td>
      <td>142</td>
      <td>9.600000</td>
      <td>多伦多电影节</td>
    </tr>
    <tr>
      <th>1957</th>
      <td>控方证人</td>
      <td>42995</td>
      <td>剧情/悬疑/犯罪</td>
      <td>1957-12-17 00:00:00</td>
      <td>116</td>
      <td>9.500000</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>意大利</th>
      <th>1997</th>
      <td>美丽人生</td>
      <td>327855</td>
      <td>剧情/喜剧/爱情</td>
      <td>1997-12-20 00:00:00</td>
      <td>116</td>
      <td>9.500000</td>
      <td>意大利</td>
    </tr>
    <tr>
      <th>美国</th>
      <th>1994</th>
      <td>阿甘正传</td>
      <td>580897</td>
      <td>剧情/爱情</td>
      <td>1994-06-23 00:00:00</td>
      <td>142</td>
      <td>9.400000</td>
      <td>洛杉矶首映</td>
    </tr>
    <tr>
      <th>中国大陆</th>
      <th>1993</th>
      <td>霸王别姬</td>
      <td>478523</td>
      <td>剧情/爱情/同性</td>
      <td>1993-01-01 00:00:00</td>
      <td>171</td>
      <td>9.400000</td>
      <td>香港</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>美国</th>
      <th>1935</th>
      <td>1935年</td>
      <td>57</td>
      <td>喜剧/歌舞</td>
      <td>1935-03-15 00:00:00</td>
      <td>98</td>
      <td>7.600000</td>
      <td>美国</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">中国大陆</th>
      <th>1986</th>
      <td>血溅画屏</td>
      <td>95</td>
      <td>剧情/悬疑/犯罪/武侠/古装</td>
      <td>1905-06-08 00:00:00</td>
      <td>91</td>
      <td>7.100000</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>1986</th>
      <td>魔窟中的幻想</td>
      <td>51</td>
      <td>惊悚/恐怖/儿童</td>
      <td>1905-06-08 00:00:00</td>
      <td>78</td>
      <td>8.000000</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>俄罗斯</th>
      <th>1977</th>
      <td>列宁格勒围困之星火战役 Блокада: Фильм 2: Ленинградский ме...</td>
      <td>32</td>
      <td>剧情/战争</td>
      <td>1905-05-30 00:00:00</td>
      <td>97</td>
      <td>6.600000</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>美国</th>
      <th>2018</th>
      <td>复仇者联盟3</td>
      <td>123456</td>
      <td>剧情/科幻</td>
      <td>2018-05-04 00:00:00</td>
      <td>142</td>
      <td>6.935704</td>
      <td>美国</td>
    </tr>
  </tbody>
</table>
<p>38163 rows × 7 columns</p>


**每一个索引都是一个元组：**

```python
df.index[0]
```


    ('美国', 1994)



**获取所有的美国电影，由于产地信息已经变成了索引，因此要是用.loc方法。**

```python
df.loc["美国"] #行标签索引行数据，注意索引多行时两边都是闭区间
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>名字</th>
      <th>投票人数</th>
      <th>类型</th>
      <th>上映时间</th>
      <th>时长</th>
      <th>评分</th>
      <th>首映地点</th>
    </tr>
    <tr>
      <th>年代</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1994</th>
      <td>肖申克的救赎</td>
      <td>692795</td>
      <td>剧情/犯罪</td>
      <td>1994-09-10 00:00:00</td>
      <td>142</td>
      <td>9.600000</td>
      <td>多伦多电影节</td>
    </tr>
    <tr>
      <th>1957</th>
      <td>控方证人</td>
      <td>42995</td>
      <td>剧情/悬疑/犯罪</td>
      <td>1957-12-17 00:00:00</td>
      <td>116</td>
      <td>9.500000</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>1994</th>
      <td>阿甘正传</td>
      <td>580897</td>
      <td>剧情/爱情</td>
      <td>1994-06-23 00:00:00</td>
      <td>142</td>
      <td>9.400000</td>
      <td>洛杉矶首映</td>
    </tr>
    <tr>
      <th>2012</th>
      <td>泰坦尼克号</td>
      <td>157074</td>
      <td>剧情/爱情/灾难</td>
      <td>2012-04-10 00:00:00</td>
      <td>194</td>
      <td>9.400000</td>
      <td>中国大陆</td>
    </tr>
    <tr>
      <th>1993</th>
      <td>辛德勒的名单</td>
      <td>306904</td>
      <td>剧情/历史/战争</td>
      <td>1993-11-30 00:00:00</td>
      <td>195</td>
      <td>9.400000</td>
      <td>华盛顿首映</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1987</th>
      <td>零下的激情</td>
      <td>199</td>
      <td>剧情/爱情/犯罪</td>
      <td>1987-11-06 00:00:00</td>
      <td>98</td>
      <td>7.400000</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>1986</th>
      <td>离别秋波</td>
      <td>240</td>
      <td>剧情/爱情/音乐</td>
      <td>1986-02-19 00:00:00</td>
      <td>90</td>
      <td>8.200000</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>1986</th>
      <td>极乐森林</td>
      <td>45</td>
      <td>纪录片</td>
      <td>1986-09-14 00:00:00</td>
      <td>90</td>
      <td>8.100000</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>1935</th>
      <td>1935年</td>
      <td>57</td>
      <td>喜剧/歌舞</td>
      <td>1935-03-15 00:00:00</td>
      <td>98</td>
      <td>7.600000</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>2018</th>
      <td>复仇者联盟3</td>
      <td>123456</td>
      <td>剧情/科幻</td>
      <td>2018-05-04 00:00:00</td>
      <td>142</td>
      <td>6.935704</td>
      <td>美国</td>
    </tr>
  </tbody>
</table>
<p>11714 rows × 7 columns</p>


```python
df.loc["中国大陆"]
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>名字</th>
      <th>投票人数</th>
      <th>类型</th>
      <th>上映时间</th>
      <th>时长</th>
      <th>评分</th>
      <th>首映地点</th>
    </tr>
    <tr>
      <th>年代</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1993</th>
      <td>霸王别姬</td>
      <td>478523</td>
      <td>剧情/爱情/同性</td>
      <td>1993-01-01 00:00:00</td>
      <td>171</td>
      <td>9.4</td>
      <td>香港</td>
    </tr>
    <tr>
      <th>1961</th>
      <td>大闹天宫</td>
      <td>74881</td>
      <td>动画/奇幻</td>
      <td>1905-05-14 00:00:00</td>
      <td>114</td>
      <td>9.2</td>
      <td>上集</td>
    </tr>
    <tr>
      <th>2015</th>
      <td>穹顶之下</td>
      <td>51113</td>
      <td>纪录片</td>
      <td>2015-02-28 00:00:00</td>
      <td>104</td>
      <td>9.2</td>
      <td>中国大陆</td>
    </tr>
    <tr>
      <th>1982</th>
      <td>茶馆</td>
      <td>10678</td>
      <td>剧情/历史</td>
      <td>1905-06-04 00:00:00</td>
      <td>118</td>
      <td>9.2</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>1988</th>
      <td>山水情</td>
      <td>10781</td>
      <td>动画/短片</td>
      <td>1905-06-10 00:00:00</td>
      <td>19</td>
      <td>9.2</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1986</th>
      <td>T省的八四、八五</td>
      <td>380</td>
      <td>剧情</td>
      <td>1905-06-08 00:00:00</td>
      <td>94</td>
      <td>8.7</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>1986</th>
      <td>失踪的女中学生</td>
      <td>101</td>
      <td>儿童</td>
      <td>1905-06-08 00:00:00</td>
      <td>102</td>
      <td>7.4</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>1986</th>
      <td>血战台儿庄</td>
      <td>2908</td>
      <td>战争</td>
      <td>1905-06-08 00:00:00</td>
      <td>120</td>
      <td>8.1</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>1986</th>
      <td>血溅画屏</td>
      <td>95</td>
      <td>剧情/悬疑/犯罪/武侠/古装</td>
      <td>1905-06-08 00:00:00</td>
      <td>91</td>
      <td>7.1</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>1986</th>
      <td>魔窟中的幻想</td>
      <td>51</td>
      <td>惊悚/恐怖/儿童</td>
      <td>1905-06-08 00:00:00</td>
      <td>78</td>
      <td>8.0</td>
      <td>美国</td>
    </tr>
  </tbody>
</table>
<p>3791 rows × 7 columns</p>


**这样做的最大好处是我们可以简化很多的筛选环节**

**每一个索引是一个元组：** 

```python
df = df.swaplevel("产地", "年代") #调换标签顺序
df
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>名字</th>
      <th>投票人数</th>
      <th>类型</th>
      <th>上映时间</th>
      <th>时长</th>
      <th>评分</th>
      <th>首映地点</th>
    </tr>
    <tr>
      <th>年代</th>
      <th>产地</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1994</th>
      <th>美国</th>
      <td>肖申克的救赎</td>
      <td>692795</td>
      <td>剧情/犯罪</td>
      <td>1994-09-10 00:00:00</td>
      <td>142</td>
      <td>9.600000</td>
      <td>多伦多电影节</td>
    </tr>
    <tr>
      <th>1957</th>
      <th>美国</th>
      <td>控方证人</td>
      <td>42995</td>
      <td>剧情/悬疑/犯罪</td>
      <td>1957-12-17 00:00:00</td>
      <td>116</td>
      <td>9.500000</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>1997</th>
      <th>意大利</th>
      <td>美丽人生</td>
      <td>327855</td>
      <td>剧情/喜剧/爱情</td>
      <td>1997-12-20 00:00:00</td>
      <td>116</td>
      <td>9.500000</td>
      <td>意大利</td>
    </tr>
    <tr>
      <th>1994</th>
      <th>美国</th>
      <td>阿甘正传</td>
      <td>580897</td>
      <td>剧情/爱情</td>
      <td>1994-06-23 00:00:00</td>
      <td>142</td>
      <td>9.400000</td>
      <td>洛杉矶首映</td>
    </tr>
    <tr>
      <th>1993</th>
      <th>中国大陆</th>
      <td>霸王别姬</td>
      <td>478523</td>
      <td>剧情/爱情/同性</td>
      <td>1993-01-01 00:00:00</td>
      <td>171</td>
      <td>9.400000</td>
      <td>香港</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1935</th>
      <th>美国</th>
      <td>1935年</td>
      <td>57</td>
      <td>喜剧/歌舞</td>
      <td>1935-03-15 00:00:00</td>
      <td>98</td>
      <td>7.600000</td>
      <td>美国</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">1986</th>
      <th>中国大陆</th>
      <td>血溅画屏</td>
      <td>95</td>
      <td>剧情/悬疑/犯罪/武侠/古装</td>
      <td>1905-06-08 00:00:00</td>
      <td>91</td>
      <td>7.100000</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>中国大陆</th>
      <td>魔窟中的幻想</td>
      <td>51</td>
      <td>惊悚/恐怖/儿童</td>
      <td>1905-06-08 00:00:00</td>
      <td>78</td>
      <td>8.000000</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>1977</th>
      <th>俄罗斯</th>
      <td>列宁格勒围困之星火战役 Блокада: Фильм 2: Ленинградский ме...</td>
      <td>32</td>
      <td>剧情/战争</td>
      <td>1905-05-30 00:00:00</td>
      <td>97</td>
      <td>6.600000</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>2018</th>
      <th>美国</th>
      <td>复仇者联盟3</td>
      <td>123456</td>
      <td>剧情/科幻</td>
      <td>2018-05-04 00:00:00</td>
      <td>142</td>
      <td>6.935704</td>
      <td>美国</td>
    </tr>
  </tbody>
</table>
<p>38163 rows × 7 columns</p>



```python
df.loc[1994]
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>名字</th>
      <th>投票人数</th>
      <th>类型</th>
      <th>上映时间</th>
      <th>时长</th>
      <th>评分</th>
      <th>首映地点</th>
    </tr>
    <tr>
      <th>产地</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>美国</th>
      <td>肖申克的救赎</td>
      <td>692795</td>
      <td>剧情/犯罪</td>
      <td>1994-09-10 00:00:00</td>
      <td>142</td>
      <td>9.6</td>
      <td>多伦多电影节</td>
    </tr>
    <tr>
      <th>美国</th>
      <td>阿甘正传</td>
      <td>580897</td>
      <td>剧情/爱情</td>
      <td>1994-06-23 00:00:00</td>
      <td>142</td>
      <td>9.4</td>
      <td>洛杉矶首映</td>
    </tr>
    <tr>
      <th>法国</th>
      <td>这个杀手不太冷</td>
      <td>662552</td>
      <td>剧情/动作/犯罪</td>
      <td>1994-09-14 00:00:00</td>
      <td>133</td>
      <td>9.4</td>
      <td>法国</td>
    </tr>
    <tr>
      <th>美国</th>
      <td>34街的</td>
      <td>768</td>
      <td>剧情/家庭/奇幻</td>
      <td>1994-12-23 00:00:00</td>
      <td>114</td>
      <td>7.9</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>中国大陆</th>
      <td>活着</td>
      <td>202794</td>
      <td>剧情/家庭</td>
      <td>1994-05-18 00:00:00</td>
      <td>132</td>
      <td>9.0</td>
      <td>法国</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>美国</th>
      <td>鬼精灵2： 恐怖</td>
      <td>60</td>
      <td>喜剧/恐怖/奇幻</td>
      <td>1994-04-08 00:00:00</td>
      <td>85</td>
      <td>5.8</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>英国</th>
      <td>黑色第16</td>
      <td>44</td>
      <td>剧情/惊悚</td>
      <td>1996-02-01 00:00:00</td>
      <td>106</td>
      <td>6.8</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>日本</th>
      <td>蜡笔小新之布里布里王国的秘密宝藏 クレヨンしんちゃん ブリブリ王国の</td>
      <td>2142</td>
      <td>动画</td>
      <td>1994-04-23 00:00:00</td>
      <td>94</td>
      <td>7.7</td>
      <td>日本</td>
    </tr>
    <tr>
      <th>日本</th>
      <td>龙珠Z剧场版10：两人面临危机! 超战士难以成眠 ドラゴンボール Z 劇場版：危険なふたり！</td>
      <td>579</td>
      <td>动画</td>
      <td>1994-03-12 00:00:00</td>
      <td>53</td>
      <td>7.2</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>中国香港</th>
      <td>重案实录之惊天械劫案 重案實錄之驚天械劫</td>
      <td>90</td>
      <td>动作/犯罪</td>
      <td>1905-06-16 00:00:00</td>
      <td>114</td>
      <td>7.3</td>
      <td>美国</td>
    </tr>
  </tbody>
</table>
<p>489 rows × 7 columns</p>


#### 3.10.1.5 取消层次化索引

```python
df = df.reset_index()
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
    </tr>
  </tbody>
</table>


### 3.10.2 数据旋转 

行列转化：以前5部电影为例


```python
data = df[:5]
data
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
    </tr>
  </tbody>
</table>


.T可以直接让数据的行列进行交换


```python
data.T #似曾相识,Numpy的矩阵转置
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>年代</th>
      <td>1994</td>
      <td>1957</td>
      <td>1997</td>
      <td>1994</td>
      <td>1993</td>
    </tr>
    <tr>
      <th>产地</th>
      <td>美国</td>
      <td>美国</td>
      <td>意大利</td>
      <td>美国</td>
      <td>中国大陆</td>
    </tr>
    <tr>
      <th>名字</th>
      <td>肖申克的救赎</td>
      <td>控方证人</td>
      <td>美丽人生</td>
      <td>阿甘正传</td>
      <td>霸王别姬</td>
    </tr>
    <tr>
      <th>投票人数</th>
      <td>692795</td>
      <td>42995</td>
      <td>327855</td>
      <td>580897</td>
      <td>478523</td>
    </tr>
    <tr>
      <th>类型</th>
      <td>剧情/犯罪</td>
      <td>剧情/悬疑/犯罪</td>
      <td>剧情/喜剧/爱情</td>
      <td>剧情/爱情</td>
      <td>剧情/爱情/同性</td>
    </tr>
    <tr>
      <th>上映时间</th>
      <td>1994-09-10 00:00:00</td>
      <td>1957-12-17 00:00:00</td>
      <td>1997-12-20 00:00:00</td>
      <td>1994-06-23 00:00:00</td>
      <td>1993-01-01 00:00:00</td>
    </tr>
    <tr>
      <th>时长</th>
      <td>142</td>
      <td>116</td>
      <td>116</td>
      <td>142</td>
      <td>171</td>
    </tr>
    <tr>
      <th>评分</th>
      <td>9.6</td>
      <td>9.5</td>
      <td>9.5</td>
      <td>9.4</td>
      <td>9.4</td>
    </tr>
    <tr>
      <th>首映地点</th>
      <td>多伦多电影节</td>
      <td>美国</td>
      <td>意大利</td>
      <td>洛杉矶首映</td>
      <td>香港</td>
    </tr>
  </tbody>
</table>


**dataframe也可以使用stack和unstack，转化为层次化索引的Series** 

```python
data.stack()
```


    0  年代                     1994
       产地                       美国
       名字                   肖申克的救赎
       投票人数                 692795
       类型                    剧情/犯罪
       上映时间    1994-09-10 00:00:00
       时长                      142
       评分                      9.6
       首映地点                 多伦多电影节
    1  年代                     1957
       产地                       美国
       名字                     控方证人
       投票人数                  42995
       类型                 剧情/悬疑/犯罪
       上映时间    1957-12-17 00:00:00
       时长                      116
       评分                      9.5
       首映地点                     美国
    2  年代                     1997
       产地                      意大利
       名字                    美丽人生 
       投票人数                 327855
       类型                 剧情/喜剧/爱情
       上映时间    1997-12-20 00:00:00
       时长                      116
       评分                      9.5
       首映地点                    意大利
    3  年代                     1994
       产地                       美国
       名字                     阿甘正传
       投票人数                 580897
       类型                    剧情/爱情
       上映时间    1994-06-23 00:00:00
       时长                      142
       评分                      9.4
       首映地点                  洛杉矶首映
    4  年代                     1993
       产地                     中国大陆
       名字                     霸王别姬
       投票人数                 478523
       类型                 剧情/爱情/同性
       上映时间    1993-01-01 00:00:00
       时长                      171
       评分                      9.4
       首映地点                     香港
    dtype: object




```python
data.stack().unstack()  #转回来
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
      <td>1994-09-10</td>
      <td>142</td>
      <td>9.6</td>
      <td>多伦多电影节</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1957</td>
      <td>美国</td>
      <td>控方证人</td>
      <td>42995</td>
      <td>剧情/悬疑/犯罪</td>
      <td>1957-12-17</td>
      <td>116</td>
      <td>9.5</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1997</td>
      <td>意大利</td>
      <td>美丽人生</td>
      <td>327855</td>
      <td>剧情/喜剧/爱情</td>
      <td>1997-12-20</td>
      <td>116</td>
      <td>9.5</td>
      <td>意大利</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1994</td>
      <td>美国</td>
      <td>阿甘正传</td>
      <td>580897</td>
      <td>剧情/爱情</td>
      <td>1994-06-23</td>
      <td>142</td>
      <td>9.4</td>
      <td>洛杉矶首映</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1993</td>
      <td>中国大陆</td>
      <td>霸王别姬</td>
      <td>478523</td>
      <td>剧情/爱情/同性</td>
      <td>1993-01-01</td>
      <td>171</td>
      <td>9.4</td>
      <td>香港</td>
    </tr>
  </tbody>
</table>


## 3.11 数据分组，分组运算

### 3.11.1 GroupBy技术：实现数据的分组，和分组运算，作用类似于数据透视表

按照电影的产地进行分组 


```python
group = df.groupby(df["产地"])
```

### 3.11.2 先定义一个分组变量group 


```python
type(group)
```


    pandas.core.groupby.generic.DataFrameGroupBy



### 3.11.3 计算分组后各个的统计量 


```python
group.mean() 
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>年代</th>
      <th>投票人数</th>
      <th>时长</th>
      <th>评分</th>
    </tr>
    <tr>
      <th>产地</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>中国台湾</th>
      <td>1999.009709</td>
      <td>8474.864078</td>
      <td>87.257282</td>
      <td>7.066667</td>
    </tr>
    <tr>
      <th>中国大陆</th>
      <td>2004.582432</td>
      <td>10915.587708</td>
      <td>81.517014</td>
      <td>6.062991</td>
    </tr>
    <tr>
      <th>中国香港</th>
      <td>1991.088865</td>
      <td>8141.709870</td>
      <td>88.553214</td>
      <td>6.473551</td>
    </tr>
    <tr>
      <th>丹麦</th>
      <td>1999.091371</td>
      <td>2003.781726</td>
      <td>88.507614</td>
      <td>7.246701</td>
    </tr>
    <tr>
      <th>俄罗斯</th>
      <td>1984.892857</td>
      <td>1021.180672</td>
      <td>96.100840</td>
      <td>7.557143</td>
    </tr>
    <tr>
      <th>其他</th>
      <td>1998.721721</td>
      <td>1619.144450</td>
      <td>87.656399</td>
      <td>7.226713</td>
    </tr>
    <tr>
      <th>加拿大</th>
      <td>2002.520451</td>
      <td>1921.834979</td>
      <td>80.592384</td>
      <td>6.727221</td>
    </tr>
    <tr>
      <th>印度</th>
      <td>2006.039326</td>
      <td>3219.587079</td>
      <td>120.949438</td>
      <td>6.864888</td>
    </tr>
    <tr>
      <th>墨西哥</th>
      <td>1992.786325</td>
      <td>1191.982906</td>
      <td>92.641026</td>
      <td>7.085470</td>
    </tr>
    <tr>
      <th>巴西</th>
      <td>1999.888889</td>
      <td>3606.333333</td>
      <td>88.373737</td>
      <td>7.232323</td>
    </tr>
    <tr>
      <th>德国</th>
      <td>1996.053869</td>
      <td>2624.736533</td>
      <td>92.258570</td>
      <td>7.187365</td>
    </tr>
    <tr>
      <th>意大利</th>
      <td>1985.599190</td>
      <td>3374.955466</td>
      <td>104.333333</td>
      <td>7.183131</td>
    </tr>
    <tr>
      <th>日本</th>
      <td>1999.886536</td>
      <td>3592.015781</td>
      <td>85.010587</td>
      <td>7.192569</td>
    </tr>
    <tr>
      <th>比利时</th>
      <td>1999.503650</td>
      <td>1244.153285</td>
      <td>83.065693</td>
      <td>7.197080</td>
    </tr>
    <tr>
      <th>法国</th>
      <td>1991.794044</td>
      <td>3663.066380</td>
      <td>90.249013</td>
      <td>7.243093</td>
    </tr>
    <tr>
      <th>波兰</th>
      <td>1987.027624</td>
      <td>881.640884</td>
      <td>80.734807</td>
      <td>7.441989</td>
    </tr>
    <tr>
      <th>泰国</th>
      <td>2009.129252</td>
      <td>5322.724490</td>
      <td>88.442177</td>
      <td>6.109184</td>
    </tr>
    <tr>
      <th>澳大利亚</th>
      <td>2002.966102</td>
      <td>4798.111864</td>
      <td>85.593220</td>
      <td>6.953559</td>
    </tr>
    <tr>
      <th>瑞典</th>
      <td>1987.106952</td>
      <td>1549.700535</td>
      <td>94.625668</td>
      <td>7.425668</td>
    </tr>
    <tr>
      <th>美国</th>
      <td>1994.519891</td>
      <td>8677.294861</td>
      <td>89.976097</td>
      <td>6.923351</td>
    </tr>
    <tr>
      <th>英国</th>
      <td>1996.630926</td>
      <td>4979.837848</td>
      <td>89.213318</td>
      <td>7.498420</td>
    </tr>
    <tr>
      <th>荷兰</th>
      <td>2001.198675</td>
      <td>957.589404</td>
      <td>75.887417</td>
      <td>7.160265</td>
    </tr>
    <tr>
      <th>西班牙</th>
      <td>2001.546275</td>
      <td>3355.266366</td>
      <td>90.905192</td>
      <td>7.025056</td>
    </tr>
    <tr>
      <th>阿根廷</th>
      <td>2004.212389</td>
      <td>2283.938053</td>
      <td>92.548673</td>
      <td>7.248673</td>
    </tr>
    <tr>
      <th>韩国</th>
      <td>2008.100596</td>
      <td>6527.518629</td>
      <td>100.018629</td>
      <td>6.351118</td>
    </tr>
  </tbody>
</table>



```python
group.sum()
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>年代</th>
      <th>投票人数</th>
      <th>时长</th>
      <th>评分</th>
    </tr>
    <tr>
      <th>产地</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>中国台湾</th>
      <td>1235388</td>
      <td>5237466</td>
      <td>53925</td>
      <td>4367.200000</td>
    </tr>
    <tr>
      <th>中国大陆</th>
      <td>7599372</td>
      <td>41380993</td>
      <td>309031</td>
      <td>22984.800000</td>
    </tr>
    <tr>
      <th>中国香港</th>
      <td>5668630</td>
      <td>23179448</td>
      <td>252111</td>
      <td>18430.200000</td>
    </tr>
    <tr>
      <th>丹麦</th>
      <td>393821</td>
      <td>394745</td>
      <td>17436</td>
      <td>1427.600000</td>
    </tr>
    <tr>
      <th>俄罗斯</th>
      <td>944809</td>
      <td>486082</td>
      <td>45744</td>
      <td>3597.200000</td>
    </tr>
    <tr>
      <th>其他</th>
      <td>3763593</td>
      <td>3048849</td>
      <td>165057</td>
      <td>13607.900000</td>
    </tr>
    <tr>
      <th>加拿大</th>
      <td>1419787</td>
      <td>1362581</td>
      <td>57140</td>
      <td>4769.600000</td>
    </tr>
    <tr>
      <th>印度</th>
      <td>714150</td>
      <td>1146173</td>
      <td>43058</td>
      <td>2443.900000</td>
    </tr>
    <tr>
      <th>墨西哥</th>
      <td>233156</td>
      <td>139462</td>
      <td>10839</td>
      <td>829.000000</td>
    </tr>
    <tr>
      <th>巴西</th>
      <td>197989</td>
      <td>357027</td>
      <td>8749</td>
      <td>716.000000</td>
    </tr>
    <tr>
      <th>德国</th>
      <td>2037971</td>
      <td>2679856</td>
      <td>94196</td>
      <td>7338.300000</td>
    </tr>
    <tr>
      <th>意大利</th>
      <td>1471329</td>
      <td>2500842</td>
      <td>77311</td>
      <td>5322.700000</td>
    </tr>
    <tr>
      <th>日本</th>
      <td>10011432</td>
      <td>17981631</td>
      <td>425563</td>
      <td>36006.000000</td>
    </tr>
    <tr>
      <th>比利时</th>
      <td>273932</td>
      <td>170449</td>
      <td>11380</td>
      <td>986.000000</td>
    </tr>
    <tr>
      <th>法国</th>
      <td>5551130</td>
      <td>10208966</td>
      <td>251524</td>
      <td>20186.500000</td>
    </tr>
    <tr>
      <th>波兰</th>
      <td>359652</td>
      <td>159577</td>
      <td>14613</td>
      <td>1347.000000</td>
    </tr>
    <tr>
      <th>泰国</th>
      <td>590684</td>
      <td>1564881</td>
      <td>26002</td>
      <td>1796.100000</td>
    </tr>
    <tr>
      <th>澳大利亚</th>
      <td>590875</td>
      <td>1415443</td>
      <td>25250</td>
      <td>2051.300000</td>
    </tr>
    <tr>
      <th>瑞典</th>
      <td>371589</td>
      <td>289794</td>
      <td>17695</td>
      <td>1388.600000</td>
    </tr>
    <tr>
      <th>美国</th>
      <td>23363806</td>
      <td>101645832</td>
      <td>1053980</td>
      <td>81100.135704</td>
    </tr>
    <tr>
      <th>英国</th>
      <td>5307045</td>
      <td>13236409</td>
      <td>237129</td>
      <td>19930.800000</td>
    </tr>
    <tr>
      <th>荷兰</th>
      <td>302181</td>
      <td>144596</td>
      <td>11459</td>
      <td>1081.200000</td>
    </tr>
    <tr>
      <th>西班牙</th>
      <td>886685</td>
      <td>1486383</td>
      <td>40271</td>
      <td>3112.100000</td>
    </tr>
    <tr>
      <th>阿根廷</th>
      <td>226476</td>
      <td>258085</td>
      <td>10458</td>
      <td>819.100000</td>
    </tr>
    <tr>
      <th>韩国</th>
      <td>2694871</td>
      <td>8759930</td>
      <td>134225</td>
      <td>8523.200000</td>
    </tr>
  </tbody>
</table>



### 3.11.4 计算每年的平均评分 


```python
df["评分"].groupby(df["年代"]).mean()
```


    年代
    1888    7.950000
    1890    4.800000
    1892    7.500000
    1894    6.633333
    1895    7.575000
              ...   
    2013    6.375974
    2014    6.249384
    2015    6.121925
    2016    5.834524
    2018    6.935704
    Name: 评分, Length: 127, dtype: float64



### 3.11.5 只对数值变量进行分组运算 


```python
df["年代"] = df["年代"].astype("str")
df.groupby(df["产地"]).median() #不会再对年代进行求取
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>投票人数</th>
      <th>时长</th>
      <th>评分</th>
    </tr>
    <tr>
      <th>产地</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>中国台湾</th>
      <td>487.0</td>
      <td>92.0</td>
      <td>7.1</td>
    </tr>
    <tr>
      <th>中国大陆</th>
      <td>502.0</td>
      <td>90.0</td>
      <td>6.4</td>
    </tr>
    <tr>
      <th>中国香港</th>
      <td>637.0</td>
      <td>92.0</td>
      <td>6.5</td>
    </tr>
    <tr>
      <th>丹麦</th>
      <td>182.0</td>
      <td>94.0</td>
      <td>7.3</td>
    </tr>
    <tr>
      <th>俄罗斯</th>
      <td>132.5</td>
      <td>93.0</td>
      <td>7.7</td>
    </tr>
    <tr>
      <th>其他</th>
      <td>158.0</td>
      <td>90.0</td>
      <td>7.4</td>
    </tr>
    <tr>
      <th>加拿大</th>
      <td>258.0</td>
      <td>89.0</td>
      <td>6.8</td>
    </tr>
    <tr>
      <th>印度</th>
      <td>139.0</td>
      <td>131.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>墨西哥</th>
      <td>183.0</td>
      <td>94.0</td>
      <td>7.2</td>
    </tr>
    <tr>
      <th>巴西</th>
      <td>131.0</td>
      <td>96.0</td>
      <td>7.3</td>
    </tr>
    <tr>
      <th>德国</th>
      <td>212.0</td>
      <td>94.0</td>
      <td>7.3</td>
    </tr>
    <tr>
      <th>意大利</th>
      <td>187.0</td>
      <td>101.0</td>
      <td>7.3</td>
    </tr>
    <tr>
      <th>日本</th>
      <td>359.0</td>
      <td>89.0</td>
      <td>7.3</td>
    </tr>
    <tr>
      <th>比利时</th>
      <td>226.0</td>
      <td>90.0</td>
      <td>7.3</td>
    </tr>
    <tr>
      <th>法国</th>
      <td>244.0</td>
      <td>95.0</td>
      <td>7.3</td>
    </tr>
    <tr>
      <th>波兰</th>
      <td>174.0</td>
      <td>87.0</td>
      <td>7.5</td>
    </tr>
    <tr>
      <th>泰国</th>
      <td>542.5</td>
      <td>92.5</td>
      <td>6.2</td>
    </tr>
    <tr>
      <th>澳大利亚</th>
      <td>323.0</td>
      <td>95.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>瑞典</th>
      <td>191.0</td>
      <td>96.0</td>
      <td>7.6</td>
    </tr>
    <tr>
      <th>美国</th>
      <td>415.0</td>
      <td>93.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>英国</th>
      <td>345.0</td>
      <td>92.0</td>
      <td>7.6</td>
    </tr>
    <tr>
      <th>荷兰</th>
      <td>180.0</td>
      <td>85.0</td>
      <td>7.3</td>
    </tr>
    <tr>
      <th>西班牙</th>
      <td>267.0</td>
      <td>97.0</td>
      <td>7.1</td>
    </tr>
    <tr>
      <th>阿根廷</th>
      <td>146.0</td>
      <td>97.0</td>
      <td>7.3</td>
    </tr>
    <tr>
      <th>韩国</th>
      <td>1007.0</td>
      <td>104.0</td>
      <td>6.5</td>
    </tr>
  </tbody>
</table>




### 3.11.6 传入多个分组变量 


```python
df.groupby([df["产地"],df["年代"]]).mean() #根据两个变量进行分组
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>投票人数</th>
      <th>时长</th>
      <th>评分</th>
    </tr>
    <tr>
      <th>产地</th>
      <th>年代</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">中国台湾</th>
      <th>1963</th>
      <td>121.000000</td>
      <td>113.000000</td>
      <td>6.400000</td>
    </tr>
    <tr>
      <th>1965</th>
      <td>153.666667</td>
      <td>105.000000</td>
      <td>6.800000</td>
    </tr>
    <tr>
      <th>1966</th>
      <td>51.000000</td>
      <td>60.000000</td>
      <td>7.900000</td>
    </tr>
    <tr>
      <th>1967</th>
      <td>4444.000000</td>
      <td>112.000000</td>
      <td>8.000000</td>
    </tr>
    <tr>
      <th>1968</th>
      <td>89.000000</td>
      <td>83.000000</td>
      <td>7.400000</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">韩国</th>
      <th>2012</th>
      <td>5812.542857</td>
      <td>100.771429</td>
      <td>6.035238</td>
    </tr>
    <tr>
      <th>2013</th>
      <td>10470.370370</td>
      <td>97.731481</td>
      <td>6.062037</td>
    </tr>
    <tr>
      <th>2014</th>
      <td>3776.266667</td>
      <td>98.666667</td>
      <td>5.650833</td>
    </tr>
    <tr>
      <th>2015</th>
      <td>3209.247706</td>
      <td>100.266055</td>
      <td>5.423853</td>
    </tr>
    <tr>
      <th>2016</th>
      <td>1739.850000</td>
      <td>106.100000</td>
      <td>5.730000</td>
    </tr>
  </tbody>
</table>
<p>1578 rows × 3 columns</p>




### 3.11.7 获得每个地区，每一年的电影的评分的均值 


```python
group = df["评分"].groupby([df["产地"], df["年代"]])
means = group.mean()
means
```


    产地    年代  
    中国台湾  1963    6.400000
          1965    6.800000
          1966    7.900000
          1967    8.000000
          1968    7.400000
                    ...   
    韩国    2012    6.035238
          2013    6.062037
          2014    5.650833
          2015    5.423853
          2016    5.730000
    Name: 评分, Length: 1578, dtype: float64




```python
means = group = df["评分"].groupby([df["产地"], df["年代"]]).mean()
means
```


    产地    年代  
    中国台湾  1963    6.400000
          1965    6.800000
          1966    7.900000
          1967    8.000000
          1968    7.400000
                    ...   
    韩国    2012    6.035238
          2013    6.062037
          2014    5.650833
          2015    5.423853
          2016    5.730000
    Name: 评分, Length: 1578, dtype: float64



### 3.11.8 Series通过unstack方法转化为dataframe

**会产生缺失值**


```python
means.unstack().T
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>产地</th>
      <th>中国台湾</th>
      <th>中国大陆</th>
      <th>中国香港</th>
      <th>丹麦</th>
      <th>俄罗斯</th>
      <th>其他</th>
      <th>加拿大</th>
      <th>印度</th>
      <th>墨西哥</th>
      <th>巴西</th>
      <th>...</th>
      <th>波兰</th>
      <th>泰国</th>
      <th>澳大利亚</th>
      <th>瑞典</th>
      <th>美国</th>
      <th>英国</th>
      <th>荷兰</th>
      <th>西班牙</th>
      <th>阿根廷</th>
      <th>韩国</th>
    </tr>
    <tr>
      <th>年代</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1888</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.950000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1890</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.800000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1892</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1894</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6.450000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1895</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2013</th>
      <td>7.076471</td>
      <td>5.306500</td>
      <td>6.105714</td>
      <td>6.555556</td>
      <td>6.875000</td>
      <td>6.853571</td>
      <td>6.018182</td>
      <td>6.400000</td>
      <td>6.983333</td>
      <td>8.00</td>
      <td>...</td>
      <td>6.966667</td>
      <td>5.568000</td>
      <td>6.76000</td>
      <td>7.100</td>
      <td>6.308255</td>
      <td>7.460140</td>
      <td>6.33</td>
      <td>6.358333</td>
      <td>6.616667</td>
      <td>6.062037</td>
    </tr>
    <tr>
      <th>2014</th>
      <td>6.522222</td>
      <td>4.963830</td>
      <td>5.616667</td>
      <td>7.120000</td>
      <td>7.175000</td>
      <td>6.596250</td>
      <td>5.921739</td>
      <td>6.374194</td>
      <td>7.250000</td>
      <td>6.86</td>
      <td>...</td>
      <td>7.060000</td>
      <td>5.653571</td>
      <td>6.56875</td>
      <td>6.960</td>
      <td>6.393056</td>
      <td>7.253398</td>
      <td>7.30</td>
      <td>6.868750</td>
      <td>7.150000</td>
      <td>5.650833</td>
    </tr>
    <tr>
      <th>2015</th>
      <td>6.576000</td>
      <td>4.969189</td>
      <td>5.589189</td>
      <td>7.166667</td>
      <td>7.342857</td>
      <td>6.732727</td>
      <td>6.018750</td>
      <td>6.736364</td>
      <td>6.500000</td>
      <td>6.76</td>
      <td>...</td>
      <td>6.300000</td>
      <td>5.846667</td>
      <td>6.88000</td>
      <td>7.625</td>
      <td>6.231486</td>
      <td>7.123256</td>
      <td>6.70</td>
      <td>6.514286</td>
      <td>7.233333</td>
      <td>5.423853</td>
    </tr>
    <tr>
      <th>2016</th>
      <td>NaN</td>
      <td>4.712000</td>
      <td>5.390909</td>
      <td>7.000000</td>
      <td>NaN</td>
      <td>6.833333</td>
      <td>6.200000</td>
      <td>6.900000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6.522581</td>
      <td>7.200000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5.730000</td>
    </tr>
    <tr>
      <th>2018</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6.935704</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>127 rows × 25 columns</p>





## 3.12 离散化处理 

**在实际的数据分析项目中，对有的数据属性，我们往往并不关注数据的绝对取值，只关心它所处的区间或者等级**

**比如，我们可以把评分9分及以上的电影定义为A，7到9分定义为B，5到7分定义为C，3到5分定义为D，小于3分定义为E。**

**离散化也可称为分组、区间化。**

Pandas为我们提供了方便的函数**cut()**:

**pd.cut(x,bins,right = True,labels = None, retbins = False,precision = 3,include_lowest = False)** 

参数解释：

**x：需要离散化的数组、Series、DataFrame对象**

**bins：分组的依据，right = True，include_lowest = False，默认左开右闭，可以自己调整。**

**labels：是否要用标记来替换返回出来的数组，retbins：返回x当中每一个值对应的bins的列表，precision精度。**


```python
df["评分等级"] = pd.cut(df["评分"], [0,3,5,7,9,10], labels = ['E','D','C','B','A']) #labels要和区间划分一一对应
df
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
      <td>9.600000</td>
      <td>多伦多电影节</td>
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
      <td>9.500000</td>
      <td>美国</td>
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
      <td>9.500000</td>
      <td>意大利</td>
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
      <td>9.400000</td>
      <td>洛杉矶首映</td>
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
      <td>9.400000</td>
      <td>香港</td>
      <td>A</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>38158</th>
      <td>1935</td>
      <td>美国</td>
      <td>1935年</td>
      <td>57</td>
      <td>喜剧/歌舞</td>
      <td>1935-03-15 00:00:00</td>
      <td>98</td>
      <td>7.600000</td>
      <td>美国</td>
      <td>B</td>
    </tr>
    <tr>
      <th>38159</th>
      <td>1986</td>
      <td>中国大陆</td>
      <td>血溅画屏</td>
      <td>95</td>
      <td>剧情/悬疑/犯罪/武侠/古装</td>
      <td>1905-06-08 00:00:00</td>
      <td>91</td>
      <td>7.100000</td>
      <td>美国</td>
      <td>B</td>
    </tr>
    <tr>
      <th>38160</th>
      <td>1986</td>
      <td>中国大陆</td>
      <td>魔窟中的幻想</td>
      <td>51</td>
      <td>惊悚/恐怖/儿童</td>
      <td>1905-06-08 00:00:00</td>
      <td>78</td>
      <td>8.000000</td>
      <td>美国</td>
      <td>B</td>
    </tr>
    <tr>
      <th>38161</th>
      <td>1977</td>
      <td>俄罗斯</td>
      <td>列宁格勒围困之星火战役 Блокада: Фильм 2: Ленинградский ме...</td>
      <td>32</td>
      <td>剧情/战争</td>
      <td>1905-05-30 00:00:00</td>
      <td>97</td>
      <td>6.600000</td>
      <td>美国</td>
      <td>C</td>
    </tr>
    <tr>
      <th>38162</th>
      <td>2018</td>
      <td>美国</td>
      <td>复仇者联盟3</td>
      <td>123456</td>
      <td>剧情/科幻</td>
      <td>2018-05-04 00:00:00</td>
      <td>142</td>
      <td>6.935704</td>
      <td>美国</td>
      <td>C</td>
    </tr>
  </tbody>
</table>
<p>38163 rows × 10 columns</p>


**同样的，我们可以根据投票人数来刻画电影的热门**

**投票越多的热门程度越高**


```python
bins = np.percentile(df["投票人数"], [0,20,40,60,80,100]) #获取分位数
df["热门程度"] = pd.cut(df["投票人数"],bins,labels = ['E','D','C','B','A'])
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




**大烂片集合：投票人数很多，评分很低**

**遗憾的是，我们可以发现，烂片几乎都是中国大陆的**


```python
df[(df.热门程度 == 'A') & (df.评分等级 == 'E')]
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
      <th>623</th>
      <td>2011</td>
      <td>中国大陆</td>
      <td>B区</td>
      <td>5187</td>
      <td>剧情/惊悚/恐怖</td>
      <td>2011-06-03 00:00:00</td>
      <td>89</td>
      <td>2.3</td>
      <td>中国大陆</td>
      <td>E</td>
      <td>A</td>
    </tr>
    <tr>
      <th>4167</th>
      <td>2014</td>
      <td>中国大陆</td>
      <td>怖偶</td>
      <td>4867</td>
      <td>悬疑/惊悚</td>
      <td>2014-05-07 00:00:00</td>
      <td>88</td>
      <td>2.8</td>
      <td>中国大陆</td>
      <td>E</td>
      <td>A</td>
    </tr>
    <tr>
      <th>5200</th>
      <td>2011</td>
      <td>中国大陆</td>
      <td>床下有人</td>
      <td>4309</td>
      <td>悬疑/惊悚</td>
      <td>2011-10-14 00:00:00</td>
      <td>100</td>
      <td>2.8</td>
      <td>中国大陆</td>
      <td>E</td>
      <td>A</td>
    </tr>
    <tr>
      <th>6585</th>
      <td>2013</td>
      <td>中国大陆</td>
      <td>帝国秘符</td>
      <td>4351</td>
      <td>动作/冒险</td>
      <td>2013-09-18 00:00:00</td>
      <td>93</td>
      <td>3.0</td>
      <td>中国大陆</td>
      <td>E</td>
      <td>A</td>
    </tr>
    <tr>
      <th>8009</th>
      <td>2011</td>
      <td>中国大陆</td>
      <td>飞天</td>
      <td>4764</td>
      <td>剧情</td>
      <td>2011-07-01 00:00:00</td>
      <td>115</td>
      <td>2.9</td>
      <td>中国大陆</td>
      <td>E</td>
      <td>A</td>
    </tr>
    <tr>
      <th>8181</th>
      <td>2014</td>
      <td>中国大陆</td>
      <td>分手达人</td>
      <td>3937</td>
      <td>喜剧/爱情</td>
      <td>2014-06-06 00:00:00</td>
      <td>90</td>
      <td>2.7</td>
      <td>中国大陆</td>
      <td>E</td>
      <td>A</td>
    </tr>
    <tr>
      <th>9372</th>
      <td>2012</td>
      <td>中国大陆</td>
      <td>孤岛惊魂</td>
      <td>2982</td>
      <td>悬疑/惊悚/恐怖</td>
      <td>2013-01-26 00:00:00</td>
      <td>93</td>
      <td>2.8</td>
      <td>中国大陆</td>
      <td>E</td>
      <td>A</td>
    </tr>
    <tr>
      <th>10275</th>
      <td>2013</td>
      <td>中国大陆</td>
      <td>海天盛宴·韦口</td>
      <td>3788</td>
      <td>情色</td>
      <td>2013-10-12 00:00:00</td>
      <td>88</td>
      <td>2.9</td>
      <td>网络</td>
      <td>E</td>
      <td>A</td>
    </tr>
    <tr>
      <th>16512</th>
      <td>2013</td>
      <td>中国大陆</td>
      <td>孪生密码</td>
      <td>6390</td>
      <td>动作/悬疑</td>
      <td>2013-11-08 00:00:00</td>
      <td>96</td>
      <td>2.9</td>
      <td>中国大陆</td>
      <td>E</td>
      <td>A</td>
    </tr>
    <tr>
      <th>21189</th>
      <td>2010</td>
      <td>日本</td>
      <td>拳皇</td>
      <td>6329</td>
      <td>动作/科幻/冒险</td>
      <td>2012-10-12 00:00:00</td>
      <td>93</td>
      <td>3.0</td>
      <td>中国大陆</td>
      <td>E</td>
      <td>A</td>
    </tr>
    <tr>
      <th>22348</th>
      <td>2013</td>
      <td>中国大陆</td>
      <td>闪魂</td>
      <td>3119</td>
      <td>惊悚/犯罪</td>
      <td>2014-02-21 00:00:00</td>
      <td>94</td>
      <td>2.6</td>
      <td>中国大陆</td>
      <td>E</td>
      <td>A</td>
    </tr>
    <tr>
      <th>22524</th>
      <td>2015</td>
      <td>中国大陆</td>
      <td>少年毛泽东</td>
      <td>3058</td>
      <td>动画/儿童/冒险</td>
      <td>2015-04-30 00:00:00</td>
      <td>76</td>
      <td>2.4</td>
      <td>中国大陆</td>
      <td>E</td>
      <td>A</td>
    </tr>
    <tr>
      <th>23754</th>
      <td>2013</td>
      <td>英国</td>
      <td>史前怪兽</td>
      <td>3543</td>
      <td>动作/惊悚/冒险</td>
      <td>2014-01-01 00:00:00</td>
      <td>89</td>
      <td>3.0</td>
      <td>中国大陆</td>
      <td>E</td>
      <td>A</td>
    </tr>
    <tr>
      <th>27832</th>
      <td>2011</td>
      <td>中国大陆</td>
      <td>无极限之危情速递</td>
      <td>6319</td>
      <td>喜剧/动作/爱情/冒险</td>
      <td>2011-08-12 00:00:00</td>
      <td>94</td>
      <td>2.8</td>
      <td>中国大陆</td>
      <td>E</td>
      <td>A</td>
    </tr>
    <tr>
      <th>31622</th>
      <td>2010</td>
      <td>中国大陆</td>
      <td>异度公寓</td>
      <td>3639</td>
      <td>惊悚</td>
      <td>2010-06-04 00:00:00</td>
      <td>93</td>
      <td>2.7</td>
      <td>中国大陆</td>
      <td>E</td>
      <td>A</td>
    </tr>
    <tr>
      <th>32007</th>
      <td>2014</td>
      <td>中国大陆</td>
      <td>英雄之战</td>
      <td>8359</td>
      <td>动作/爱情</td>
      <td>2014-03-21 00:00:00</td>
      <td>90</td>
      <td>3.0</td>
      <td>中国大陆</td>
      <td>E</td>
      <td>A</td>
    </tr>
    <tr>
      <th>32180</th>
      <td>2013</td>
      <td>中国大陆</td>
      <td>咏春小龙</td>
      <td>8861</td>
      <td>剧情/动作</td>
      <td>2013-07-20 00:00:00</td>
      <td>90</td>
      <td>3.0</td>
      <td>中国大陆</td>
      <td>E</td>
      <td>A</td>
    </tr>
    <tr>
      <th>32990</th>
      <td>2014</td>
      <td>中国大陆</td>
      <td>再爱一次好不好</td>
      <td>6999</td>
      <td>喜剧/爱情</td>
      <td>2014-04-11 00:00:00</td>
      <td>94</td>
      <td>3.0</td>
      <td>中国大陆</td>
      <td>E</td>
      <td>A</td>
    </tr>
    <tr>
      <th>38090</th>
      <td>2014</td>
      <td>中国大陆</td>
      <td>大话天仙</td>
      <td>21629</td>
      <td>喜剧/奇幻/古装</td>
      <td>2014-02-02 00:00:00</td>
      <td>91</td>
      <td>3.0</td>
      <td>中国大陆</td>
      <td>E</td>
      <td>A</td>
    </tr>
    <tr>
      <th>38092</th>
      <td>2013</td>
      <td>中国大陆</td>
      <td>天机·富春山居图</td>
      <td>74709</td>
      <td>动作/冒险</td>
      <td>2013-06-09 00:00:00</td>
      <td>122</td>
      <td>2.9</td>
      <td>中国大陆</td>
      <td>E</td>
      <td>A</td>
    </tr>
    <tr>
      <th>38093</th>
      <td>2014</td>
      <td>中国大陆</td>
      <td>特工艾米拉</td>
      <td>10852</td>
      <td>动作/悬疑</td>
      <td>2014-04-11 00:00:00</td>
      <td>96</td>
      <td>2.7</td>
      <td>中国大陆</td>
      <td>E</td>
      <td>A</td>
    </tr>
    <tr>
      <th>38097</th>
      <td>2015</td>
      <td>中国大陆</td>
      <td>汽车人总动员</td>
      <td>12892</td>
      <td>喜剧/动画/冒险</td>
      <td>2015-07-03 00:00:00</td>
      <td>85</td>
      <td>2.3</td>
      <td>中国大陆</td>
      <td>E</td>
      <td>A</td>
    </tr>
    <tr>
      <th>38102</th>
      <td>2016</td>
      <td>中国大陆</td>
      <td>2016年中央电视台春节</td>
      <td>17328</td>
      <td>歌舞/真人秀</td>
      <td>2016-02-07 00:00:00</td>
      <td>280</td>
      <td>2.3</td>
      <td>中国大陆</td>
      <td>E</td>
      <td>A</td>
    </tr>
    <tr>
      <th>38108</th>
      <td>2014</td>
      <td>中国大陆</td>
      <td>放手爱</td>
      <td>29254</td>
      <td>喜剧/爱情</td>
      <td>2014-04-30 00:00:00</td>
      <td>93</td>
      <td>2.3</td>
      <td>中国大陆</td>
      <td>E</td>
      <td>A</td>
    </tr>
  </tbody>
</table>




**冷门高分电影**


```python
df[(df.热门程度 == 'E') & (df.评分等级 == 'A')]
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
      <th>563</th>
      <td>2011</td>
      <td>英国</td>
      <td>BBC喜剧音</td>
      <td>38</td>
      <td>喜剧/音乐/歌舞</td>
      <td>2011-08-13 00:00:00</td>
      <td>95</td>
      <td>9.3</td>
      <td>美国</td>
      <td>A</td>
      <td>E</td>
    </tr>
    <tr>
      <th>895</th>
      <td>2014</td>
      <td>日本</td>
      <td>JOJO的奇妙冒险 特别见面会 Walk Like Crusade</td>
      <td>36</td>
      <td>纪录片</td>
      <td>2014-10-26 00:00:00</td>
      <td>137</td>
      <td>9.3</td>
      <td>美国</td>
      <td>A</td>
      <td>E</td>
    </tr>
    <tr>
      <th>1099</th>
      <td>2012</td>
      <td>英国</td>
      <td>Pond一家最</td>
      <td>45</td>
      <td>纪录片</td>
      <td>2012-09-29 00:00:00</td>
      <td>12</td>
      <td>9.2</td>
      <td>美国</td>
      <td>A</td>
      <td>E</td>
    </tr>
    <tr>
      <th>1540</th>
      <td>2007</td>
      <td>英国</td>
      <td>阿森纳：温格的十一人</td>
      <td>74</td>
      <td>运动</td>
      <td>2007-10-22 00:00:00</td>
      <td>78</td>
      <td>9.5</td>
      <td>美国</td>
      <td>A</td>
      <td>E</td>
    </tr>
    <tr>
      <th>1547</th>
      <td>2009</td>
      <td>英国</td>
      <td>阿斯加德远征</td>
      <td>59</td>
      <td>纪录片</td>
      <td>2011-09-17 00:00:00</td>
      <td>85</td>
      <td>9.3</td>
      <td>美国</td>
      <td>A</td>
      <td>E</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>36846</th>
      <td>2012</td>
      <td>中国大陆</td>
      <td>末了，未了</td>
      <td>34</td>
      <td>剧情/喜剧/爱情</td>
      <td>2012-12-16 00:00:00</td>
      <td>90</td>
      <td>9.5</td>
      <td>美国</td>
      <td>A</td>
      <td>E</td>
    </tr>
    <tr>
      <th>37000</th>
      <td>2015</td>
      <td>中国大陆</td>
      <td>身经百战</td>
      <td>74</td>
      <td>纪录片</td>
      <td>2015-03-24 00:00:00</td>
      <td>91</td>
      <td>9.1</td>
      <td>美国</td>
      <td>A</td>
      <td>E</td>
    </tr>
    <tr>
      <th>37033</th>
      <td>1986</td>
      <td>英国</td>
      <td>歌唱神探</td>
      <td>36</td>
      <td>剧情/悬疑/歌舞</td>
      <td>1986-11-16 00:00:00</td>
      <td>415</td>
      <td>9.1</td>
      <td>美国</td>
      <td>A</td>
      <td>E</td>
    </tr>
    <tr>
      <th>37557</th>
      <td>1975</td>
      <td>美国</td>
      <td>山那边</td>
      <td>70</td>
      <td>剧情</td>
      <td>1975-11-14 00:00:00</td>
      <td>103</td>
      <td>9.1</td>
      <td>美国</td>
      <td>A</td>
      <td>E</td>
    </tr>
    <tr>
      <th>37883</th>
      <td>2015</td>
      <td>美国</td>
      <td>奎</td>
      <td>62</td>
      <td>纪录片/短片</td>
      <td>2015-08-19 00:00:00</td>
      <td>9</td>
      <td>9.1</td>
      <td>纽约电影论坛</td>
      <td>A</td>
      <td>E</td>
    </tr>
  </tbody>
</table>
<p>177 rows × 11 columns</p>


**将处理后的数据进行保存 **


```python
df.to_excel("movie_data3.xlsx")
```



## 3.13 合并数据集

### 3.13.1 append

先把数据集拆分为多个，再进行合并

```python
df_usa = df[df.产地 == "美国"]
df_china = df[df.产地 == "中国大陆"]
```


```python
df_china.append(df_usa) #直接追加到后面，最好是变量相同的
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
      <th>4</th>
      <td>1993</td>
      <td>中国大陆</td>
      <td>霸王别姬</td>
      <td>478523</td>
      <td>剧情/爱情/同性</td>
      <td>1993-01-01 00:00:00</td>
      <td>171</td>
      <td>9.400000</td>
      <td>香港</td>
      <td>A</td>
      <td>A</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1961</td>
      <td>中国大陆</td>
      <td>大闹天宫</td>
      <td>74881</td>
      <td>动画/奇幻</td>
      <td>1905-05-14 00:00:00</td>
      <td>114</td>
      <td>9.200000</td>
      <td>上集</td>
      <td>A</td>
      <td>A</td>
    </tr>
    <tr>
      <th>29</th>
      <td>2015</td>
      <td>中国大陆</td>
      <td>穹顶之下</td>
      <td>51113</td>
      <td>纪录片</td>
      <td>2015-02-28 00:00:00</td>
      <td>104</td>
      <td>9.200000</td>
      <td>中国大陆</td>
      <td>A</td>
      <td>A</td>
    </tr>
    <tr>
      <th>38</th>
      <td>1982</td>
      <td>中国大陆</td>
      <td>茶馆</td>
      <td>10678</td>
      <td>剧情/历史</td>
      <td>1905-06-04 00:00:00</td>
      <td>118</td>
      <td>9.200000</td>
      <td>美国</td>
      <td>A</td>
      <td>A</td>
    </tr>
    <tr>
      <th>45</th>
      <td>1988</td>
      <td>中国大陆</td>
      <td>山水情</td>
      <td>10781</td>
      <td>动画/短片</td>
      <td>1905-06-10 00:00:00</td>
      <td>19</td>
      <td>9.200000</td>
      <td>美国</td>
      <td>A</td>
      <td>A</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>38151</th>
      <td>1987</td>
      <td>美国</td>
      <td>零下的激情</td>
      <td>199</td>
      <td>剧情/爱情/犯罪</td>
      <td>1987-11-06 00:00:00</td>
      <td>98</td>
      <td>7.400000</td>
      <td>美国</td>
      <td>B</td>
      <td>D</td>
    </tr>
    <tr>
      <th>38153</th>
      <td>1986</td>
      <td>美国</td>
      <td>离别秋波</td>
      <td>240</td>
      <td>剧情/爱情/音乐</td>
      <td>1986-02-19 00:00:00</td>
      <td>90</td>
      <td>8.200000</td>
      <td>美国</td>
      <td>B</td>
      <td>C</td>
    </tr>
    <tr>
      <th>38156</th>
      <td>1986</td>
      <td>美国</td>
      <td>极乐森林</td>
      <td>45</td>
      <td>纪录片</td>
      <td>1986-09-14 00:00:00</td>
      <td>90</td>
      <td>8.100000</td>
      <td>美国</td>
      <td>B</td>
      <td>E</td>
    </tr>
    <tr>
      <th>38158</th>
      <td>1935</td>
      <td>美国</td>
      <td>1935年</td>
      <td>57</td>
      <td>喜剧/歌舞</td>
      <td>1935-03-15 00:00:00</td>
      <td>98</td>
      <td>7.600000</td>
      <td>美国</td>
      <td>B</td>
      <td>E</td>
    </tr>
    <tr>
      <th>38162</th>
      <td>2018</td>
      <td>美国</td>
      <td>复仇者联盟3</td>
      <td>123456</td>
      <td>剧情/科幻</td>
      <td>2018-05-04 00:00:00</td>
      <td>142</td>
      <td>6.935704</td>
      <td>美国</td>
      <td>C</td>
      <td>A</td>
    </tr>
  </tbody>
</table>
<p>15505 rows × 11 columns</p>
将这两个数据集进行合并



### 3.13.2 merge？

```python
pd.merge(left, right, how = 'inner', on = None, left_on = None, right_on = None,
    left_index = False, right_index = False, sort = True,
    suffixes = ('_x', '_y'), copy = True, indicator = False, validate=None) 
```

left : DataFrame

right : DataFrame or named Series
    Object to merge with.

how : {'left', 'right', 'outer', 'inner'}, default 'inner'
    Type of merge to be performed.



以六部热门电影为例：


```python
df1 = df.loc[:5]
df1
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
    <tr>
      <th>5</th>
      <td>2012</td>
      <td>美国</td>
      <td>泰坦尼克号</td>
      <td>157074</td>
      <td>剧情/爱情/灾难</td>
      <td>2012-04-10 00:00:00</td>
      <td>194</td>
      <td>9.4</td>
      <td>中国大陆</td>
      <td>A</td>
      <td>A</td>
    </tr>
  </tbody>
</table>



```python
df2 = df.loc[:5][["名字","产地"]]
df2["票房"] = [123344,23454,55556,333,6666,444]
```


```python
df2
```



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>名字</th>
      <th>产地</th>
      <th>票房</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>肖申克的救赎</td>
      <td>美国</td>
      <td>123344</td>
    </tr>
    <tr>
      <th>1</th>
      <td>控方证人</td>
      <td>美国</td>
      <td>23454</td>
    </tr>
    <tr>
      <th>2</th>
      <td>美丽人生</td>
      <td>意大利</td>
      <td>55556</td>
    </tr>
    <tr>
      <th>3</th>
      <td>阿甘正传</td>
      <td>美国</td>
      <td>333</td>
    </tr>
    <tr>
      <th>4</th>
      <td>霸王别姬</td>
      <td>中国大陆</td>
      <td>6666</td>
    </tr>
    <tr>
      <th>5</th>
      <td>泰坦尼克号</td>
      <td>美国</td>
      <td>444</td>
    </tr>
  </tbody>
</table>





```python
df2 = df2.sample(frac = 1) #打乱数据
```


```python
df2.index = range(len(df2))
df2
```



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>名字</th>
      <th>产地</th>
      <th>票房</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>泰坦尼克号</td>
      <td>美国</td>
      <td>444</td>
    </tr>
    <tr>
      <th>1</th>
      <td>阿甘正传</td>
      <td>美国</td>
      <td>333</td>
    </tr>
    <tr>
      <th>2</th>
      <td>控方证人</td>
      <td>美国</td>
      <td>23454</td>
    </tr>
    <tr>
      <th>3</th>
      <td>美丽人生</td>
      <td>意大利</td>
      <td>55556</td>
    </tr>
    <tr>
      <th>4</th>
      <td>霸王别姬</td>
      <td>中国大陆</td>
      <td>6666</td>
    </tr>
    <tr>
      <th>5</th>
      <td>肖申克的救赎</td>
      <td>美国</td>
      <td>123344</td>
    </tr>
  </tbody>
</table>




现在，我们需要把df1和df2合并

我们发现，df2有票房数据，df1有评分等其他信息  
由于样本的顺序不一致，因此不能直接采取直接复制的方法


```python
pd.merge(df1, df2, how = "inner", on = "名字")
```



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>年代</th>
      <th>产地_x</th>
      <th>名字</th>
      <th>投票人数</th>
      <th>类型</th>
      <th>上映时间</th>
      <th>时长</th>
      <th>评分</th>
      <th>首映地点</th>
      <th>评分等级</th>
      <th>热门程度</th>
      <th>产地_y</th>
      <th>票房</th>
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
      <td>美国</td>
      <td>123344</td>
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
      <td>美国</td>
      <td>23454</td>
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
      <td>意大利</td>
      <td>55556</td>
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
      <td>美国</td>
      <td>333</td>
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
      <td>中国大陆</td>
      <td>6666</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2012</td>
      <td>美国</td>
      <td>泰坦尼克号</td>
      <td>157074</td>
      <td>剧情/爱情/灾难</td>
      <td>2012-04-10 00:00:00</td>
      <td>194</td>
      <td>9.4</td>
      <td>中国大陆</td>
      <td>A</td>
      <td>A</td>
      <td>美国</td>
      <td>444</td>
    </tr>
  </tbody>
</table>




由于两个数据集都存在产地，因此合并后会有两个产地信息



### 3.13.3 concat

将多个数据集进行批量合并

```python
df1 = df[:10]
df2 = df[100:110]
df3 = df[200:210]
dff = pd.concat([df1,df2,df3],axis = 0) #默认axis = 0，列拼接需要修改为1
dff
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
    <tr>
      <th>5</th>
      <td>2012</td>
      <td>美国</td>
      <td>泰坦尼克号</td>
      <td>157074</td>
      <td>剧情/爱情/灾难</td>
      <td>2012-04-10 00:00:00</td>
      <td>194</td>
      <td>9.4</td>
      <td>中国大陆</td>
      <td>A</td>
      <td>A</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1993</td>
      <td>美国</td>
      <td>辛德勒的名单</td>
      <td>306904</td>
      <td>剧情/历史/战争</td>
      <td>1993-11-30 00:00:00</td>
      <td>195</td>
      <td>9.4</td>
      <td>华盛顿首映</td>
      <td>A</td>
      <td>A</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1997</td>
      <td>日本</td>
      <td>新世纪福音战士剧场版：Air/真心为你 新世紀エヴァンゲリオン劇場版 Ai</td>
      <td>24355</td>
      <td>剧情/动作/科幻/动画/奇幻</td>
      <td>1997-07-19 00:00:00</td>
      <td>87</td>
      <td>9.4</td>
      <td>日本</td>
      <td>A</td>
      <td>A</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2013</td>
      <td>日本</td>
      <td>银魂完结篇：直到永远的万事屋 劇場版 銀魂 完結篇 万事屋よ</td>
      <td>21513</td>
      <td>剧情/动画</td>
      <td>2013-07-06 00:00:00</td>
      <td>110</td>
      <td>9.4</td>
      <td>日本</td>
      <td>A</td>
      <td>A</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1994</td>
      <td>法国</td>
      <td>这个杀手不太冷</td>
      <td>662552</td>
      <td>剧情/动作/犯罪</td>
      <td>1994-09-14 00:00:00</td>
      <td>133</td>
      <td>9.4</td>
      <td>法国</td>
      <td>A</td>
      <td>A</td>
    </tr>
    <tr>
      <th>100</th>
      <td>1993</td>
      <td>韩国</td>
      <td>101</td>
      <td>146</td>
      <td>喜剧/爱情</td>
      <td>1993-06-19 00:00:00</td>
      <td>112</td>
      <td>7.4</td>
      <td>韩国</td>
      <td>B</td>
      <td>D</td>
    </tr>
    <tr>
      <th>101</th>
      <td>1995</td>
      <td>英国</td>
      <td>10</td>
      <td>186</td>
      <td>喜剧</td>
      <td>1995-01-25 00:00:00</td>
      <td>101</td>
      <td>7.4</td>
      <td>美国</td>
      <td>B</td>
      <td>D</td>
    </tr>
    <tr>
      <th>102</th>
      <td>2013</td>
      <td>韩国</td>
      <td>素媛</td>
      <td>114819</td>
      <td>剧情/家庭</td>
      <td>2013-10-02 00:00:00</td>
      <td>123</td>
      <td>9.1</td>
      <td>韩国</td>
      <td>A</td>
      <td>A</td>
    </tr>
    <tr>
      <th>103</th>
      <td>2003</td>
      <td>美国</td>
      <td>101忠狗续集：伦敦</td>
      <td>924</td>
      <td>喜剧/动画/家庭</td>
      <td>2003-01-21 00:00:00</td>
      <td>70</td>
      <td>7.5</td>
      <td>美国</td>
      <td>B</td>
      <td>B</td>
    </tr>
    <tr>
      <th>104</th>
      <td>2000</td>
      <td>美国</td>
      <td>10</td>
      <td>9514</td>
      <td>喜剧/家庭</td>
      <td>2000-09-22 00:00:00</td>
      <td>100</td>
      <td>7.0</td>
      <td>美国</td>
      <td>C</td>
      <td>A</td>
    </tr>
    <tr>
      <th>105</th>
      <td>2013</td>
      <td>韩国</td>
      <td>10</td>
      <td>601</td>
      <td>剧情</td>
      <td>2014-04-24 00:00:00</td>
      <td>93</td>
      <td>7.2</td>
      <td>美国</td>
      <td>B</td>
      <td>C</td>
    </tr>
    <tr>
      <th>106</th>
      <td>2006</td>
      <td>美国</td>
      <td>10件或</td>
      <td>1770</td>
      <td>剧情/喜剧/爱情</td>
      <td>2006-12-01 00:00:00</td>
      <td>82</td>
      <td>7.7</td>
      <td>美国</td>
      <td>B</td>
      <td>B</td>
    </tr>
    <tr>
      <th>107</th>
      <td>2014</td>
      <td>美国</td>
      <td>10年</td>
      <td>1531</td>
      <td>喜剧/同性</td>
      <td>2015-06-02 00:00:00</td>
      <td>90</td>
      <td>6.9</td>
      <td>美国</td>
      <td>C</td>
      <td>B</td>
    </tr>
    <tr>
      <th>108</th>
      <td>2012</td>
      <td>日本</td>
      <td>11·25自决之日 三岛由纪夫与年轻人们 11・25自決の</td>
      <td>149</td>
      <td>剧情</td>
      <td>2012-06-02 00:00:00</td>
      <td>119</td>
      <td>5.6</td>
      <td>日本</td>
      <td>C</td>
      <td>D</td>
    </tr>
    <tr>
      <th>109</th>
      <td>1997</td>
      <td>美国</td>
      <td>泰坦尼克号</td>
      <td>535491</td>
      <td>剧情/爱情/灾难</td>
      <td>1998-04-03 00:00:00</td>
      <td>194</td>
      <td>9.1</td>
      <td>中国大陆</td>
      <td>A</td>
      <td>A</td>
    </tr>
    <tr>
      <th>200</th>
      <td>2014</td>
      <td>日本</td>
      <td>最完美的离婚 2014特别篇</td>
      <td>18478</td>
      <td>剧情/喜剧/爱情</td>
      <td>2014-02-08 00:00:00</td>
      <td>120</td>
      <td>9.1</td>
      <td>日本</td>
      <td>A</td>
      <td>A</td>
    </tr>
    <tr>
      <th>201</th>
      <td>2009</td>
      <td>日本</td>
      <td>2001夜物</td>
      <td>84</td>
      <td>剧情/动画</td>
      <td>2009-10-02 00:00:00</td>
      <td>80</td>
      <td>6.6</td>
      <td>美国</td>
      <td>C</td>
      <td>D</td>
    </tr>
    <tr>
      <th>202</th>
      <td>2009</td>
      <td>中国香港</td>
      <td>头七 頭</td>
      <td>7039</td>
      <td>恐怖</td>
      <td>2009-05-21 00:00:00</td>
      <td>60</td>
      <td>6.2</td>
      <td>美国</td>
      <td>C</td>
      <td>A</td>
    </tr>
    <tr>
      <th>203</th>
      <td>1896</td>
      <td>法国</td>
      <td>火车进站 L</td>
      <td>7001</td>
      <td>纪录片/短片</td>
      <td>1896-01-06</td>
      <td>60</td>
      <td>8.8</td>
      <td>法国</td>
      <td>B</td>
      <td>A</td>
    </tr>
    <tr>
      <th>204</th>
      <td>2009</td>
      <td>美国</td>
      <td>银行舞蹈</td>
      <td>6944</td>
      <td>短片</td>
      <td>1905-07-01 00:00:00</td>
      <td>60</td>
      <td>7.8</td>
      <td>美国</td>
      <td>B</td>
      <td>A</td>
    </tr>
    <tr>
      <th>205</th>
      <td>2003</td>
      <td>荷兰</td>
      <td>2003提雅</td>
      <td>48</td>
      <td>音乐</td>
      <td>2003-10-07 00:00:00</td>
      <td>200</td>
      <td>8.9</td>
      <td>美国</td>
      <td>B</td>
      <td>E</td>
    </tr>
    <tr>
      <th>206</th>
      <td>2012</td>
      <td>美国</td>
      <td>死亡飞车3：地狱烈</td>
      <td>6937</td>
      <td>动作</td>
      <td>2012-12-12 00:00:00</td>
      <td>60</td>
      <td>5.8</td>
      <td>美国</td>
      <td>C</td>
      <td>A</td>
    </tr>
    <tr>
      <th>207</th>
      <td>2012</td>
      <td>日本</td>
      <td>时光钟摆 振り</td>
      <td>6876</td>
      <td>剧情/动画/短片</td>
      <td>2012-03-20 00:00:00</td>
      <td>60</td>
      <td>8.7</td>
      <td>美国</td>
      <td>B</td>
      <td>A</td>
    </tr>
    <tr>
      <th>208</th>
      <td>2011</td>
      <td>中国香港</td>
      <td>你还可爱么 你還可愛</td>
      <td>6805</td>
      <td>短片</td>
      <td>2011-04-22 00:00:00</td>
      <td>60</td>
      <td>8.3</td>
      <td>美国</td>
      <td>B</td>
      <td>A</td>
    </tr>
    <tr>
      <th>209</th>
      <td>2002</td>
      <td>中国香港</td>
      <td>一碌蔗</td>
      <td>6799</td>
      <td>剧情/喜剧/爱情</td>
      <td>2002-09-19 00:00:00</td>
      <td>60</td>
      <td>6.7</td>
      <td>美国</td>
      <td>C</td>
      <td>A</td>
    </tr>
  </tbody>
</table>
