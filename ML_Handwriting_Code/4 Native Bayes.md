# 4 Native Bayes

## 4.1 数据清洗


```python
# 导入依赖库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 忽略警告
import warnings
warnings.filterwarnings('ignore')

#导入数据
train=pd.read_csv(r'D:\RDC\gender_classification.csv')
```


```python
# 查看数据集信息
train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 5001 entries, 0 to 5000
    Data columns (total 8 columns):
     #   Column                     Non-Null Count  Dtype  
    ---  ------                     --------------  -----  
     0   long_hair                  5001 non-null   int64  
     1   forehead_width_cm          5001 non-null   float64
     2   forehead_height_cm         5001 non-null   float64
     3   nose_wide                  5001 non-null   int64  
     4   nose_long                  5001 non-null   int64  
     5   lips_thin                  5001 non-null   int64  
     6   distance_nose_to_lip_long  5001 non-null   int64  
     7   gender                     5001 non-null   object 
    dtypes: float64(2), int64(5), object(1)
    memory usage: 312.7+ KB



```python
# 查看前五行
train.head()
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>long_hair</th>
      <th>forehead_width_cm</th>
      <th>forehead_height_cm</th>
      <th>nose_wide</th>
      <th>nose_long</th>
      <th>lips_thin</th>
      <th>distance_nose_to_lip_long</th>
      <th>gender</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>11.8</td>
      <td>6.1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>14.0</td>
      <td>5.4</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>11.8</td>
      <td>6.3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>14.4</td>
      <td>6.1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>13.5</td>
      <td>5.9</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Female</td>
    </tr>
  </tbody>
</table>





```python
# 描述性统计
train.describe()
```



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>long_hair</th>
      <th>forehead_width_cm</th>
      <th>forehead_height_cm</th>
      <th>nose_wide</th>
      <th>nose_long</th>
      <th>lips_thin</th>
      <th>distance_nose_to_lip_long</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>5001.000000</td>
      <td>5001.000000</td>
      <td>5001.000000</td>
      <td>5001.000000</td>
      <td>5001.000000</td>
      <td>5001.000000</td>
      <td>5001.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.869626</td>
      <td>13.181484</td>
      <td>5.946311</td>
      <td>0.493901</td>
      <td>0.507898</td>
      <td>0.493101</td>
      <td>0.498900</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.336748</td>
      <td>1.107128</td>
      <td>0.541268</td>
      <td>0.500013</td>
      <td>0.499988</td>
      <td>0.500002</td>
      <td>0.500049</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>11.400000</td>
      <td>5.100000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.000000</td>
      <td>12.200000</td>
      <td>5.500000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.000000</td>
      <td>13.100000</td>
      <td>5.900000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.000000</td>
      <td>14.000000</td>
      <td>6.400000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>15.500000</td>
      <td>7.100000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>




### 4.1.1 缺失值处理


```python
# 查看缺失值
train.isnull().sum()
```


    long_hair                    0
    forehead_width_cm            0
    forehead_height_cm           0
    nose_wide                    0
    nose_long                    0
    lips_thin                    0
    distance_nose_to_lip_long    0
    gender                       0
    dtype: int64

我们发现数据集中没有缺失值，因此无需处理缺失值



### 4.1.2 离群值处理


```python
# 画箱线图
data = ['forehead_width_cm', 'forehead_height_cm']

plt.subplots(figsize=(17, 7))
for i, col in enumerate(data):
    plt.subplot(1, 2, i + 1)
    sb.boxplot(train[col])
plt.show()
```

![output_10_0.png](https://aquazone.oss-cn-guangzhou.aliyuncs.com/output_10_0.png)

根据箱线图，我们可知这两列没有离群值



### 4.1.3 类型转换


```python
# 强制类型转换
columns_to_convert = ['gender']

train['gender'] = train['gender'].map({'Female': 1, 'Male': 0})

# 查看转换后的数据类型
print(train.dtypes)

ym = train['gender'].mean()

ys = train['gender'].std()
```

    long_hair                      int64
    forehead_width_cm            float64
    forehead_height_cm           float64
    nose_wide                      int64
    nose_long                      int64
    lips_thin                      int64
    distance_nose_to_lip_long      int64
    gender                         int64
    dtype: object



### 4.1.4 训练前准备


```python
# 将特征X和目标变量y分离
X = train.drop('gender', axis=1)
y = train['gender']

# 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

```



## 4.2 模型构建与训练


```python
class NaiveBayes:
    def __init__(self):
        self.class_p = {} # 存储每个类的概率
        self.feature_p = {} # 存储每个类中每个特征的概率

    # 训练模型函数
    def fit(self, X, y):
        self.class_p = {c: np.mean(y == c) for c in np.unique(y)}

        # 对每个类别c计算该类别在y中的占比，并将结果存储在self.class_p字典
        for c in self.class_p:
            self.feature_p[c] = {}
            for feature in X.columns:
                feature_p = {}
                for value in np.unique(X[feature]):
                    # 计算取值value在类c中的概率（计算布尔数组的均值，就是求True的概率）
                    p = np.mean(X[feature][y == c] == value)
                    feature_p[value] = p # 键为特征取值value，值为概率p
                self.feature_p[c][feature] = feature_p # 将特征feature的每个取值对应的概率字典feature_p存储在self.feature_p[c]中

    # 预测函数
    def predict(self, X):
        predictions = []
        for i in range(X.shape[0]):
            max_p = -1
            pred_class = None
            for c in self.class_p:
                p = self.class_p[c]# 类c的先验概率，我们先前已经求得
                for feature in X.columns:
                    # 将样本i的特征值作为键，从字典self.feature_p[c][feature]中获取相应的条件概率，并将其乘以当前概率prob
                    p *= self.feature_p[c][feature][X.iloc[i][feature]]# 条件概率乘以先验概率等于后验概率
                # 取最大概率，即贝叶斯原理
                if p > max_p:
                    max_p = p
                    pred_class = c
            predictions.append(pred_class)
        return predictions

# 初始化朴素贝叶斯分类器
model = NaiveBayes()

# 模型训练
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)


# 统计预测值中1和0的数量
unique, counts = np.unique(y_pred, return_counts=True)
counts_dict = dict(zip(unique, counts))
num_zeros = counts_dict.get(0, 0)
num_ones = counts_dict.get(1, 0)

print("预测值中0的数量：", num_zeros)
print("预测值中1的数量：", num_ones)
```

    预测值中0的数量： 478
    预测值中1的数量： 523



## 4.3 可视化预测结果


```python
# 计算总数
total = num_zeros + num_ones

# 创建带有数目和比例的label
labels = [f'Male\n{num_zeros} ({num_zeros/total*100:.1f}%)', f'Female\n{num_ones} ({num_ones/total*100:.1f}%)']
colors = ['blue', 'orange']

plt.pie([num_zeros, num_ones], labels=labels, colors=colors, autopct='', startangle=140, explode=[0, 0.1])
plt.axis('equal')  

plt.title('Predicted Class Distribution')

plt.show()
```

![output_19_0.png](https://aquazone.oss-cn-guangzhou.aliyuncs.com/output_19_0.png)

根据数据以及饼图可得知，与逻辑回归以及决策树模型的预测结果区别不大



## 4.4 模型评估


```python
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"正确率为：{accuracy:.2f}")
print(f"精确度为：{precision:.2f}")
print(f"召回率为：{recall:.2f}")
print(f"f1值为：{f1:.2f}")
```

    正确率为：0.97
    精确度为：0.95
    召回率为：0.99
    f1值为：0.97


根据评估指标，模型整体表现良好。它的正确率很高，且在保持较高的精确度的同时成功识别了大部分的正例，整体性能较好。



## 4.5  反思复盘

+ 数学理解决定高度，强化概率论内容
+ 模型过于简单，可能会出现异常
+ 深入学习其他类别的贝叶斯模型
