# 2 Logistic Regression

## 2.1 数据清洗


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


## 2.1.1 缺失值处理


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


```python
我们发现数据集中没有缺失值，因此无需处理缺失值
```



### 2.1.2 离群值处理


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
​    

根据箱线图，我们可知这两列没有离群值



### 2.1.3 独热编码处理


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



### 2.1.4 标准化处理


```python
# 标准化处理
train = (train - train.mean()) / train.std()
```



### 2.1.5 训练前准备


```python
# 将特征X和目标变量y分离
X = train.drop('gender', axis=1)
y = train['gender']

# 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 将 DataFrame 转换为 NumPy 数组
X_train = X_train.values
X_test = X_test.values
```



## 2.2 梯度下降模型训练


```python
# sigmoid 函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 逻辑回归类
class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iters=1000):
        self.learning_rate = learning_rate
        self.num_iters = num_iters
        self.weights = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)

        for _ in range(self.num_iters):
            linear_model = np.dot(X, self.weights)
            y_predicted = sigmoid(linear_model)

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            self.weights -= self.learning_rate * dw

    def predict(self, X):
        linear_model = np.dot(X, self.weights)
        y_predicted = sigmoid(linear_model)
        y_predicted_cls = [1 if y >= 0.5 else 0 for y in y_predicted]
        return np.array(y_predicted_cls)

# 创建逻辑回归模型实例
model = LogisticRegression()

# 训练模型
model.fit(X_train, y_train)

# 进行预测
y_pred = model.predict(X_test)

# 反标准化预测结果
y_pred_ = y_pred * ys + ym

# 格式化输出结果为整数
y_pred_ = y_pred_.astype(int)

# 统计预测值中1和0的数量
num_zeros = np.sum(y_pred_ == 0)
num_ones = np.sum(y_pred_ == 1)

print("预测值中0的数量：", num_zeros)
print("预测值中1的数量：", num_ones)
```

    预测值中0的数量： 489
    预测值中1的数量： 512



## 2.3 可视化预测结果


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

![output_21_0.png](https://aquazone.oss-cn-guangzhou.aliyuncs.com/output_21_0.png)

```python
我们可以得知，测试集中预测的男女比例相当
```



## 2.4 模型评估


```python
# 由于未知真实标签，我们在这里只能使用随机真实标签
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

true_labels = np.random.randint(2, size=len(X_test))

# 计算准确率
accuracy = accuracy_score(true_labels, y_pred)

# 计算精确率
precision = precision_score(true_labels, y_pred)

# 计算召回率
recall = recall_score(true_labels, y_pred)

# 计算F1值
f1 = f1_score(true_labels, y_pred)

print("准确率：", accuracy)
print("精确率：", precision)
print("召回率：", recall)
print("F1 值：", f1)
```

    准确率： 0.4965034965034965
    精确率： 0.498046875
    召回率： 0.5079681274900398
    F1 值： 0.5029585798816568


根据以上四个指标，我们可以推测模型并不理想，这可能与我们使用了随机真实标签有关



## 2.5 带正则化的梯度下降


```python
# sigmoid 函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 逻辑回归类
class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iters=1000, lambda_val=0.01):
        self.learning_rate = learning_rate
        self.num_iters = num_iters
        self.lambda_val = lambda_val
        self.weights = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)

        for _ in range(self.num_iters):
            linear_model = np.dot(X, self.weights)
            y_predicted = sigmoid(linear_model)

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y)) + (self.lambda_val / n_samples) * self.weights
            self.weights -= self.learning_rate * dw

    def predict(self, X):
        linear_model = np.dot(X, self.weights)
        y_predicted = sigmoid(linear_model)
        y_predicted_cls = [1 if y >= 0.5 else 0 for y in y_predicted]
        return np.array(y_predicted_cls)

# 创建逻辑回归模型实例
model = LogisticRegression(lambda_val=0.01)  # 添加 lambda_val 参数用于控制正则化强度

# 训练模型
model.fit(X_train, y_train)

# 进行预测
y_pred = model.predict(X_test)

# 反标准化预测结果
y_pred_ = y_pred * ys + ym

# 格式化输出结果为整数
y_pred_ = y_pred_.astype(int)

# 统计预测值中1和0的数量
num_zeros = np.sum(y_pred_ == 0)
num_ones = np.sum(y_pred_ == 1)

print("预测值中0的数量：", num_zeros)
print("预测值中1的数量：", num_ones)

```

    预测值中0的数量： 489
    预测值中1的数量： 512



### 模型评估


```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

true_labels = np.random.randint(2, size=len(X_test))

# 计算准确率
accuracy = accuracy_score(true_labels, y_pred)

# 计算精确率
precision = precision_score(true_labels, y_pred)

# 计算召回率
recall = recall_score(true_labels, y_pred)

# 计算F1值
f1 = f1_score(true_labels, y_pred)

print("准确率：", accuracy)
print("精确率：", precision)
print("召回率：", recall)
print("F1 值：", f1)
```

    准确率： 0.5014985014985015
    精确率： 0.49609375
    召回率： 0.5131313131313131
    F1 值： 0.5044687189672294

根据以上四个指标，我们可以推测模型并不理想，这可能与我们使用了随机真实标签有关

## 2.6 改进反思

+ 模型评估最好学习一下无需真实标签的方法，例如交叉验证等
+ 进一步优化模型，学习深层次的优化方法
