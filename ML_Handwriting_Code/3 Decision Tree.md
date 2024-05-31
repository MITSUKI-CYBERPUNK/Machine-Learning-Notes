# 3 Dicision Tree

## 3.1 数据清洗


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


### 3.1.1 缺失值处理


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



### 3.1.2 离群值处理


```python
data = ['forehead_width_cm', 'forehead_height_cm']

plt.subplots(figsize=(17, 7))
for i, col in enumerate(data):
    plt.subplot(1, 2, i + 1)
    sb.boxplot(train[col])
plt.show()
```

![output_9_0.png](https://aquazone.oss-cn-guangzhou.aliyuncs.com/output_9_0.png)

### 3.1.3 类型转换


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



### 3.1.4 训练前准备


```python
# 将特征X和目标变量y分离
X = train.drop('gender', axis=1)
y = train['gender']
```

## 3.2 决策树模型训练


```python
"""
# 计算信息熵
def entropy(labels):
    # np.unique 函数用于获取数组 labels 中唯一的元素，并返回每个唯一元素的出现次数
    _, counts = np.unique(labels, return_counts=True)  # _ 在这里起到占位符的作用，帮助我们忽略不需要的结果，只提取出需要的部分数据

    # 计算每个类别在数据集中出现的概率，即该类别样本数量除以总样本数量，得到每个类别的比例
    probs = counts / len(labels)

    # 对每个类别的概率值乘以其对数值（以2为底），然后求和，最后取负值（信息熵定义）
    return -np.sum(probs * np.log2(probs))


# 计算信息增益
def information_gain(data, feature_index, labels):
    # 计算总体的信息熵
    total_entropy = entropy(labels)

    # 根据特征划分数据集
    unique_values = np.unique(data[:, feature_index])  # 获取特征列 feature_index 中的唯一取值，表示该特征可能的取值情况
    weighted_entropy = 0
    for value in unique_values:
        subset_labels = labels[data[:, feature_index] == value]  # 根据当前取值 value 对数据进行划分，获取子集对应的标签
        # 计算当前子集的信息熵，并乘以子集所占比例（即子集样本数除以总样本数），然后累加到加权信息熵中
        weighted_entropy += len(subset_labels) / len(labels) * entropy(subset_labels)

    # 返回信息增益（信息增益 = 总体信息熵 - 加权信息熵之和）
    return total_entropy - weighted_entropy

"""
```


```python
# 计算基尼不纯度
def gini_impurity(labels):
    # 计算每个类别在数据集中出现的概率
    probabilities = np.bincount(labels) / len(labels)

    # 计算基尼不纯度
    return 1 - np.sum(probabilities ** 2)
```


```python
# 计算信息增益（使用基尼不纯度）
def information_gain_gini(data, feature_index, labels):
    # 计算总体的基尼不纯度
    total_gini = gini_impurity(labels)

    # 根据特征划分数据集
    unique_values = np.unique(data[:, feature_index])
    weighted_gini = 0
    for value in unique_values:
        subset_labels = labels[data[:, feature_index] == value]
        subset_size = len(subset_labels)
        # 计算当前子集的基尼不纯度，并乘以子集所占比例
        weighted_gini += subset_size / len(labels) * gini_impurity(subset_labels)

    # 返回信息增益（基尼不纯度减少量）
    return total_gini - weighted_gini
```


```python
# 选择最佳特征
def choose_best_feature(data, labels):
    num_features = data.shape[1]  # 获取数据集中特征的数量(列数)
    best_feature_index = -1
    best_info_gain = 0

    # 求取最佳增益与最佳特征
    for i in range(num_features):
        info_gain = information_gain_gini(data, i, labels)
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature_index = i

    return best_feature_index
```


```python
# 构建决策树
def build_decision_tree(data, labels, max_depth=5, min_samples=10):
    labels = np.array(labels)  # 将labels转换为NumPy数组
    # 检查标签是否属于同一类别,如果是,则返回该类别作为叶节点的预测结果(纯度拉满)
    if len(np.unique(labels)) == 1:
        return labels[0]

    # 检查剩余特征是否达到阈值
    if data.shape[1] == 0:
        return np.bincount(labels).argmax()

    # 检查是否达到最大深度或最小样本数
    if len(labels) < min_samples or max_depth == 0:
        return np.bincount(labels).argmax()

    # 选择最佳特征进行分裂
    best_feature_index = choose_best_feature(data, labels)
    best_feature = data[:, best_feature_index]

    # 字典 tree 来存储决策树的结构,最佳特征索引作为键
    tree = {best_feature_index: {}}

    # 遍历最佳特征列中的唯一取值
    for value in np.unique(best_feature):
        # 使用布尔掩码获取子集索引
        subset_mask = best_feature == value

        # 根据布尔掩码将数据集分割为子集
        subset_data = data[subset_mask]
        subset_labels = labels[subset_mask]

        # 递归地构建决策树,将子集数据和标签传递给下一层节点
        tree[best_feature_index][value] = build_decision_tree(subset_data, subset_labels, max_depth - 1, min_samples)

    return tree
```


```python
# 将标签y转换为整数类型
y = y.astype(int)

# 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 将 DataFrame 转换为 NumPy 数组
X_train = X_train.values
X_test = X_test.values

# 设置预剪枝参数
max_depth = 5
min_samples_split = 10

# 训练决策树
decision_tree = build_decision_tree(X_train, y_train, max_depth, min_samples_split)

def get_leaf_values(tree):
    leaf_values = []
    for subtree in tree.values():
        if isinstance(subtree, np.integer):
            leaf_values.append(subtree)
        else:
            leaf_values.extend(get_leaf_values(subtree))
    return leaf_values

def predict(tree, instance):
    # 如果当前节点是叶节点(整数值),直接返回该值作为预测结果
    if isinstance(tree, np.integer):
        return tree

    # 获取当前节点的最佳特征索引
    feature_index = list(tree.keys())[0]

    # 获取当前实例中最佳特征的值
    feature_value = instance[feature_index]

    # 如果特征值不存在于子节点中,返回当前节点下所有叶节点的多数类别
    if feature_value not in tree[feature_index]:
        leaf_values = []
        for subtree in tree[feature_index].values():
            if isinstance(subtree, np.integer):
                leaf_values.append(subtree)
            else:
                leaf_values.extend(get_leaf_values(subtree))
        return np.bincount(leaf_values).argmax()

    # 根据特征值递归地进入相应的子树
    sub_tree = tree[feature_index][feature_value]
    return predict(sub_tree, instance)

# 对测试集进行预测
y_pred = [predict(decision_tree, x) for x in X_test]
```



## 3.3 模型评估


```python
# 评估模型性能
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# 计算混淆矩阵
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# 计算精确率、召回率和F1分数
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
```

    Confusion Matrix:
    [[480  19]
     [ 18 484]]
    Precision: 0.9622
    Recall: 0.9641
    F1-score: 0.9632


由混淆矩阵，精确率，召回率，F1分数得知：模型在对性别进行分类时具有较高的准确性和可靠性



## 3.4 反思复盘

+ 对数学基础还需要进一步加强
+ 对连续值进行离散处理

