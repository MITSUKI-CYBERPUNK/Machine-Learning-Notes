# 5 K-means

## 5.1 数据清洗


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
train=pd.read_csv(r'D:\RDC\wine-clustering.csv')
```


```python
# 查看数据集信息
train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 178 entries, 0 to 177
    Data columns (total 13 columns):
     #   Column                Non-Null Count  Dtype  
    ---  ------                --------------  -----  
     0   Alcohol               178 non-null    float64
     1   Malic_Acid            178 non-null    float64
     2   Ash                   178 non-null    float64
     3   Ash_Alcanity          178 non-null    float64
     4   Magnesium             178 non-null    int64  
     5   Total_Phenols         178 non-null    float64
     6   Flavanoids            178 non-null    float64
     7   Nonflavanoid_Phenols  178 non-null    float64
     8   Proanthocyanins       178 non-null    float64
     9   Color_Intensity       178 non-null    float64
     10  Hue                   178 non-null    float64
     11  OD280                 178 non-null    float64
     12  Proline               178 non-null    int64  
    dtypes: float64(11), int64(2)
    memory usage: 18.2 KB



```python
# 查看前五行
train.head()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Alcohol</th>
      <th>Malic_Acid</th>
      <th>Ash</th>
      <th>Ash_Alcanity</th>
      <th>Magnesium</th>
      <th>Total_Phenols</th>
      <th>Flavanoids</th>
      <th>Nonflavanoid_Phenols</th>
      <th>Proanthocyanins</th>
      <th>Color_Intensity</th>
      <th>Hue</th>
      <th>OD280</th>
      <th>Proline</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>14.23</td>
      <td>1.71</td>
      <td>2.43</td>
      <td>15.6</td>
      <td>127</td>
      <td>2.80</td>
      <td>3.06</td>
      <td>0.28</td>
      <td>2.29</td>
      <td>5.64</td>
      <td>1.04</td>
      <td>3.92</td>
      <td>1065</td>
    </tr>
    <tr>
      <th>1</th>
      <td>13.20</td>
      <td>1.78</td>
      <td>2.14</td>
      <td>11.2</td>
      <td>100</td>
      <td>2.65</td>
      <td>2.76</td>
      <td>0.26</td>
      <td>1.28</td>
      <td>4.38</td>
      <td>1.05</td>
      <td>3.40</td>
      <td>1050</td>
    </tr>
    <tr>
      <th>2</th>
      <td>13.16</td>
      <td>2.36</td>
      <td>2.67</td>
      <td>18.6</td>
      <td>101</td>
      <td>2.80</td>
      <td>3.24</td>
      <td>0.30</td>
      <td>2.81</td>
      <td>5.68</td>
      <td>1.03</td>
      <td>3.17</td>
      <td>1185</td>
    </tr>
    <tr>
      <th>3</th>
      <td>14.37</td>
      <td>1.95</td>
      <td>2.50</td>
      <td>16.8</td>
      <td>113</td>
      <td>3.85</td>
      <td>3.49</td>
      <td>0.24</td>
      <td>2.18</td>
      <td>7.80</td>
      <td>0.86</td>
      <td>3.45</td>
      <td>1480</td>
    </tr>
    <tr>
      <th>4</th>
      <td>13.24</td>
      <td>2.59</td>
      <td>2.87</td>
      <td>21.0</td>
      <td>118</td>
      <td>2.80</td>
      <td>2.69</td>
      <td>0.39</td>
      <td>1.82</td>
      <td>4.32</td>
      <td>1.04</td>
      <td>2.93</td>
      <td>735</td>
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
      <th>Alcohol</th>
      <th>Malic_Acid</th>
      <th>Ash</th>
      <th>Ash_Alcanity</th>
      <th>Magnesium</th>
      <th>Total_Phenols</th>
      <th>Flavanoids</th>
      <th>Nonflavanoid_Phenols</th>
      <th>Proanthocyanins</th>
      <th>Color_Intensity</th>
      <th>Hue</th>
      <th>OD280</th>
      <th>Proline</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>13.000618</td>
      <td>2.336348</td>
      <td>2.366517</td>
      <td>19.494944</td>
      <td>99.741573</td>
      <td>2.295112</td>
      <td>2.029270</td>
      <td>0.361854</td>
      <td>1.590899</td>
      <td>5.058090</td>
      <td>0.957449</td>
      <td>2.611685</td>
      <td>746.893258</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.811827</td>
      <td>1.117146</td>
      <td>0.274344</td>
      <td>3.339564</td>
      <td>14.282484</td>
      <td>0.625851</td>
      <td>0.998859</td>
      <td>0.124453</td>
      <td>0.572359</td>
      <td>2.318286</td>
      <td>0.228572</td>
      <td>0.709990</td>
      <td>314.907474</td>
    </tr>
    <tr>
      <th>min</th>
      <td>11.030000</td>
      <td>0.740000</td>
      <td>1.360000</td>
      <td>10.600000</td>
      <td>70.000000</td>
      <td>0.980000</td>
      <td>0.340000</td>
      <td>0.130000</td>
      <td>0.410000</td>
      <td>1.280000</td>
      <td>0.480000</td>
      <td>1.270000</td>
      <td>278.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>12.362500</td>
      <td>1.602500</td>
      <td>2.210000</td>
      <td>17.200000</td>
      <td>88.000000</td>
      <td>1.742500</td>
      <td>1.205000</td>
      <td>0.270000</td>
      <td>1.250000</td>
      <td>3.220000</td>
      <td>0.782500</td>
      <td>1.937500</td>
      <td>500.500000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>13.050000</td>
      <td>1.865000</td>
      <td>2.360000</td>
      <td>19.500000</td>
      <td>98.000000</td>
      <td>2.355000</td>
      <td>2.135000</td>
      <td>0.340000</td>
      <td>1.555000</td>
      <td>4.690000</td>
      <td>0.965000</td>
      <td>2.780000</td>
      <td>673.500000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>13.677500</td>
      <td>3.082500</td>
      <td>2.557500</td>
      <td>21.500000</td>
      <td>107.000000</td>
      <td>2.800000</td>
      <td>2.875000</td>
      <td>0.437500</td>
      <td>1.950000</td>
      <td>6.200000</td>
      <td>1.120000</td>
      <td>3.170000</td>
      <td>985.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>14.830000</td>
      <td>5.800000</td>
      <td>3.230000</td>
      <td>30.000000</td>
      <td>162.000000</td>
      <td>3.880000</td>
      <td>5.080000</td>
      <td>0.660000</td>
      <td>3.580000</td>
      <td>13.000000</td>
      <td>1.710000</td>
      <td>4.000000</td>
      <td>1680.000000</td>
    </tr>
  </tbody>
</table>




### 5.1.1 缺失值处理


```python
# 查看缺失值
train.isnull().sum()
```


    Alcohol                 0
    Malic_Acid              0
    Ash                     0
    Ash_Alcanity            0
    Magnesium               0
    Total_Phenols           0
    Flavanoids              0
    Nonflavanoid_Phenols    0
    Proanthocyanins         0
    Color_Intensity         0
    Hue                     0
    OD280                   0
    Proline                 0
    dtype: int64


```python
无缺失值，因此无需处理
```



## 5.2 模型训练


```python
# 提取特征
X = train.values

# 数据标准化处理
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-means函数
def k_means(X, K, max_iters=100):
    # 随机初始化K个中心点
    centroids = X[np.random.choice(X.shape[0], K, replace=False)]
    
    for _ in range(max_iters):
        # 计算每个样本到中心点的距离
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        
        # 将每个样本分配给最近的中心点
        labels = np.argmin(distances, axis=0)
        
        # 更新中心点位置为各簇样本的平均值
        new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(K)])
        
        # 如果中心点不再更新，则结束迭代
        if np.all(centroids == new_centroids):
            break
            
        centroids = new_centroids
    
    return centroids, labels

# 调用K-means函数进行聚类
K = 3  # 三种葡萄酒
centroids, labels = k_means(X, K)

# 将聚类结果添加到原始数据中
train['Cluster'] = labels
```



## 5.3 模型评估


```python
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# 计算评估参数
silhouette = silhouette_score(X, labels)
calinski_harabasz = calinski_harabasz_score(X, labels)
davies_bouldin = davies_bouldin_score(X, labels)

print("轮廓系数:", silhouette)
print("CH指数:", calinski_harabasz)
print("DBI指数:", davies_bouldin)
```

    轮廓系数: 0.30001078325847186
    CH指数: 78.67404420628469
    DBI指数: 1.3409232762585293


CH指数较大，DBI指数较小，表示簇内的样本紧密，簇间的样本分散，但轮廓系数表现并不好，需要进一步优化



## 5.4 优化反思

+ 特征较多，未能实现二维的可视化，应该继续学习降维部分
+ 轮廓系数不理想，继续优化模型
