# 5 线性代数回顾Linear Algebra

具体见：[[线性代数公式概览]]
## 5.1 矩阵和向量

这个是4×2矩阵，即4行2列，如$m$为行，$n$为列，那么$m×n$即4×2

![image-20240130154746851](https://aquazone.oss-cn-guangzhou.aliyuncs.com/image-20240130154746851.png)

矩阵的维数即行数×列数

矩阵元素（矩阵项）：$A=\left[ \begin{matrix}   1402 & 191  \\   1371 & 821  \\   949 & 1437  \\   147 & 1448  \\\end{matrix} \right]$

$A_{ij}$指第$i$行，第$j$列的元素。

向量是一种特殊的矩阵，讲义中的向量一般都是列向量，如：

$y=\left[ \begin{matrix}   {460}  \\   {232}  \\   {315}  \\   {178}  \\\end{matrix} \right]$

为四维列向量（4×1）。

如下图为1索引向量和0索引向量，左图为1索引向量，右图为0索引向量，一般我们用1索引向量。

$y=\left[ \begin{matrix}   {{y}_{1}}  \\   {{y}_{2}}  \\   {{y}_{3}}  \\   {{y}_{4}}  \\\end{matrix} \right]$，$y=\left[ \begin{matrix}   {{y}_{0}}  \\   {{y}_{1}}  \\   {{y}_{2}}  \\   {{y}_{3}}  \\\end{matrix} \right]$



## 5.2 加法和标量乘法

矩阵的加法：行列数相等的可以加。

例：

![image-20240130160219811](https://aquazone.oss-cn-guangzhou.aliyuncs.com/image-20240130160219811.png)

矩阵的乘法：每个元素都要乘

![image-20240130160227072](https://aquazone.oss-cn-guangzhou.aliyuncs.com/image-20240130160227072.png)

组合算法也类似。



## 5.3 矩阵向量乘法

矩阵和向量的乘法如图：$m×n$的矩阵乘以$n×1$的向量，得到的是$m×1$的向量

![image-20240130160311995](https://aquazone.oss-cn-guangzhou.aliyuncs.com/image-20240130160311995.png)

算法举例：

![image-20240130160323054](https://aquazone.oss-cn-guangzhou.aliyuncs.com/image-20240130160323054.png)



## 5.4 矩阵乘法

矩阵乘法：

$m×n$矩阵乘以$n×o$矩阵，变成$m×o$矩阵。

如果这样说不好理解的话就举一个例子来说明一下，比如说现在有两个矩阵$A$和$B$，那么它们的乘积就可以表示为图中所示的形式。

![image-20240130160338943](https://aquazone.oss-cn-guangzhou.aliyuncs.com/image-20240130160338943.png)

![image-20240130160347905](https://aquazone.oss-cn-guangzhou.aliyuncs.com/image-20240130160347905.png)



## 5.5  矩阵乘法的性质

矩阵乘法的性质：

矩阵的乘法不满足交换律：$A×B≠B×A$

矩阵的乘法满足结合律。即：$A×(B×C)=(A×B)×C$

单位矩阵：在矩阵的乘法中，有一种矩阵起着特殊的作用，如同数的乘法中的1,我们称这种矩阵为单位矩阵．它是个方阵，一般用 $I$ 或者 $E$ 表示，本讲义都用 $I$ 代表单位矩阵，从左上角到右下角的对角线（称为主对角线）上的元素均为1以外全都为0。如：

$A{{A}^{-1}}={{A}^{-1}}A=I$

对于单位矩阵，有$AI=IA=A$



## 5.6 逆、转置

矩阵的逆：如矩阵$A$是一个$m×m$矩阵（方阵），如果有逆矩阵，则：$A{{A}^{-1}}={{A}^{-1}}A=I$

矩阵的转置：设$A$为$m×n$阶矩阵（即$m$行$n$列），第$i $行$j $列的元素是$a(i,j)$，即：$A=a(i,j)$

定义$A$的转置为这样一个$n×m$阶矩阵$B$，满足$B=a(j,i)$，即 $b (i,j)=a(j,i)$（$B$的第$i$行第$j$列元素是$A$的第$j$行第$i$列元素），记${{A}^{T}}=B$。(有些书记为A'=B）

直观来看，将$A$的所有元素绕着一条从第1行第1列元素出发的右下方45度的射线作镜面反转，即得到$A$的转置。

例：

${{\left| \begin{matrix}   a& b  \\   c& d  \\   e& f  \\\end{matrix} \right|}^{T}}=\left|\begin{matrix}   a& c & e  \\   b& d & f  \\\end{matrix} \right|$

矩阵的转置基本性质:

$ {{\left( A\pm B \right)}^{T}}={{A}^{T}}\pm {{B}^{T}} $
${{\left( A\times B \right)}^{T}}={{B}^{T}}\times {{A}^{T}}$
${{\left( {{A}^{T}} \right)}^{T}}=A $
${{\left( KA \right)}^{T}}=K{{A}^{T}} $

**matlab**中矩阵转置：直接打一撇，`x=y'`