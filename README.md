# machine-learning

## 实践二

**约定**：
- $z^{(i)}$ 表示第 $i$ 层的**加权**输入，$a^{(i)}$ 为第 $i$ 层的经激活函数作用后的输出；
- 输入层为第 $0$ 层；
- 训练数据 $\text{feature}$ 为 $\vec{x}$，$\text{label}$ 为 $\vec{y}$.

给定的模型只有一层隐藏层，隐藏层和输出层分别使用两种激活函数 $\tanh(x)$ 和 $\text{softmax}(x)$，损失函数 $L$ 定义如下：

$$
L = -\sum_{j} y_j\log(a_j^{(2)})
$$

$\text{softmax}(z_i|\vec{z})$ 的定义为

$$
\text{softmax}(z_i|\vec{z})=\frac{e^{z_i}}{\sum_{k}e^{z_k}}
$$

分析其梯度，对于输出层：

$$
\delta_i = 
$$