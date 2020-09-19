
## 熵
概率分布$P(x)$的熵定义为
$$
H(P) = -\sum_i P(x_i)\log(P(x_i)) = \mathbb{E}_{x\thicksim P(x)}[-\log{P(x)}]
$$

**熵恒大于等于0，且$P(x)$为均匀分布时，熵最大**

证:
令$\phi(x)=-x\log(x),\lambda_i = \frac{1}{n},x_i=P(A_i)$,


由Janson不等式
$$
\sum_i\lambda_i \phi(x_i)\leq \phi(\sum_i\lambda_i x_i)
$$
有
$$
-\frac{1}{n}\sum_i P(A_i)\log P(A_i) \leq -\sum_i\frac{1}{n}P(A_i)\log(\sum_i\frac{1}{n}P(A_i))\\
=-\frac{1}{n}\sum_iP(A_i)\log(\frac{1}{n}\sum_iP(A_i))\\
=-\frac{1}{n}\log\frac{1}{n}
=\frac{1}{n}\log{n}\\
$$
$\Rightarrow$
$$
H(P) \leq \log n =-\sum_i\frac{1}{n}\log\frac{1}{n}=H(\frac{1}{n},\cdots,\frac{1}{n})
$$


## 交叉熵
概率分布$p(x),q(x)$的交叉熵为:
$$
H(p,q) = -\sum_i p(x)\log q(x)\\
=\mathbb{E}_{x\thicksim p(x)}[-\log q(x)]
$$
交叉熵可以衡量两个概率分布差异，差异越大交叉熵越大，

**当$p=q$时，交叉熵最小。**


证：
$$
-\sum_i p(x_i)\log q(x_i)-(-\sum_ip(x_i)\log p(x_i)) \\
=-\sum_i p(x_i)\log\frac{q(x_i)}{p(x_i)}
$$
取$\phi(x)=-\log(x),\lambda_i=p(x_i),x_i=q(x_i)$.
则由Janson不等式
$$
\phi(\sum_i\lambda_i x_i)\leq\sum_i\lambda_i \phi(x_i)
$$
有
$$
-\sum_i p(x_i)\log\frac{q(x_i)}{p(x_i)} \geq-\log\sum_i p(x_i)\frac{q(x_i)}{p(x_i)}\\
        =-\log\sum_i q(x_i) = 0
$$
故
$$
H(p,q) = -\sum_ip(x_i)\log q(x_i) \geq -\sum_i p(x_i)\log p(x_i)=H(p)
$$

## KL散度

概率分布$p,q$的KL散度为:
$$
KL(p(x)||q(x))=\sum_i p(x_i)\log\frac{p(x_i)}{q(x_i)}
$$

KL散度,一般用来衡量两个概率分布之间的“距离”。

**$KL(p(x)||(q(x)))\geq 0$**,当$p=q$时，等号成立。

证：
$$
KL(p(x)||q(x)) = \sum_ip(x_i)\log p(x_i)-\sum_ip(x_i)\log q(x_i)\\
                 = -H(p)+H(p,q)\geq 0.
$$

