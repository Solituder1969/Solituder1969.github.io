---
title: 10-Triple Generative Adversarial Networks
published: 2023-11-15
description: 'CGAN在训练时存在过渡依赖标签数据的问题，同时在分类结果和判别结果的期望上存在矛盾困境。Triple-GAN通过引入分类器，实现了这两个矛盾的解耦，并设计全新的训练流程，加强了模型训练的收敛性。'
tags: [DeepLearning model, GANs]
category: '论文研读'
draft: false 
---
# 一、问题分析

        在处理监督或者半监督学习的任务时，GAN 遇到了困境。一方面，我们希望鉴别器 D 应该无法分辨数据集的真伪，体现的是较高的信息熵值；而在另一方面，我们往往又希望分类的结果尽可能地真实，体现的是较低的信息熵值。这种矛盾使得 GAN 在处理类似的任务时，往往会陷入困境。一种改进的方式是使用条件概率，通过组成“标签-数据”对的方式，来进行训练。这种方式被称为 CGAN。由于训练的记录已经由条件概率分类好了，所以 D 在进行鉴别的时候自然不用概率熵高还是熵低的矛盾了。但是面对存在标签缺失的数据集，CGAN的训练效果就往往不佳了。

        既然分类任务需要结果呈现较低的信息熵值，训练结果又期望获得较高的信息熵值，那么理论上应该可以将这两者剥离开来。于是 Triple-GAN 应运而生。它包含了以下三个主体：

- Generator       ：生成器，接收一个真实的标签，生成一个伪数据

- Classifier         ：分类器，接收一个真实的数据，生成一个伪标签

- Discriminator  ：鉴别器，判定 <数据，标签> 对是否是真实的

# 二、问题定义

        假设 P(x) 和 P(y) 都是易得的。我们可以给出如下三条推理：

- 首先从 P(x) 中采样一个数据，获得一个 x，分类器 C 在条件概率 $P_c(y|x)$ 下生成一个伪标签 y。因此，组合概率 $P_c(x,y)=P(x)P_c(y|x)$

- 然后从 P(y) 中采样一个标签，获得一个 y，生成器 G 在条件概率 $P_g(x|y)$ 下生成一个伪数据 x。因此，组合概率 $P_g(x,y)=P(y)P_g(x|y)$

- 需要引起注意的是，生成器的生成的 x 是由给定标签 y后，隐空间变量 z 得到的。也就是说，$x=G(y,z), z\sim P(z)$，这里的 Z 遵循一个简单的分布，例如正态分布。

        模仿 GAN 的博弈函数，triple-GAN 的博弈函数可以如下表示：

$$
\mathop{min}\limits_{C,G} \;

\mathop{max}\limits_{D} \;

E_{P(x,y)}[log(D(x,y)]+\alpha

E_{P_c(x,y)}[log(1-D(x,y))]+(1-\alpha)E_{P_g(x,y)}[log(1-D(G(y,z),y))],\;\alpha\in(0,1)

\;\;\;\;\;\;E(1)

$$

        相比起 GAN 的初始版本，这里一共为3项。不难发现：

- 对于 D 来说，它希望能够尽量更精确地识别真伪数据，也就是在 P(x,y) 分布下，D 应该越大越好；在其余分布下，D 应该越小越好。反馈到上式里，就是 D 希望整体的结果最大化；

- 对于 C 和 G 来说，它们当然更喜欢自己生成的数据对不被识别出来，最好可以被判别为真。体现到式子里，就是在其他分布下，D 应该越大越好，从而使得整体的值趋小化了。

        但引入了新的感知器后，又带来了几个严重的问题。最主要的是训练的稳定性问题：

- 鉴别器 D 的收敛性是什么样的？是否存在唯一的收敛结果？

- G 和 C 的训练是否对抗？在全局最小值上是否会出现多个极值点？

# 三、相关的理论研究

## 3.1 D 的收敛性质

        不妨记 E(1) 为 U(C, D, G)。将其写成莱布尼茨积分式的形式：

$$
U(C,G,D)=\int\int P(x,y)logD(x,y)dxdy \;+

\alpha\int\int P_c(x,y)log(1-D(x,y))dxdy \;+(1-\alpha)

\int\int P_g(x,y)log(1-D(x,y))dxdy

$$

        令：

$$
P_\alpha(x,y)=\alpha P_c(x,y)+(1-\alpha)P_g(x,y)

$$

        则：

$$
U(C,G,D)=\int\int P(x,y)logD(x,y)dxdy \;+\int\int P_\alpha(x,y)log(1-D(x,y))dxdy

$$

        D 的训练结果是使得 U 取得最大值。这里假设log底数是自然底数，令 U 对 D 求偏导，有：

$$
\frac{\partial \;\int\int P(x,y)logD(x,y)dxdy}{\partial \; D(x,y)}=

\int\int\frac{ P(x,y)dxdy}{D(x,y)}\\



\frac{\partial \;\int\int P_\alpha(x,y)log(1-D(x,y))dxdy}{\partial \; D(x,y)}=

\int\int\frac{ -P_\alpha(x,y)dxdy}{1-D(x,y)}

$$

        求极值，导数结果为0，有：

$$
\frac{P(x,y)}{D(x,y)}-\frac{P_\alpha(x,y)}{1-D(x,y)}=0

$$

        解上式，可以得到：

$$
D(x,y)=\frac{P(x,y)}{P(x,y)+P_\alpha(x,y)}

$$

        这个值是唯一的，也就是说，D 的训练使得 U 最大后，D 的取值也就收敛了下来。

## 3.2 $P_c$ 和 $P_g$ 的关系

        我们一般希望 D(x,y) 最终为 1/2。从而有：

$$
P_\alpha(x,y)=P(x,y) \;\;\;\; i.e.

\;\;\;\;

\alpha P_c(x,y)+(1-\alpha)P_g(x,y)=P(x,y)

$$

        可以发现，当 $P_c(x,y)$ 或者  $P_\alpha (x,y)$  接近 $P(x,y)$ 时，另一方也会接近  $P(x,y)$。也就是说，两者不存在竞争的关系。

        我们还可以这样证明，将 D 带入 U 中，原式可以改写成：

$$
V(C,G)=\int\int P(x,y)log\frac{P(x,y)}{P(x,y)+P_\alpha(x,y)}dxdy\;+\int\int P_\alpha(x,y)log\frac{P_\alpha(x,y)}{P(x,y)+P_\alpha(x,y)}dxdy

$$

        训练到理想状况下，V(C,G) 应该取 min 值。观察到这里写成 JS 散度的格式：

$$
V(C,G)=-log4+2D_{JS}(P(x,y)||P_\alpha(x,y))

$$

        JS 散度具备非负性和对称性。当且仅当 $P(x,y)=P_\alpha(x,y)$ 时，JS散度取极值。从而证明了：

$$
V(C,G)\; achieve\;its\;minimum \; value,if\;and\;only\;if\;:P(x,y)=P_\alpha(x,y)=\alpha P_c(x,y)+(1-\alpha)P_g(x,y)

$$

## 3.3 $P_c$ 和 $P_g$ 的训练思路

        先证明一个引理，给定 $P(x,y) = P_\alpha(x,y)$ , 那么它们的边际分布都相等，也即是说：

$$
P(x)=P_g(x)=P_c(x) \;\; \& \;\;

P(y)=P_g(y)=P_c(y)

$$

        【证明】：

        两边对 x 积分：

$$
\int P(x,y)dx = (1-\alpha)\int P_g(x,y)dx+\alpha \int P_c(x,y)dx

$$

        在 “二、问题定义”部分，我们给出了如下的前提：

$$
P_c(x,y)=P(x)P_c(y|x)\\

 P_g(x,y)=P(y)P_g(x|y)

$$

        注意到：

$$
\int P_g(x|y)dx = 1

$$

        所以：

$$
P(y)=\int P(x,y)dx =(1-\alpha)P_(y)+\alpha P_c(y),\;\; i.e. \;\; P(y)=P_c(y)

$$

        代入替换式之前的原式中，可以得到：

$$
P(y)=P_c(y)=P_g(y)

$$

        同理，可以得到关于 x 的分布的类似结论：

$$
P(x)=P_g(x)=P_c(x)

$$

        证毕！

---

        我们最终希望 $P_c(x,y)$ 和 $P_g(x,y)$ 能够符合 $P(x,y)$的分布。这里使用 KL 散度来进行限制。考虑最小化如下两个 DL 散度值，以此来优化 $P_c$ 和  $P_g$  的分布：

$$
D_{KL}(P(x,y)||P_c(x,y))\;\;\&\;\; D_{KL}(P_g(x,y)||P_c(x,y))\\

let：R_c=E_{P(x,y)}[-logP_c(y|x)]\;\; \& \;\; R_p=E_{P_g(x,y)}[-logP_c(y|x)]

$$

        接下来证明，优化 $R_c$ ，效果上等于优化 $P_c$ 对 P 的 DL 散度；优化 $R_p$， 效果上等于优化 $P_c$ 对 $P_g$ 的 DL 散度。

        首先是 $R_c$，我们可以利用条件概率来进行改写。这里面用到了之前证明的边际分布相等的结论：

$$
\begin{align*}

R_c &= E_{P(x,y)}[-logP_c(y|x)]\\

&= E_{P(x,y)}[log(\frac{P_c(x)}{P_c(x,y)})]\\

&= E_{P(x,y)}[log(\frac{P(x,y)}{P_c(x,y)}·\frac{P_c(x)}{P(x,y)})]\\

&= D_{KL}(P(x,y)||P_c(x,y))-E_{P(x,y)}[logP(y|x)]

\end{align*}

$$

        第二项是一个和真实分布相关的值，可以认为是常数值。从而证明了，优化 R_C，就是在优化 $P(x,y)$  和 $P_c(x,y)$ 的 KL 散度。

        然后是 $R_p$，我们可以进行如下的推导：

$$
\begin{align*}

R_p &= E_{P_g(x,y)}[-logP_c(y|x)]\\

&= \int\int P_g(x,y)log\frac{P(x)}{P_c(x,y)}dxdy\\

&= \int\int P_g(x,y)log\frac{P(x)·P_g(x,y)}{P_c(x,y)·P_g(x)P_g(y|x)}dxdy\\

&= \int\int P_g(x,y)log\frac{P_g(x,y)}{P_c(x,y)}dxdy-\int P_g(x)log\frac{P_g(x)}{P_c(x)}dx+\int\int P_g(x,y)log\frac{1}{P_g(y|x)}dxdy\\

&=D_{KL}(P_g(x,y)||P_c(x,y))+E_{P_g(x,y)}[H(P_g(x,y))]-D_{kl}(P_g(x)||p(x))

\end{align*}

$$

        注意到后面两项是和生成器的参数 θ 相关的值。如果只是优化 C 的参数的话，那么可以认为优化 $P_g(x,y)$ 和 $P_c(x,y)$ 的 KL 散度，就是在优化 $R_p$。

## 3.4 正则化

        为了防止训练 C 的时候出现过拟合，需要添加正则化的惩罚项。这里最关键的一个问题在于，添加正则化的表达式后，是否会破坏原分布的平衡？

        观察 U 的表达式，对于 D 而言，结果出现在

$$
D(x,y)=\frac{P(x,y)}{P(x,y)+P_\alpha(x,y)}

$$

        上，这个结论不会改变（求解的过程是对 D 的求导，不含 D 的项都被去掉了）。

        D 的平衡位置不变，那么理想情况下，$P$ 和 $P_\alpha$ 的值仍旧应该相等。那么：

$$
\alpha P_c(x,y)+(1-\alpha)P_g(x,y)=P(x,y)

$$

        这个式子不会改变。最终的收敛结果为：

$$
P(x,y)=P_g(x,y)=P_c(x,y)

$$

        时，不论后面再添加什么样的正则式，KL散度的结果都将趋向于0。也就是说，最终的理想情况入手考虑，正则式的添加不会影响最终结果的平衡点。

# 四、流程梳理

        在第三步的过程中，我们证明了一系列的结论，确保了 Triple-GAN 的可用性：

- 引入 C 后，D 仍旧是收敛的，且收敛结果唯一，为：$D(x,y)=\frac{P(x,y)}{P(x,y)+P_\alpha(x,y)}$

- 达成最终的理想训练结果，应该具备：$\alpha P_c(x,y)+(1-\alpha)P_g(x,y)=P(x,y)$。这个结论表明，不添加额外的损失函数，G 和 C 有可能并不会收敛到近似真实的分布上。同时，该式子也表明，当 $P_c$ 或者 $P_g$ 向真实分布收敛时，另一方也会跟着收敛， C 和 G 不存在竞争关系，这降低了训练的难度。

- 为了让最终 C  和 G 都收敛到近似真实的分布上，所以添加了两个 KL 散度的约束，并证明了优化 $R_c$ 和 $R_p$，效果上和优化 KL 散度等效。

- 最后为了防止过拟合，需要添加正则化。同时证明了，正则化不会影响全局平衡点的收敛。

- 最终的博弈函数可以写为：

$$
\mathop{min}\limits_{C,G} \;

\mathop{max}\limits_{D} \;

E_{P(x,y)}[log(D(x,y)]+\alpha

E_{P_c(x,y)}[log(1-D(x,y))]+(1-\alpha)E_{P_g(x,y)}[log(1-D(G(y,z),y))]+R_c+\alpha_pR_p,\;\alpha\in(0,1)

\;\;\;\;\;\;E(2)

$$

# 五、算法介绍

        最后我们可以给出如下的 triple-GAN 伪代码：

---

Algorithm 1 Minibatch stochastic gradient descent training of Triple-GAN

---

对于每一次更新的步骤：

- 采样，包括：

    - 从生成器 G 生成的分布 $P_g(x,y)$ 中采样一批容量为 $m_g$ 的配对 $(x_g, y_g)$

    - 从分类器 C 生成的分布 $P_c(x,y)$ 中采样一批容量为 $m_c$ 的配对 $(x_c, y_c)$

    - 从真实的分布 $P(x,y)$ 中采样一批容量为 $m_d$  的配对 $(x_g, y_g)$

- 采用 SGD 更新 D：

    $$

    \nabla_{\theta_d}[\frac{1}{m_d}(\sum_{(x_d,y_d)}logD(x_d,y_d))+\frac{\alpha}{m_c}\sum_{(x_c,y_c)}log(1-D(x_c,y_c))+\frac{1-\alpha}{m_g}\sum_{(x_g,y_g)}log(1-D(x_g,y_g))]

    $$

- 计算无偏估计量 $R_c$ 和 $R_p$：

$$
R_c=-\frac{1}{m_d}\sum_{(x_d,y_d)}log(P_c(y_d|x_d))\;\;\&\;\;R_p=-\frac{1}{m_g}\sum_{(x_g,y_g)}log(P_c(y_g|x_g))

$$

- 如有必要，可以引入正则化来避免过拟合，记为  $R_u$

- 采用 SGD 来更新 C：

$$
\nabla_{\theta_c}[\frac{\alpha}{m_c}\sum_{x_c}\sum_{y\in Y}P_c(y|x_c)log(1-D(x_c,y))+R_c+\alpha_pR_p+\alpha_uR_u]

$$

- 最后更新 G：

$$
\nabla_{\theta_g}[\frac{1-\alpha}{m_g}\sum_{x_g,y_g}log(1-D(x_g,y_g))]

$$

---
