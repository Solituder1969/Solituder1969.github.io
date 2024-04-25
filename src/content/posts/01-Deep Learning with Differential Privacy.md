---
title: 01-Deep Learning with Differential Privacy
published: 2023-07-01
description: '差分隐私和深度学习结合的开篇之作，首次提出对模型梯度裁切、添加噪声的脱敏算法，Moments Accountant 给出相比强组合定理更紧的隐私保护上界。'
tags: [Differential Privacy, Moments Accountant, DeepLearning model]
category: '论文研读'
draft: false 
---
# 一、前言

        机器学习会使用到大量的众包数据，过程中可能会隐含一系列的敏感数据，模型可能会暴露其中的隐私问题。因此作者在差分隐私的框架下开发了一种算法技术，它可以分析隐私成本，基于非凸对象训练深度神经网络，基于适当的训练预算和算法复杂度，训练出有效的模型。

# 二、介绍

## 2.1 作者为此实施的策略

1. 对隐私损失的详细信息进行追踪，可以在渐近性和经验性两方面对整体的隐私损失获得一个严格的评估。

2. 作者通过“**引入对单个训练样例有效的梯度计算算法**”、“**任务细分为较小的批次以减少内存的占用**”、“**在输入层使用差分隐私保护策略”来提高差分隐私的计算效率**。

3. 基于“ Tensor Flow”构建程序，以“MNIST”和“ CIFAR-10”这两个标准图像分类任务来评估。实验表明：**深度神经网络的隐私保护，能够以一种在软件复杂度、训练效率、模型质量中适量的成本，来实现**。

## 2.2 一些基本事实

1. 常规的技术，为了避免过拟合，会隐藏一些样例的细节；

2. 解释深度神经网络是困难的，大容量的数据**可能潜在的编码了一些训练样例的细节**；

3. 一些攻击者能够因此发现训练样例的一些部分。

## 2.3 基于的假设

1. 攻击者**充分了解了训练的机制和模型参数的访问**；

2. 模型参数**本身可能收到敌人的检视**；

3. 当我们关注于一个训练样本的隐私记录时，**攻击者能够控制其他的部分乃至全部数据**；

# 三、差分隐私和深度学习的背景介绍

## 3.1 差分隐私的定义：

        设有随机算法M，Pm为M所有可能的输出构成的集合。对于任意两个临近数据集D和D'以及Pm的任何子集Sm，若算法M满足：

<img src="https://cdn.nlark.com/yuque/__latex/a87cc30bb46d088a6b533ea40bfe518d.svg" title="" alt="https://cdn.nlark.com/yuque/__latex/a87cc30bb46d088a6b533ea40bfe518d.svg" data-align="center">

        则称算法M提供“ε-差分隐私保护”，其中参数ε称为隐私保护预算。

        初始定义中并不含有δ，这里引用了Dwork等人的工作，它允许普通的ε-差分隐私有δ的概率被破坏。

## 3.2 差分隐私的应用：

        在**组合性、组隐私、非机密信息的鲁棒性**上，有着特别的优势。

        组合性：**如果一个机制的所有组成部分都是差分隐私的，则这个机制本身也是如此。**

        组隐私：**如果数据集包含的输入是相关的，例如都是同一个个体提供的输入，那么组隐私意味着隐私保护的适当降级。**

        鲁棒性：**意味着隐私保护不受任何可能由攻击者掌握的附带信息的影响。**

## 3.3 差分隐私的设计机制：

        （1）通过有界灵敏度函数的顺序组合来逼近函数性；

        （2）选择添加噪声的参数；

        （3）对结果机制进行隐私分析

# 四、 方法

## 4.1 差分隐私的随机梯度下降算法（SGD）

        人们可能试图只通过对训练过程产生的最终参数进行处理，将这个过程视为黑盒，以此来保护训练数据的隐私。但对训练集上参数的独立性缺少没有一个很好的认知，向参数里添加过于保守的噪声（这些噪声可能还是根据最糟样例的分析来选取的），有可能会对学习模型的效用造成毁灭性的打击。

        因此，作者更青睐于一种，能够在训练过程里就控制训练集影响的复杂算法（尤其是在随机梯度下降算法的计算中）。其流程为：

![https://cdn.nlark.com/yuque/0/2022/jpeg/27569747/1662521246014-02817b6f-a43b-4b5d-9947-0281bb1bc420.jpeg](https://cdn.nlark.com/yuque/0/2022/jpeg/27569747/1662521246014-02817b6f-a43b-4b5d-9947-0281bb1bc420.jpeg)

        整体过程符合了SGD的步骤：随机选取样例，计算梯度，梯度下降，加噪，调参，如此迭代，直到满足退出机制。

几个关键的点：

1. **Norm clipping** : 为证明算法1的差分隐私保证，需要限定每个个例对梯度 $\hat{g}_{t}$ 的影响。由于没有对梯度范围的先验限定，因此这里以l2标准裁剪了每个梯度，梯度向量g被$g/max(1,\frac{||g||_2}{C})$ 所替代。这种方式确保了：如果||g||2 不大于C，那么保留g。否则将其规模裁剪到C的尺度。

2. **Per-layer and time-dependent parameters :** 算法1的伪代码把所有的参数都组合成了损失函数L(·）的一个单独输入。对多层神经网络，分别独立考察每一个层，这样就允许设置不同的裁剪阈值C以及为不同的层设置噪声规模σ。

3. **Lots :** 算法1通过计算一组样例的损失的梯度并且取均值，来评估L的梯度。这个均值提供了一个无偏估计量，它的方差会随着群组的规模而快速减小。文中将其称为“一个lot”，用来和常规分组的“batch”区别。为了限制内存的消耗，这里将batch的规模限制到小于lot的规模L(L是算法的参数)。**我们基于多个batch计算梯度，然后将多个batch组合成一个lot，并添加参数**。实践中，为了提高效率，batch 和 lot 的分组是通过随机排序样本，然后把它们以适当的大小进行分割。为了易于分析，假设 lot 是按照概率 $q=\frac{L}{N}$  采样每个样本得到的，其中 N 是训练数据包含的样本量。本文将 epoch的数量作为训练算法的运行时间的衡量，其中每一个 epoch 包含多个 batch ，在一个 epoch 中将处理 N 个数据样本，一个 epoch 包含 N/L个 lots。

4. **Privacy account :** 对差分隐私SGD而言，一个重要的议题是计算训练的整体隐私成本。由于差分隐私的组合性，我们能够设置一个“accountant”的过程，在每次使用训练数据的时候，就计算隐私成本，并在训练过程中累积该成本。训练的每个步骤都可能需要多层的梯度，这个时候accountant就计算这些梯队对应的累积隐私成本。

5. **Moment accountant** : 诸多的研究都致力于研究特定的隐私损失和组成它的噪声分布。对于作者他们使用的高斯噪声而言，在算法1中设置了参数 $\sigma=\sqrt{2log\frac{1.25}{\delta}/\epsilon}$，然后根据相关的理论，每一步关于lot都满足了(ε，δ)-差分隐私。由于lot是从数据集里随机采样的，根据“隐私扩大”理论，每一步关于整个数据集都符合(qε，qδ)-差分隐私，其中q=L/N，是每个lot的采样率。“强组合定理”揭示了最佳的总体边界。

       但是，强组合定理给出的隐私上界可能是松的，并且它并没有关注特殊的噪声分布。在作者的工作中，他们创造了一种更强的统计方法，称之为“Moment accountant”（矩统计）。根据这个方法，选择合适的噪声规模和裁剪的阈值，可以使得算法1满足($O(q\epsilon\sqrt{T}),\delta$)-差分隐私。对比于强组合定理，作者给出的界在两个方面更紧：一个是在ε部分省略了一个$\sqrt{log(1/\delta)}$，在δ部分省略了一个Tq因子。因为总是期望δ更小一点，但是T远大于了1/q。作者对此给出了充分的证明，这是本文的主要贡献之一。

## 4.2 矩统计

        矩统计追踪了隐私损失速记变量的矩的界。作者的研究表明，矩统计方法不仅适用于组合高斯机制，而且也使用于随机采样和高斯机制的组合，并且能够对算法1的隐私损失提供更紧的估计。

        隐私损失是一个随机的变量，依赖于算法添加的随机噪声。因此，如果说一个机制M满足了(ε，δ)-差分隐私，那么就相当于机制M的隐私损失随机变量存在一个确定的尾部界。

        直接使用尾部界构建会导致一个松散的界。因此作者联合了矩边界、标准马尔可夫不等式，来获得尾部界。这就是差分隐私意义上的隐私损失。

更一般的，对于邻近数据集d,d'∈D^{n}，一个机制M，辅助输入aux，一个输出o，输出o上的隐私损失就可以被定义为：

<img title="" src="https://cdn.nlark.com/yuque/__latex/df167d7d793fc45a5999539877be0170.svg" alt="https://cdn.nlark.com/yuque/__latex/df167d7d793fc45a5999539877be0170.svg" data-align="center">

        作者通过设置第k个机制M_k的输入为前面机制的输出，以此来建立一个自适应组合的设计模式。

        对于一个给定的机制M，定义第λ阶矩a_m(λ，aux,d,d')为矩生成函数在λ点处的期望的对数：

<img src="https://cdn.nlark.com/yuque/__latex/1e2025940d1a78779136e5b42bc66804.svg" title="" alt="https://cdn.nlark.com/yuque/__latex/1e2025940d1a78779136e5b42bc66804.svg" data-align="center">

        为了证明一个机制的隐私保护，关于所有可能的a_M(λ;aux,d,d')的界是非常有用的。因此，给出定义：

$$
a_M(\lambda)\triangleq \underset{aux,d,d'}{max} a_M(\lambda;aux,d,d')


$$

        其中关于最大化的变量是所有可能的aux和所有的邻近数据库d,d'。

        关于a的一些性质（让a_{M}(λ)如上定义）：

- 【可组合性】假设机制M集成了一系列自适应机制M1,...,Mk，其中 $M_i: Π_{j=1}^{i-1}R_j×D——>R_j$，对任意λ有：

<img src="https://cdn.nlark.com/yuque/__latex/92ba329116589992bec50feea86cd42b.svg" title="" alt="https://cdn.nlark.com/yuque/__latex/92ba329116589992bec50feea86cd42b.svg" data-align="center">

- 【尾部有界】对任意ε＞0，机制M是(ε，δ)-差分隐私的，如果：

$$
\delta=\underset{\lambda}{min} \;exp(a_M(\lambda)-\lambda_{\epsilon})
$$

        需要注意的是，上述性质只在“机制自身的选取是基于先前机制的输出”时才成立。

        通过上述定理，可以在每一步都方便的计算或约束a_{Mi}(λ)，将其累加并得出所有机制的界。然后使用尾部界来转换矩界来保证满足（ε，δ）-差分隐私。

        接下来的主要挑战就是在每一步里求值a_M(λ)的界。在随机采样中，使用高斯机制的，只需要估计下方的矩即可。设μ0表示N(0,σ^{2})的概率密度函数，μ1表示N(1,σ^{2})的概率密度函数，μ是两个高斯分布的混合μ=(1-q)μ0+qμ1。接着需要计算a(λ)=logmax(E1,E2)，其中：

![https://cdn.nlark.com/yuque/__latex/85b8c5cbd791decb09d3746c0d553315.svg](https://cdn.nlark.com/yuque/__latex/85b8c5cbd791decb09d3746c0d553315.svg)

        除此之外，还可以得出近似界：

<img src="https://cdn.nlark.com/yuque/__latex/0747807f21131bfec49cc9a097220440.svg" title="" alt="https://cdn.nlark.com/yuque/__latex/0747807f21131bfec49cc9a097220440.svg" data-align="center">

## 4.3 超参数调节

        超参数能够用来平衡隐私、准确性和性能。

        作者的两个超参数选择是：batch size和学习率。

        虽然凸目标函数的差分隐私优化最好使用小到1的批次大小来实现，但非凸学习适合使用更大的批次，而定理1解释了，大的批次会增加隐私成本，并且合理的平衡每个epoch的batch数量。

        当模型收敛到局部最优时，非私有训练的学习率通常会向下调整。但差分隐私的训练需要的时间相对较少，因此无需降低学习率。

## 4.4 实施流程

        为了实现隐私保护，作者做了两件事：1、设置过滤器（净化梯度，sanitizer）；2、追踪隐私开销(隐私统计，privacy accountant)。

- **过滤器**：主要做了两件事：①通过梯度裁剪来限制每个独立样本的敏感度；②在更新网络参数之前，给每个batch的梯度加噪声。

- **隐私统计**：通过渐近界、计算闭合表达式、应用数值方法，来计算a(λ)。渐近界弥补了一般组合定理的不足，后两者给出了一个更准确的累积隐私损失。作者发现针对这些参数，只需计算 λ≤32 时的 α(λ) 的值即可。

- **差分隐私PCA ：**执行PCA会产生隐私消耗，但同时会提高模型质量、减少训练时间。

- **卷积层：**根据Jarrett等人的研究工作，探索在公共数据上学习卷积层的想法。这样的卷积层可以基于图像模型的GoogLeNet或AlexNet特征，也可以基于语言模型中预先训练的word2vec或GloVe嵌入。

# 五、实验结论

## 5.1 应用矩统计

        令q=0.01，σ=4，δ=10^(-5)，训练得出，矩统计比强组合定理给出的隐私损失上界更紧。在epoch=400时，两者达到了一个数量级的差距。

## 5.2 MNIST

- **基线模型：**一个60维的PCA投影层，一个具有1000个隐藏单元的隐藏层，lot大小为600，在100个epoch内获得了98.3%的精确度。

- **差分隐私模型：**使用了相同的结构，即：一个60维的PCA投影层，1个具有1000个隐藏单元的隐藏层，lot大小为600。为了限制敏感度，裁剪了每层梯度的阈值为4。根据σ的大小划分了三种不同的噪声：small(σ=2，σp=4), medium(σ=4，σp=7), large(σ=8，σp=16)。其中，σ代表了训练神经网络的噪声级别，σp代表了PCA投影的噪声设置。学习率初始时设置为0.1，然后周期性地降低，经历10个epochs周期后，降低并稳定到0.052。作者还发现，单个隐藏层与PCA组合的效果，要好于两个隐藏层。

        应用差分隐私SGD，模型在训练集和测试集之间的差距非常小。而在非隐私训练中，由于过拟合的存在，导致两者的差别随着数据集的增大而扩大。

        使用矩统计，对任意给定的ε，都可以获得一个δ值。对于固定的δ，变动ε的值，最终的精确度变化会非常大。但对于固定的 ε，δ的变化却不怎么会影响精确度的值。

- **参数的影响：**分类的准确度被多个参数所影响，包括网络的拓扑结构，PCA的维数，以及隐藏神经元的数量，以及一些学习过程中的参数，比如Lot的大小和学习率。此外，梯度的正则裁切阈值以及噪声的等级，会对隐私产生比较大的影响。

        为了验证参数的影响，作者对此进行了单独的测试。初始化与“差分隐私模型”一致，对值的每种组合，直到不满足(2，10^{-5})-差分隐私为止。

- **PCA投影：**准确率关于PCA维数的变化比较稳定，在维数达到60时最佳。把维数从784减少到60时，训练时间减少了10倍，随机投影的准确率达到了92.5%

- **隐藏层神经元数量：**对差分隐私训练来说，隐藏层神经元数量的增加是否会提高精确度，这不是先验知道的，因为隐藏层神经元的增多，这会增加梯度的敏感度，进而需要增加更多的噪声来保护隐私。

        但是，反直觉的是，增加隐藏层的神经元数量，不会降低训练模型的准确率。一个可能的解释是，大型的网络多噪声有更高的容忍度。

- **Lot大小：** 在满足预算限制的同时，可以运行N/L个epochs。这里需要平衡两个冲突的因素：一方面，小规模的lot能够运行更多的epochs，进而提高精确度。另一方面，大的lot，加入的噪声对准确度的影响相对较小。

        实验表明，lot大小对实验的精确度有极大的影响，最佳的范围是$\sqrt{N}$，其中N是训练样例的数量。

- **学习率：** 当学习率处于[0.01，0.07]时，精确度会比较稳定，并且在0.05时达到巅峰。但当学习率比较高时，精确度会急速下跌。不过也有一些实验表明，即使对于大的学习率，通过降低噪声尺度也可以达到较高的准确率，据此，通过减小训练周期以满足隐私预算。

- **裁剪界：** 限制梯度范数有两个消极的影响：一个是，如果裁剪的过小了，那么梯度下降就会与真实情况的方向产生很大的偏差；另外一个是，过大的阈值C会要求增加更多的噪声。实际中，通常选择训练过程中，未裁剪梯度值的中位数来作为C的值。

- **噪声等级：** 通过添加更多噪声，每轮训练的隐私损失在比例上更小，因此可以在给定的累积隐私预算内，可以进行更多轮训练。噪声尺度的选择对精确度有着较大的影响。

        总结：

（1）PCA投影同时提高了训练精度和模型性能。准确率对于大多数投影维度及其使用的噪声尺度的取值而言很稳定。

（2）网络规模很大程度上能够影响精确度的稳定性。当我们只能运行很小的的epochs时，使用更大的网络，效果会更好。

（3）训练参数，尤其是lot大小和噪声规模σ，对模型的精确度有着很大的影响。

## 5.3 CIFAR

        首先通过取中心点，将 32×32 大小的图片裁剪为 24×24 。网络结构由两个卷积层与两个全连接层组成。卷积层使用 5×5 的卷积核，卷积步长stride为1，激活函数为ReLU，2×2 的最大池化层，每层为64通道，因此，第一个卷积层关于每个图片输出一个 12×12×64 的张量，第二层输出一个 6×6×64 的张量。后者被展开为一个向量，并输入到两个全连接神经网络中，这两个神经网络的输入维数均为384。

        该框架在非隐私训练的情况下，进行500次训练周期 epoch 可以达到86%的准确率。本文实验选择该框架的原因是其非常简单。不过应当注意，通过使用更深的网络，以及不同的非线性映射和其它高级技术，可以实现更高的准确率，目前该数据集上最高的准确率大约为96.5%

        作为基线标准，epoch为250，batch为120，准确率为80%

- **差分隐私版本：** 对于差分隐私版本，本文使用相同的网络结构。如上所述，使用预训练的卷积层。全连接层也初始化为预训练网络。训练softmax层以及顶部或整个全连接层。基于对梯度范数的观察，softmax层的梯度范数大约是其它层的两倍，以不同的裁剪阈值（3～10之间）对梯度进行裁剪，同时保持该大小比例不变。lot 的大小是另一个需要调整的参数，本文实验尝试了600，2000和4000三种取值。并且随着 lot 的增加，每轮训练周期 epoch 的耗时从40秒上升到180秒。

        与MNIST数据集相比，非隐私训练与隐私训练的准确率相差大约1.3%，在CIFAR-10上的实验该差距更大（大约为7%）。

# 六、相关工作和总结

        最后总结了相关的工作，包括隐私保护的历史，a(λ)的引入，学习算法，模型类别……

        本文证明了在满足差分隐私的条件下对深度神经网络进行训练，关于具有很多参数的模型的隐私损失较小。在MNIST的实验中，本文方法实现了97%的训练准确率，在CIFAR-10数据集上实现了73%的准确率，两种情况下均满足 (8,10−5)-差分隐私。

        本文算法为一个随机梯度下降的差分隐私版本；其基于TensorFlow软件库运行。由于本文方法直接应用于梯度计算，因此，其可以适用于很多其它类型的优化算法。