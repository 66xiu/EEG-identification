# 基于EEG的生物特征识别

对于脑机接口（BCI）的研究有一个突出的主题是安全问题，安全问题可以分为身份验证和身份识别

## 深度学习在基于EEG识别方面

早期基于EEG的脑生物特征识别研究是从脑电图信号中提取PSD、AR、离散傅里叶变换(DFT)和小波包变换(WPT)等特征，并使用基于相似性的分类器来确定个体的身份。相似度可以用欧几里得距离、马氏距离或互相关来衡量。随后，k-NN及其变体
。

然后，监督机器学习(ML)，包括支持向量机(SVM)，随机森林(RF)，线性判别分析(LDA)，高斯混合模型(GMM)和多层感知器(MLP)。这些方法有一定的局限性，特征提取步骤需要领域知识，
不能保证创建完全不同的特征。

深度学习方法试图解决上面问题。利用CNN从原始脑电图信号中自动提取基本特征。然后发展有  RNN、GRU、CNN+RNN、CNN+LSTM、CNN+GRU、GCNN、CNN+GCNN等等。

----
**CNN**:    
*2018:《EEGNet: A Compact Convolutional Neural Network for EEG-based Brain-Computer Interfaces》：介绍了EEGNet，
一种用于基于脑电图的BCI的紧凑卷积神经网络,使用深度卷积和可分离卷积来构建一个特定于脑电信号的模型，该模型封装了脑电信号特征提取概念。*

**RNN**:    
*2018:《MindID: Person Identification from Brain Waves through Attention-based Recurrent Neural Network》：
提出了一种基于脑电图的生物特征识别方法MindID，分解Delta信号输入到基于注意的编码器-解码器RNNs结构中，该结构根据不同的EEG通道的重要性为其分配不同的注意权重。从基于注意的RNN中学习到的判别表示用于通过增强分类器识别用户.*   

**CNN+RNN**:     
*2015：《Affective EEG-Based Person Identification Using the Deep Learning Approach》：提出了一个级联深度学习，使用卷积神经网络(cnn)和循环神经网络(RNNs)的组合。
CNN用于处理脑电图的空间信息，而rnn则提取脑电图的时间信息.评估了两种类型的rnn，即长短期记忆(LSTM)和门控循环单元(GRU)*
     
*2019:《EEG-based user identification system using 1D-convolutional long short-term memory neural networks》:提出了一种基于一维卷积长短期记忆神经网络的基于eeg的用户识别系统的新方法。
提出的一维卷积LSTM结合使用cnn和LSTM，利用脑电信号的时空特征。*

**GCNN**：     
*2019:《Convolutional Neural Networks Using Dynamic Functional Connectivity for EEG-based Person Identification in Diverse Human States》:提出了一种新的脑电图生物特征识别模型,其新颖之处在于将脑电信号表示为基于频内和跨频函数连通性估计的图，
并使用图卷积神经网络(GCNN)自动捕获脑电信号的深层内在结构表征以进行人员识别。*

**CNN+GCNN**:    
*2021:《EEG-BBNet: a Hybrid Framework for Brain Biometric using Graph Connectivity》:
提出了EEG-BBNet，一种集成了卷积神经网络(CNN)和图卷积神经网络(GCNN)的混合网络。联合利用了CNN在自动特征提取方面的优势和GCNN通过图表示学习EEG电极之间连通性的能力。*

----

## 基于半监督学习
近年来，需要大量类别标签的监督方法在脑电图表示学习中取得了良好的效果。然而，标记脑电图数据是一项具有挑战性的任务。
需要很少输出标签的整体半监督学习方法在计算机视觉领域显示出良好的效果。然而，半监督学习在BCI领域的应用非常少，其中主要都是应用在基于情绪识别方面的，而很少用于身份识别方面。

------
*2021:《Semi-Supervised Contrastive Learning for Generalizable Motor Imagery EEG Classification》：提出了一个具有对比学习和对抗性训练策略的领域独立的端到端半监督学习框架。*

*2022：《A Semi-Supervised Progressive Learning Algorithm for Brain–Computer Interface》：提出了一种端到端的半监督学习框架，用于脑电分类和脑电-肌电融合分析。*

*2022：《HOLISTIC SEMI-SUPERVISED APPROACHES FOR EEG REPRESENTATION LEARNING》：在本文中，采用了三种最先进的整体半监督方法，
即MixMatch， FixMatch和admatch以及五种经典的半监督方法进行EEG学习。*

-----------

## 发展现状      

为基于脑电图的BCI应用程序探索的数据集分布。     

![image](https://user-images.githubusercontent.com/109055774/226892836-dc18ff88-e92c-41bd-9065-945cb90ca45e.png)


在BCI竞赛数据集上，各种深度学习方法（即卷积神经网络(CNN) (Islam等人，2021)、长短期记忆(LSTM)、堆叠自编码器(SAE)和变分自编码器(VAE)）的准确性的比较：


![image](https://user-images.githubusercontent.com/109055774/226607117-f1d623fc-9dca-42f3-a0b0-e146e4b1d734.png)

各种深度学习方法在DEAP数据集上的准确性图表：


![image](https://user-images.githubusercontent.com/109055774/226607259-7a89b944-dcb1-4bb9-ada1-571299fb0ea8.png)


在现有的出版期刊中，使用频率最高的是判别模型，尤其是CNN。因为几乎所有的BCI问题都可以归入分类问题的范畴。     

    超过75%的模型是由CNN算法提供支持的。
    只有15%的基于模型的文章使用了循环神经网络(RNN)，RNN处理长序列需要时间，而脑电图信号通常是长序列。
    对于BCI研究的混合模型中，RNN和CNN的组合约占三分之一。因为RNN和CNN有出色的时间和空间特征提取能力。
    结合代表性模型和鉴别模型是另一种混合模型。前者用来提取特征，后者用来把东西分组。例如，一些研究提倡将CNN与MLP结合，利用CNN结构提取空间数据，然后将这些数据交给MLP进行分类。

## 基于EEG的身份识别面临的研究挑战与未来研究方向
