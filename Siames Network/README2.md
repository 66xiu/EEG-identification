# Few-shot Learning/Meat Learning
* Few-shot Learning is a kind of meta learning                 
* Meta Learning: learn to lear.

## Supervised Learning vs Few-shot Learing
* Traditional supervised learing
  - Test samples are never seen before.
  - Test samples are from known classed.
 
![捕获12](https://user-images.githubusercontent.com/109055774/236821374-ed5c0554-8107-4236-8d91-463f81dd78db.GIF)

* Few-shot learning
  - Query samples are never seen before.
  - Query samples are from unkown classed.

![image](https://user-images.githubusercontent.com/109055774/236820835-791488a7-f018-4063-9fa1-598cebf7d084.png)

## k-ways n-shot Support Set
* k-way:the support set has k classes.
* n-shot:every class has n samples.

![1](https://user-images.githubusercontent.com/109055774/236821976-f2adbcfc-eaa9-4e5e-9732-a4a3593c5449.GIF)

![image](https://user-images.githubusercontent.com/109055774/236822136-eae85ad4-cc41-42ac-9512-c4a81a865b12.png)
![image](https://user-images.githubusercontent.com/109055774/236822220-94d2f993-2605-436b-beae-9234c1736c9c.png)

## Basic Idea of Few-Shot Learing
* Train a Siames network on large-scale training set.
* Give a support set of k-way n-shot.
  - k-way means k classes.
  - n-shot means every class has n samples.
  - The training set does not contain the k classes.
* Given a query, predict its class.
  - Use the Siamese network to compute similarity or distance.

### Few-Shot Prediction by Pretrained CNN
* Step1:Pretraining
  - Pretrain a CNN on large-scale training data.
  - Use the CNN for feature extraction.
* Step2:Few-Shot Prediction
  - Map images in the support set to feature vectors.
  - Obtain the mean feature vector of each class:u1,...uk.
  - Compare the feature of query with u1,...uk.
-------------------------------------------------
**优化**
* Step1:Pretraining
  - Pretrain a CNN on large-scale training data.
  - Use the CNN for feature extraction.
* Step2:Fine Tunning  //提高预测百分率
  - Training a classifier on the support set.
  - Tricks:
    1.Using M to initialize W.
    2.Entropy regularization.
    3.Cosine similarity + Softmax classifier.
* Step3:Few-Shot Prediction
  - Map images in the support set to feature vectors.
  - Obtain the mean feature vector of each class:u1,...uk.
  - Compare the feature of query with u1,...uk.

# Siamese Network（孪生网络/连体网络）
## 两种训练Siamese Network的方法：
* 第一中方法：每次取两个样本，比较它们的相似度

![image](https://user-images.githubusercontent.com/109055774/236823383-40f7f94a-f96b-49c2-82ed-2263205e5408.png)

![qq](https://user-images.githubusercontent.com/109055774/236818986-884fe14e-8c99-4daf-b1f0-64077817348c.GIF)

* 第二种方法：Triplet Loss

![image](https://user-images.githubusercontent.com/109055774/236823931-4c5983e4-7f09-41c8-9918-010a5d8af63b.png)


