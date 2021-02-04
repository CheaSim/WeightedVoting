# WeightedVoting
Weighted Model Ensemble Trick

## 目标

本项目的目标是，全自动融合模型。 以前模型融合的权重是由自己设定或者基于dev上的分数，但是既然我们有了Pytorch这样的框架，为什么不可以利用模型在验证集上的输出分布和验证集的ground truth来输出每个模型的权重呢？ 冲！


输入是logits，输出是概率分布


## 流程

Firstly, use the train set to trian the model. Then, use the trained model to evaluate on the dev dataset to get the dev logits. Now, we have about N model and N dev logits. Usually, we will average the dev logits to ensemble the model. In this project, you could get the final logits more than averaging the logits. We could train the weight in ensemble the models.

<center>
<a href="https://www.codecogs.com/eqnedit.php?latex=logit&space;=&space;\sum_n^i&space;w_i&space;logit_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?logit&space;=&space;\sum_n^i&space;w_i&space;logit_i" title="logit = \sum_n^i w_i logit_i" /></a>
</center>

首先使用几个模型针对验证集输出验证集上的概率分布，之后
