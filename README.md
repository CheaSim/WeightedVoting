# WeightedVoting
Weighted Model Ensemble Trick

## 目标

本项目的目标是，全自动融合模型。 以前模型融合的权重是由自己设定或者基于dev上的分数，但是既然我们有了Pytorch这样的框架，为什么不可以利用模型在验证集上的输出分布和验证集的ground truth来输出每个模型的权重呢？ 冲！
