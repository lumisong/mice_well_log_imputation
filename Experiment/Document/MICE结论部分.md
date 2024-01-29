# 6. 结论

Many subsurface analysis tasks and workflows rely upon or can benefit from a complete well logging data set.
许多地下分析任务和工作流程依赖于或可以从完整的井下测井数据集中获益。
However, in many cases the logging measurements are rarely complete with gaps or logs missing entirely.
然而，在许多情况下，测井测量很少完整，存在间隙或完全缺少某些测井记录。
This study has applied the MICE approach to successfully and completely impute multiple well logs simultaneous using ML algo rithms.
本研究成功并完整地应用了MICE方法，使用机器学习算法同时插补了多个井下测井记录。
Of the four algorithms that were tested, GBTs performed the best.
在测试的四种算法中，GBT表现最佳。
Although MICE did not always improve the direct imputation of logs when using GBT, imputation when certain combinations of missing log values, may benefit from the iterative approach.
尽管使用GBT时，MICE并不总是改善直接插补的测井记录，但在特定组合的缺失测井值时，可能会从迭代方法中获益。
MICE can also improve GBT results when the sparsity of the input data is high.
当输入数据的稀疏度较高时，MICE也可以改善GBT的结果。

Finally, while GBT have the ability to naturally handle missing values in the input features, many ML algorithms cannot.
最后，虽然GBT有能力自然处理输入特征中的缺失值，但许多机器学习算法却做不到。
MICE may be most applicable in scenarios where algorithms require complete input features for training.
在需要完整输入特征进行训练的算法场景中，MICE可能最为适用。

## 作者贡献声明

Antony Hallam 开发了方法论、代码，参与了应用和写作；Debajoy Mukherjee 开发了代码和方法论；Romain Chassagne 讨论了方法论，指导了研究和写作。

## 计算机代码和数据可用性

用于分析和使用MICE进行测井插补的源代码可从第一作者处获得，可从 [https://github.com/trhallam/mice_well_log_imputation](https://github.com/trhallam/mice_well_log_imputation) 下载。Volve井下测井数据可从Equinor提供的Volve Data Village下载，网址为 [https://www.equinor.com/en/what-we-do/digitalisation-in-our-dna/volve-field-data-village-download.html](https://www.equinor.com/en/what-we-do/digitalisation-in-our-dna/volve-field-data-village-download.html)。Force 2020数据可从Xeek下载，网址为 [https://xeek.ai/challenges/force-well-logs/overview](https://xeek.ai/challenges/force-well-logs/overview)。

## 利益冲突声明

作者声明他们没有已知的与本文报告的工作有关的财务利益冲突或个人关系。

## 致谢

我们要感谢爱丁堡时间推移项目第七阶段的赞助商：AkerBP、BP、CGG、Chevron/Ithaca Energy、CNOOC、Equinor、ConocoPhillips、ENI、Petrobras、Norsar、Woodside、Taqa、Halliburton、ExxonMobil、OMV和Shell的财务支持。感谢Equinor提供Volve公共数据集。感谢FORCE2020的赞助商和挪威政府提供F2020数据集。感谢同事和同行的帮助和反馈。我们还要感谢斯伦贝谢公司提供其软件和Python开源社区的支持。我们还要感谢本文的审稿人，在出版过程中提供了宝贵的反馈和建议。
