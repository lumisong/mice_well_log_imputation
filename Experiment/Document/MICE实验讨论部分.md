# 5. Discussion 讨论

## 5.1. Explanation of results结果解释

On the whole, multiple iterations of the MICE algorithm do not appear to improve the overall predictive capacity of the models implemented (Tables 4 and 6).
总体而言，MICE算法的多次迭代似乎并未改善实施模型的整体预测能力（表4和表6）。
There are exceptions where certain combinations of missing logs and sequential imputation and prediction are desirable, but this does not appear to form the bulk of missing values in the tested data sets.
存在某些情况下，特定的缺失测井组合和顺序插补以及预测是可取的，但这似乎并不构成测试数据集中缺失值的主体。
GBT generally out performed the BRR and KNN models.
GBT通常优于BRR和KNN模型。
The ability for D-GBT to handle missing values directly is a significant advantage with performance as good or betterthan the MICEGBT method.
D-GBT能够直接处理缺失值，这一点是一个重大优势，其性能与MICE-GBT方法一样好或更好。
The exceptions where MICE improves upon direct imputation were observed, when the elastic logs were all missing.
当所有的弹性测井都缺失时，观察到MICE改进了直接插补的情况。
In these scenarios, complete prediction of all three targets is required and accuracy improves when sequential imputation is performed.
在这些情况下，需要完整地预测所有三个目标，当执行顺序插补时，准确性会提高。
It is speculated that the more complex non-parametric relations between logs, are handled better with MICE.
推测MICE更好地处理了测井之间更复杂的非参数关系。
For example, DRHO may be used to predict RHOB, which is subsequently used to predict DT and then DTS - as in Fig. 4 samples 25,000 to 30,000.
例如，可能使用DRHO来预测RHOB，随后用于预测DT，然后是DTS - 如图4中的样本25,000至30,000所示。

Any application of MICE will be more expensivethan direct or single pass sequential imputation.
任何MICE的应用都比直接或单次顺序插补更昂贵。
However, in this study, the prediction algorithms chosen did not make the additional computational cost un manageable.
然而，在本研究中，所选择的预测算法并未使额外的计算成本变得难以管理。
We found that MICE typically runs for 10–20 iterations before reaching convergence, requiring multiple imputation predictors to be trained and stored.
我们发现MICE通常在进行10-20次迭代后达到收敛，需要训练并存储多个插补预测器。

Of the three machinelearning predictive models tested, GBT appears to be a clear leader,exhibiting superior correlation and lower error.
在测试的三种机器学习预测模型中，GBT显然是领先者，展现了更优的相关性和更低的误差。
The strength of the ensemble approach over BRR and KNN, may reflect a superior capacity to identify and model the complex non-parametric relationships between logs (Ke et al., 2017).
与BRR和KNN相比，集成方法的优势可能反映了更好的能力来识别和模拟测井之间复杂的非参数关系（Ke等，2017）。

An observation from both data sets was the relative under performance of models in geological zones with small sample sizes, where the log relationships is more difficult to define due to a lack of data.
从两个数据集的观察中，我们发现在样本量较小的地质区域中模型相对表现不佳，因为由于数据缺乏，定义测井关系更为困难。
A subsequent analysis calculating metric scores for each zone (Fig. 7), identified significant variations in metric scores for the test logs, when sample sizes are small.
随后对每个区域计算指标得分的分析（图7），发现当样本量较小时，测试测井的指标得分存在显著变化。
We attribute the variance in the scores, at small sample size, to complex and multi-factored interactions between the available training samples, the sparsity of the input features, for the zone, and the effectiveness of cross-training between lithologically similar zones.
我们将小样本量下得分的差异归因于可用训练样本之间复杂且多因素的交互，区域内输入特征的稀疏性，以及跨岩性相似区域的训练有效性。
The zones with larger sample sizes tend to exhibit more stable metric behaviour, leading us to suspect insufficient data; both for training purposes and for the data available to calculate representative metric scores at the zone scale.
样本量较大的区域倾向于展现更稳定的指标行为，这使我们怀疑数据不足；无论是用于训练目的还是用于计算区域尺度代表性指标得分的数据。
Further studies could investigate the limits of how much log data is required to achieve stable results.
进一步的研究可以探究需要多少测井数据才能获得稳定的结果。

At larger sample sizes, there is a compression in the error variance across zones, (particularly for the error metrics MSE/MAE) suggesting a minimum number of samples needed in a zone to achievestability in the predictive model.
在较大样本量下，各区域间误差方差压缩（特别是对于误差指标MSE/MAE），表明一个区域内需要一定数量的样本才能在预测模型中实现稳定性。

A more robust approach, using geological zones, might be to consider undertaking this analysis based upon lithological characteristics, rather than formation names.
使用地质区域的更稳健方法，可能是基于岩性特征而不是地层名称来进行此分析。
This could perhaps be approached either via manual labelling, grouping of similar zone labels or, if the sample set is sufficiently dense, clustering algorithms.
这可能通过手动标记、对相似区域标签进行分组或者，如果样本集足够密集，使用聚类算法来实现。

## 5.2. MICE limitations and assumptions MICE的局限性和假设

All imputation or prediction methodologies including MICE, rely upon a sufficient quantity of data to correctly calibrate the model.
所有的插补或预测方法，包括MICE，都依赖于足够数量的数据来正确校准模型。
If characteristically unique sections of log are absent from the training dataset, the predictive models will be unable to reproduce such trend or data behaviour.
如果训练数据集中缺少具有特征性的唯一测井段，则预测模型将无法再现此类趋势或数据行为。
Consequently, this limits the application of MICE to data sets with representative sampling that can lead to poor generalisationof the model, unless the training data set is sufficientlydiverse.
因此，这限制了将MICE应用于具有代表性采样的数据集，除非训练数据集足够多样化，否则可能导致模型泛化能力差。
In practice, and for well log imputation, we suggest the pragmatic approach would be to tailor an imputation model for each unique input data set, because the cost of training with the test predictors was acceptably low (on the order of minutes).
在实践中，对于井下测井插补，我们建议的实用方法是为每个独特的输入数据集定制一个插补模型，因为使用测试预测器进行训练的成本在接受范围内（以分钟为单位）。

There is also a degree of non-repeatability with most ML predictors. 
大多数ML预测器也存在一定程度的不可重复性。
If the input data set is added to or changed, the output predictions and imputations are also likely to change.
如果输入数据集被添加或更改，输出预测和插补也可能改变。
The degree of observed differences, will depend upon the changes to the input and the dependence upon randomness in the training of the prediction models.
观察到的差异程度将取决于输入的变化以及预测模型训练中随机性的依赖程度。
The nonrepeatable nature of the imputations, may discourage downstream users of the data, who require stable logs as input, to their own work flows.
插补的不可重复性可能会阻碍下游数据用户，他们需要稳定的测井作为自己工作流程的输入。
In these cases, the general prediction capability of ML regression models, must be traded off against the labour intensive but more stable empirical, or manual prediction approach.
在这些情况下，ML回归模型的一般预测能力必须与更稳定但更费力的经验性或手动预测方法权衡。
A potential solution, could be to capture and store multiple imputed versions, as a measure of the imputation uncertainty, in a similar manner as developed in Diaz and Zadrozny (2020).
一个潜在的解决方案可能是捕获并存储多个插补版本，作为插补不确定性的衡量，类似于Diaz和Zadrozny（2020）开发的方法。

## 5.3. Further research 进一步研究

MICE performed better than direct imputation for the elasticlogs DT, DTS and RHOB, in cases where all three logs were absent or mostly absent. 
在所有三个弹性测井DT、DTS和RHOB都缺失或大部分缺失的情况下，MICE比直接插补表现更好。
Therefore, future research should focus on scenarios where elastic data is sparse.
因此，未来的研究应关注弹性数据稀疏的情况。

For training validation, we use randomisedselection of points within logs (Darling, 2005), which is not typically how gaps occur in well log data.
对于训练验证，我们使用了随机选择测井中的点（Darling，2005），这通常不是井下测井数据中出现间隙的方式。
Alternative approaches such as the methodemployed by Lopes and Jorge (2018) of pseudo modelling the gaps based upon a statistical analysis may further test the robustnessof our approach.
如Lopes和Jorge（2018）所采用的方法，基于统计分析对间隙进行伪建模的替代方法可能会进一步测试我们方法的鲁棒性。
Validation tests could also be augmented to check for over-training by utilising k-fold cross-validation methods common in machine learning.
验证测试还可以通过使用机器学习中常见的k折交叉验证方法来增强，以检查是否存在过度训练。

We also suggest that the MICE algorithm might be modified to improve and or automate noise or bad data rejection.
我们还建议可以修改MICE算法以改进和/或自动化噪声或坏数据的拒绝。
Currently, log editing is required beforehand to quality control the input, the MICE process may be a tool that can identify and automatically remove data which fails a tolerancecriterion when compared with predictions.
目前，需要事先进行测井编辑以控制输入质量，MICE过程可能是一个工具，可以识别并自动移除与预测相比不符合容忍标准的数据。
Initial imputation values use by MICE methods could also be improved by using empirical relationships rather than the mean value of a feature (Azur et al., 2011).
MICE方法使用的初始插补值也可以通过使用经验关系而不是特征的平均值来改进（Azur等，2011）。

Unlike Brown et al. (2020) , we have not included derived petrophysical logs in this study.
与Brown等人（2020）不同，我们在本研究中没有包括派生的岩石物理测井。
From a machine learning perspective, petrophysical logs such as water saturation, porosity and clay volume can be viewed as engineered features which augment or extend our view of the raw input data.
从机器学习的角度来看，岩石物理测井，如含水饱和度、孔隙度和黏土体积，可以被视为增强或扩展我们对原始输入数据的了解的工程特征。
Their addition to the imputation workflow may improve correlations and relationships between the raw data that were ignored previously.
将它们添加到插补工作流程中可能会改善之前忽略的原始数据之间的相关性和关系。

There are many ML algorithms and we have tested some of the easiest to implement.
有许多ML算法，我们测试了一些最易于实现的算法。
Deep learning such as convolutional neural networks which can better account for adjacency in samples may benefit from the MICE approach to imputation.
深度学习，如卷积神经网络，可以更好地考虑样本的邻近性，可能会从MICE方法的插补中受益。
Indeed, most ML methods cannot handle missing data so iterative imputation may improve these models.
实际上，大多数ML方法无法处理缺失数据，因此迭代插补可能会改善这些模型。
