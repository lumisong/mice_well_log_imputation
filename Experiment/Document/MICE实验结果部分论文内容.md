# 4. Results 结果

In this study, the models and predicted values were evaluated twice.
在本研究中，模型和预测值被两次评估。
First via a training set, where 30% of values were reserved for validation, and subsequently by using a test set containing a selection of blind wells.
首先通过一个训练集，其中30%的值被保留用于验证，然后通过包含一系列盲井的测试集。
The blind well test is designed to test the robustness of the imputation approach when generalised to a logical data subset absent during training.
盲井测试旨在测试当泛化到训练期间缺失的逻辑数据子集时，插补方法的鲁棒性。
Imputation models were then tested against varying degrees of input sparsity(retraining for each level of sparseness)ranging from 10 to 90%. Sparsity to the original data set was introduced randomly to all input features.
随后，针对不同程度的输入稀疏性（对每个稀疏级别重新训练）进行了插补模型的测试，范围从10%到90%。稀疏性是随机引入到所有输入特征中的。

## 4.1.Volve log imputation Volve测井插补

Imputation of the logs has been performed with seven different prediction approaches (Table 2).
使用七种不同的预测方法进行了测井的插补（表2）。
Four of these use the MICE algorithm and impute features with ascending order of missing values (MICE-BRR, MICE-O-BRR, MICE-KNN, MICE-O-KNN and MICE-A-GBT) and one model uses a random imputation order each iteration (MICE-R-GBT).
其中四种使用MICE算法，并按缺失值的升序对特征进行插补（MICE-BRR、MICE-O-BRR、MICE-KNN、MICE-O-KNN和MICE-A-GBT），一个模型在每次迭代中使用随机插补顺序（MICE-R-GBT）。
Changing the order of imputation is designed to test assertions by Murray (2018) that randomisation between iterations can improve sampling statistics and bias.
改变插补顺序旨在测试Murray（2018）的断言，即迭代间的随机化可以改善抽样统计和偏差。

## 4.1. Volve测井插补

For a comparison against the MICE tests we also perform direct imputation using GBT (D-GBT model).
为了与MICE测试进行比较，我们还使用GBT（D-GBT模型）进行直接插补。
This is not possible for BRR and KNN models because they cannot handle prediction when the input features are incomplete.
对于BRR和KNN模型，这是不可能的，因为它们不能在输入特征不完整时进行预测。
Instead, we perform imputation using just a single pass of MICE (one imputation model per input feature, MICE-OBRR and MICE-O-KNN).
相反，我们使用MICE的单次通过（每个输入特征一个插补模型，MICE-OBRR和MICE-O-KNN）进行插补。
All applications of MICE utilise the mean feature value as the initial guess for missing values.
所有MICE的应用都使用特征的平均值作为缺失值的初始猜测。

To evaluate the models, we compute five metrics to quantify the accuracy and bias of the predictions (Buitinck et al., 2013).
为了评估模型，我们计算了五个指标来量化预测的准确性和偏差（Buitinck等，2013）。
The corre lation between the known test values and the imputed results is calculated using Pearson’s R2 factor which returns values ranging from 0 to 1, where 1 is a perfect linear correlation.
使用Pearson的R²系数计算已知测试值和插补结果之间的相关性，其值范围从0到1，其中1表示完美的线性相关。
Explained variance similarly measures the correlation between data sets on a range from 0 to 1.
解释方差类似地测量数据集之间的相关性，范围也是从0到1。
The maximum error, mean absolute error (MAE) and mean square error (MSE) all relate information regarding the magnitude and distribution of the predictions.
最大误差、平均绝对误差（MAE）和均方误差（MSE）都与预测的幅度和分布有关。
The error metrics range from zero to infinity with smaller values indicating better performance.
误差指标的范围从零到无穷大，较小的值表示更好的性能。

Initially we consider the results from the validationsub-set (Table 3).
我们首先考虑验证子集的结果（表3）。
For Volve, the explained variance and R2 values are approximately ≥ 0.9 except, for the BRR modelwhich performed poorly as compared to KNN and GBT models.
对于Volve，解释方差和R²值大约≥0.9，除了与KNN和GBT模型相比表现不佳的BRR模型。
Error rates exhibited similar trends with KNN and GBT models consistently performing well.
误差率展示了类似的趋势，KNN和GBT模型的性能一直很好。

Compared to the baseline single imputation models (Once), BRR and KNN showed only small improvements or similar performance suggesting that MICE does not significantly improve the result over direct imputation.
与基线单次插补模型（Once）相比，BRR和KNN只显示出小幅改进或类似的性能，这表明MICE并没有显著改善直接插补的结果。
The exception was for the RHOB log imputations where performance degraded, perhaps due to over-fitting which is visible as reduced diversity in the response (Fig. 4).
例外是RHOB测井插补，性能下降，可能是由于过拟合导致的响应多样性降低（图4）。
The D-GBT model showed similar or slightly worse performance than MICE approaches.
D-GBT模型的性能与MICE方法类似或略差。

Metrics calculated using the test data were significantly lower (Table 4).
使用测试数据计算的指标显著降低（表4）。
The explained variance and R2 metrics dropped to between 0.6 and 0.8.
解释方差和R²指标下降到0.6到0.8之间。
Overall, error metrics also degraded increasing by 200–300%.
总体而言，误差指标也恶化，增加了200-300%。
As the test data was removed randomly, this suggests that there may be a degree of over-fitting to the training data.
由于测试数据是随机移除的，这表明可能存在对训练数据过拟合的程度。

In Fig. 3 we explore the link between geological zones, which define packages of rock with similar properties (denoted by color).
在图3中，我们探讨了地质带之间的联系，这些地质带定义了具有类似属性的岩石包（用颜色表示）。
The correlation of the input and predicted logs for three distinct model types;
绘制了三种不同模型类型输入和预测测井之间的相关性；

MICE-R-GBT, MICE-KNN and MICE-BRR are plotted.
MICE-R-GBT、MICE-KNN和MICE-BRR被绘制。
The MICE-R-GBT results tend to show the tightest correlation around the 1-1 line (per fect prediction).
MICE-R-GBT的结果倾向于在1-1线（完美预测）周围显示最紧密的相关性。
This is particularly true for the high slowness values in DTS.
特别是对于DTS的高慢度值，这一点尤其如此。
The MICE-BRR and MICE-KNN models both appear to underpredict DTS slowness at high values as well as RHOB in specific zones.
MICE-BRR和MICE-KNN模型在高值时都似乎低估了DTS的慢度以及特定区域的RHOB。
Overall, the D-GBT approach was the best imputeron the test data set for the three logs comingfirst in 10 out of 15 metrics.
总体而言，D-GBT方法在测试数据集上是三个测井中最好的插补器，在15个指标中首次获得10个。
This conclusion is drawn by comparing the scores of the D-GBTapproach in a relativesense to other models in Table 4.
这个结论是通过将D-GBT方法的分数与表4中其他模型的相对表现进行比较得出的。

## 4.2. olve qualitative analysis Volve定性分析

A qualitative analysis is used to gauge the suitability of the imputed results.
定性分析用于衡量插补结果的适用性。
Metrics provide a quantitative view of the data match but the imputed and predicted values must be assessedfor their acceptabilityby a geoscience professional.
指标提供了数据匹配的量化视图，但必须由地球科学专业人士评估插补和预测值的可接受性。
The BRR model results (Fig. 4) show some interestingtrends.
BRR模型结果（图4）显示了一些有趣的趋势。
The DT log appears to be well matched even where there are a high number of missing feature samples (between 10,000 and 15,000).
即使在缺失特征样本数量较高的情况下（在10,000到15,000之间），DT测井似乎匹配得很好。
Where the values for DT are high however,BRR appears to greatly over predict the DT log.
然而，当DT值较高时，BRR似乎大大高估了DT测井。
A limited number of DTS values for testing were available but where they exist, the MICE-BRR model seems to consistently under predict the slowness.
可用于测试的DTS值有限，但在存在的地方，MICE-BRR模型似乎一致地低估了慢度。
There may be some bias from the other wells used for training against these samples.
可能存在一些来自用于训练的其他井的偏差。
The RHOB predictions appear overly smooth compared with the known values, and they become inaccurate where the PEF and DRHO logs are absent.
与已知值相比，RHOB预测似乎过于平滑，并且在PEF和DRHO测井缺失的情况下变得不准确。

The MICE-KNN model (Fig. 4) matches the low frequency trends in the data but appears more prone to noise overall than the other models.
MICE-KNN模型（图4）与数据中的低频趋势匹配，但总体上比其他模型更容易受到噪声的影响。
The MICE-KNN model also returned no extrema beyond the models known values due to its averaging approach.
由于其平均处理方法，MICE-KNN模型也没有返回超出已知值的极值。
This is preferable to empirical approaches where trends may be extrapolated to non-physical values.
与经验方法相比，这更可取，经验方法可能会将趋势外推到非物理值。
Compared with MICE-BRR, the MICE-KNN model better honours the known RHOB values without over smoothing but still struggles to eliminate the bias where PEF and DRHO are missing.
与MICE-BRR相比，MICE-KNN模型更好地尊重已知的RHOB值，没有过度平滑，但仍然难以消除PEF和DRHO缺失时的偏差。

Both of the MICE models (MICE-A-GBT, MICE-R-GBT) perform well, overall the predictions are superior to MICE-BRR and MICE-KNN.
两种MICE模型（MICE-A-GBT、MICE-R-GBT）总体上表现良好，预测结果优于MICE-BRR和MICE-KNN。
The presence of the RHOB bias when missing DRHO and PEF suggest an inherent limitation between the available input features and the output.
当DRHO和PEF缺失时，RHOB偏差的存在表明了可用输入特征与输出之间的固有限制。

Although the D-GBT model performed well in the metricstest we can build an appreciation for the limits of the method when analysing the qualitative results.
尽管D-GBT模型在指标测试中表现良好，但通过分析定性结果，我们可以了解该方法的局限性。
Where the deliberate absence of any elastic values has been introduced between samples 25,000 and 31,000 (Fig. 4) the quality of the prediction begins to break down for both RHOB and especially for DTS.
在样本25,000到31,000之间故意引入任何弹性值缺失的情况下（图4），RHOB和特别是DTS的预测质量开始破裂。
The MICE implementation of GBT tends to outperform D-GBT in these situations where directly imputing for DTS from non-elastic logs is more difficult.
在这些情况下，从非弹性测井直接插补DTS更为困难，MICE对GBT的实现往往优于D-GBT。
It appears that sequential imputation tends to improve the overall prediction result in these extreme cases of many missing features.
看来，序列插补倾向于在这些许多特征缺失的极端情况下改善总体预测结果。
DTS for example with D-GBT imputation has a mean squared error of 0.36 in this specific test zone vs 0.06 for MICEGBT.
例如，对于DTS，D-GBT插补在这个特定测试区域的均方误差为0.36，而MICE-GBT为0.06。
The results for RHOB are less convincing, 0.25 and 0.23 but the direct method can rely upon logs bettersuited to predicting RHOB which are available.
对于RHOB的结果不那么令人信服，分别为0.25和0.23，但直接方法可以依赖于更适合预测RHOB的测井。
This suggests that MICE is able to first predict RHOB before using the RHOB prediction to predict DTS. The sequential prediction from MICE in this case significantly improves the final result.
这表明MICE能够首先预测RHOB，然后使用RHOB预测来预测DTS。在这种情况下，MICE的序列预测显著改善了最终结果。

## 4.3. volve log imputation error with increasing sparsity of input 输入稀疏度增加时的Volve测井插补误差

A key challenge to accurate imputation of well logs is the sparsity of the input (Lopes and Jorge, 2018).
准确插补井下测井的一个关键挑战是输入的稀疏性（Lopes和Jorge，2018）。
Sufficient training data is required to develop, calibrate and test a model.
需要足够的训练数据来开发、校准和测试模型。
In this section we test the capabilities of the imputation models as sparseness is gradually increased in a random fashion to the input features.
在本节中，我们测试了随着输入特征的稀疏度以随机方式逐渐增加时，插补模型的能力。

As sparsity increases there is an identifiable decrease in accuracy for all predictors.
随着稀疏度的增加，所有预测器的准确性都有可识别的下降。
The change is more systematic when measuring the validation results as compared with the test results (Fig. 5).
与测试结果相比，测量验证结果的变化更为系统化（图5）。
Also, the results for MICE imputers are very similar to the baseline MICE-O or direct approaches.
此外，MICE插补器的结果与基线MICE-O或直接方法非常相似。
BRR continues to under perform reaching a critical point of failureat a sparsity fraction of 0.5.
BRR继续表现不佳，达到0.5的稀疏分数时达到临界失败点。
The failure point for the other models tested appears at a sparsity fraction closer to 0.7.
其他测试模型的失败点出现在更接近0.7的稀疏分数。
Once BRR has failed, the metrics become unstable which may be related to the distribution of missing values within zones.
一旦BRR失败，指标变得不稳定，这可能与区域内缺失值的分布有关。
The results when applying to the test data set were slightlydifferent.
在高稀疏度时，MICE-X-GBT模型似乎优于D-GBT模型。
At high sparsity the MICE-X-GBT models appears to outperform the DGBT model. Breakdown of the models occurs around 50% sparsity.
模型在50%稀疏度时开始崩溃。

## 4.4. Force 2020 log imputation Force 2020测井插补

For the FORCE 2020 data set we follow the same imputation procedure of train, validate and test that was appliedto Volve data set. 
对于FORCE 2020数据集，我们遵循了与Volve数据集相同的训练、验证和测试的插补程序。
KNN type models were excluded from testing due to technical problems applying the method to the size of the data set.
由于将该方法应用到数据集大小时的技术问题，排除了KNN类型模型的测试。

All of the GBT type models again performed the best both in the training (Table 5) and testing (Table 6).
所有GBT类型模型在训练（表5）和测试（表6）中再次表现最佳。
The MICE-GBT models perform slightly better for DTS and much better for DT when compared with the D-GBT approach.
与D-GBT方法相比，MICE-GBT模型对DTS的表现略好，对DT的表现好得多。
This suggests there is a benefit to the chained imputation approach of MICE.
这表明MICE的链式插补方法有其优势。

A manual comparison of the predicted and test data via a qualitative analysis, shows a very good fit (Fig. 6).
通过定性分析手动比较预测和测试数据，显示出非常好的拟合（图6）。
In places, the BRR modelis prone to generating large noise spikes in the DTS log (betweensamples 19,000 and 42,000, Fig. 6).
在某些地方，BRR模型在DTS测井中倾向于生成大的噪声尖峰（在样本19,000到42,000之间，图6）。
The noise spikes don’t appear to be associated with any particular missing log and are perhaps due to a lack of training in a particular zone.
噪声尖峰似乎与任何特定的缺失测井无关，可能是由于在特定区域缺乏训练。
Comparatively, the GBT logs show a consistently good fit outperforming the other models and validating the quantitative metric results.
相比之下，GBT测井显示出一致的良好拟合，优于其他模型并验证了定量指标结果。
The fit to the long wavelength variations is particularly strong.
相对长波长变化的拟合尤为强。

An analysis of the relationship between metric based prediction performance and the number of samples available for training within each geological formation (Fig. 7) shows erratic trends in performance when the training sample size is small.
根据每个地质层系中可用于训练的样本数量，指标预测性能与关系的分析（图7）显示出当训练样本数量较小时，性能的趋势不稳定。
As the number of samples in a geological zone increases beyond approximately 20,000 points both MAE and MSE metrics trend towards a more stable value.
随着地质区域内样本数量增加超过大约20,000点，MAE和MSE指标趋向于更稳定的值。
Depending upon the complexity of the formation, this may represent an approximate lower limit to the number of samples need for accurate prediction models to be trained. Trends in R2 and explained variance are less clear.
根据地层的复杂性，这可能代表了训练准确预测模型所需样本数量的大致下限。R²和解释方差的趋势不太清楚。
