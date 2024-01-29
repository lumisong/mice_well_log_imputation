""" 1. Log Imputer 导入库

- pandas：用于数据处理和分析的库。
- numpy：用于科学计算的库，提供了高性能的多维数组对象和工具。
- sklearn.preprocessing：scikit-learn库中的预处理模块，提供了数据预处理的功能，如标签编码和标准化。
- sklearn.experimental.enable_iterative_imputer：scikit-learn库中的实验性模块，启用了迭代式插补的功能。
- sklearn.impute.IterativeImputer：scikit-learn库中的插补类，用于迭代式插补缺失值。
- sklearn.metrics：scikit-learn库中的度量模块，提供了各种评估指标和度量方法。

"""

"""Log Imputation Class

Tony Hallam 2021
"""

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn import metrics

from collections import defaultdict

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn import metrics


""" Log Imputer Class 结构

- `__init__`：初始化函数，用于初始化类的实例。
- `encode`：编码函数，用于将logs编码为int。
- `decode`：解码函数，用于将logs解码为str。
- `scale`：标准化函数，用于对logs进行标准化和log10变换。
- `_test_data_prep`：测试数据准备函数，用于将测试数据中的一部分值设置为缺失值。
- `fit`：拟合函数，用于拟合插补模型。
- `predict`：预测函数，用于预测缺失值。
- `impute`：插补函数，用于插补缺失值。
- `_impute_training`：训练集插补函数，用于插补训练集中的缺失值。
- `_impute_test`：测试集插补函数，用于插补测试集中的缺失值。

"""

class LogImputer:
    def __init__(
        self,
        train,
        test,
        imputer,
        iterative=False,
        log10logs=None,
        imputer_kwargs=dict(),
    ):
        """ 
        初始化函数，用于初始化类的实例。
        
        1. Args:
        
        数据集：
        - train (DataFrame): 训练数据集，不包含测试井。
        - test (DataFrame): 测试数据集，只包含测试井。
        训练数据集和测试数据集的结构相同，包含以下列：特征列（logs）和目标列（target）。特征列中可能含有缺失值，数值型特征或者字符型特征，都可能有。
        
        插补模型：
        - imputer (class): 插补模型，如sklearn.impute.IterativeImputer。
        - iterative (bool): 是否使用迭代式插补，默认为False。
        
        数据预处理：
        - log10logs (list): 需要进行log10变换的logs，默认为None。
        
        插补模型参数：
        - imputer_kwargs (dict): 插补模型的参数，默认为空字典。
        
        2. 代码核心：
        
        - self.train (DataFrame): 训练数据集，不包含测试井。
        - self.test (DataFrame): 测试数据集，只包含测试井。
        - self.imputer (class): 插补模型，如sklearn.impute.IterativeImputer。
        - self.imputer_init (dict): 插补模型的参数，默认为空字典。
        - self.iterative (bool): 是否使用迭代式插补，默认为False。
        - self.log10logs (list): 需要进行log10变换的logs，默认为None。
        - self.encoders (dict): 编码器字典，用于存储编码器。
        - self.scalar (class): 标准化模型，如sklearn.preprocessing.StandardScaler。
        - self.imputation_train_impts (dict): 训练集插补模型字典，用于存储训练集插补模型。
        - self.imputation_train_dfs (dict): 训练集插补数据集字典，用于存储训练集插补数据集。
        
        """

        self.train = train
        self.test = test
        self.imputer = imputer
        self.imputer_init = imputer_kwargs
        self.iterative = iterative
        self.log10logs = log10logs
        self.encoders = dict()
        self.scalar = None
        self.imputation_train_impts = dict()
        self.imputation_train_dfs = dict()

    def encode(self, logs):
        """Encode logs to int.

        Args:
            logs ([type]): [description]
        """
        
        """ 
        编码函数，用于将logs编码为int。
        
        1. Args:
        
        - logs ([type]): [description]
        
        2. 代码核心：
        
        if isinstance(logs, str): 判断logs是否为str类型，如果是，则将logs转换为list类型。
        data = pd.concat([self.train, self.test])：将训练集和测试集合并。
        
        for log in logs: 遍历logs中的每一个log。
        self.encoders[log] = LabelEncoder()：创建编码器。
        self.encoders[log].fit(data[log])：对log进行编码。
        self.train[log] = self.encoders[log].transform(self.train[log])：对训练集中的log进行编码。
        
        总结：对训练集和测试集中的logs进行编码。
        
        输入的logs为str类型时，将logs转换为list类型。这个应该表示的是需要对哪些logs进行编码。
        
        """
        if isinstance(logs, str):
            logs = [logs]

        data = pd.concat([self.train, self.test])

        for log in logs:
            self.encoders[log] = LabelEncoder()
            self.encoders[log].fit(data[log])
            self.train[log] = self.encoders[log].transform(self.train[log])
            self.test[log] = self.encoders[log].transform(self.test[log])

    def decode(self):
        """Decode logs based unpon previous encodings."""
        
        """ 
        解码函数，用于将logs解码为str。
        """
        for log in self.encoders:
            self.train[log] = self.encoders[log].inverse_transform(self.train[log])
            self.test[log] = self.encoders[log].inverse_transform(self.test[log])

    def scale(self):
        """Standard scaling and log10 transform"""
        
        """ 
        标准化函数，用于对logs进行标准化和log10变换。
        
        self.log10logs 初始值为None，所以不会进行log10变换。
        
        当self.log10logs不为None时，对self.train和self.test中的logs进行log10变换。
        
        一般进行log10变换的logs为RM、RS、RD。
        
        而标准化数据是为了让数据符合标准正态分布，即均值为0，方差为1。
        """
        for log in self.log10logs:
            self.train[log] = np.log10(self.train[log])
            self.test[log] = np.log10(self.test[log])

        data = pd.concat([self.train, self.test])
        self.scalar = StandardScaler()
        self.scalar.fit(data)

        self.train.loc[:, :] = self.scalar.transform(self.train)
        self.test.loc[:, :] = self.scalar.transform(self.test)

    def _test_data_prep(self, data, test_target, set_to_nan=0.3):
        """This method sets set_to_nan fraction of the values to nan so we can measure the model accuracy."""
        
        """ 
        测试数据准备函数，用于将测试数据中的一部分值设置为缺失值。
        
        1. Args:
        
        - data (DataFrame): 
        - test_target ([type]): [description]
        - set_to_nan (float, optional): [description]. Defaults to 0.3.
        
        2. 代码核心：
        
        data = data.copy()：复制data。目的是不改变原始数据。
        sub = data.dropna(subset=[test_target])：删除data中test_target列中的缺失值。
        rand_set_mask = np.random.random(len(sub)) < set_to_nan：随机生成一个0-1之间的数，如果小于set_to_nan，则为True，否则为False。
        replace = sub.index[rand_set_mask]：将sub中的index赋值给replace。
        data.loc[replace, test_target] = np.nan：将data中的test_target列中的replace的值设置为缺失值。
        data["set_nan"] = False：创建set_nan列，初始值为False。
        data.loc[replace, "set_nan"] = True：将replace对应的set_nan列的值设置为True。
        data["was_nan"] = data[test_target].isna()：创建was_nan列，初始值为data[test_target].isna()。
        
        总结：将data中的test_target列中的一部分值设置为缺失值。
        """
        data = data.copy()
        # just get non-nan values for test target
        sub = data.dropna(subset=[test_target])
        # introduce random missing
        rand_set_mask = np.random.random(len(sub)) < set_to_nan
        replace = sub.index[rand_set_mask]
        # create flags, create and return new data
        data.loc[replace, test_target] = np.nan
        data["set_nan"] = False
        data.loc[replace, "set_nan"] = True
        data["was_nan"] = data[test_target].isna() 
        # print("Col, InputSize, Number of Nan, % NaN, Original Nan", "Training Size")
        # print(
        #     f"{j:>3}",
        #     f"{data.shape[0]:>10}",
        #     f"{replace.size:>14}",
        #     f"{100*np.sum(data.set_nan)/sub.shape[0]:>6.2f}",
        #     f"{np.sum(data.was_nan):>13}",
        #     f"{sub.shape[0]-replace.size:>13}",
        # )
        return data

    def fit(self, logs=None, target_logs=None, test_proportion=0.3, **kwargs):
        """Fit the imputer/s.

        Args:
            logs ([type], optional): [description]. Defaults to None.
        """
        if logs is None:
            logs = self.train.columns
            self.fitted_logs = logs
        if target_logs is None:
            target_logs = self.train.columns

        for key in target_logs:
            self.imputation_train_dfs[key] = self._test_data_prep(
                self.train, key, set_to_nan=test_proportion
            )
        # mice mode
        if self.iterative:
            for key in target_logs:
                self.imputation_train_impts[key] = IterativeImputer(
                    self.imputer(**self.imputer_init), **kwargs
                )
                self.imputation_train_impts[key].fit(
                    self.imputation_train_dfs[key][logs].copy()
                )
        # direct prediction mode (won't work for reg. that don't handle nans)
        else:
            for key in target_logs:
                self.imputation_train_impts[key] = self.imputer(**self.imputer_init)
                self.imputation_train_impts[key].fit(
                    self.imputation_train_dfs[key]
                    .dropna(subset=[key])
                    .loc[:, set(logs).difference((key,))],
                    self.imputation_train_dfs[key][key].dropna(),
                )

    def predict(self, predict_for="train"):
        """Prediction Mode - Not available for iterative imputer."""
        pass

    def impute(self, impute_for="train"):
        """Imputation Mode"""

        df = self.__getattribute__(impute_for)
        if self.iterative:
            imputed = {
                key: self.imputation_train_impts[key].transform(df[self.fitted_logs])
                for key in self.imputation_train_impts
            }

            output_df = imputed[tuple(imputed.keys())[0]]
            for key in imputed.keys():
                output_df[key] = imputed[key][key]

        else:
            predicted = {
                key: self.imputation_train_impts[key].predict(
                    df.loc[:, set(self.fitted_logs).difference((key,))]
                )
                for key in self.imputation_train_impts
            }
            imputed = {
                key: np.where(df[key].isna().values, ar, df[key].values)
                for key, ar in predicted.items()
            }

            output_df = df.copy()
            for key in imputed:
                output_df[key] = imputed[key]

        return output_df

    def _impute_training(self):

        if self.iterative:
            imputed = {}
            for key in self.imputation_train_impts:
                imputed[key] = self.imputation_train_dfs[key].copy()
                imputed[key].loc[:, self.fitted_logs] = self.imputation_train_impts[
                    key
                ].transform(self.imputation_train_dfs[key][self.fitted_logs])
        else:
            imputed = {}
            for key in self.imputation_train_dfs:
                fitted_logs = set(self.fitted_logs).difference((key,))
                pred = self.imputation_train_impts[key].predict(
                    self.imputation_train_dfs[key][fitted_logs]
                )
                imputed[key] = self.imputation_train_dfs[key].copy()
                mask = self.imputation_train_dfs[key][key].isna()
                imputed[key].loc[mask, key] = pred[mask]

        return imputed

    def _impute_test(self):

        if self.iterative:
            imputed = {}
            for key in self.imputation_train_impts:
                imputed[key] = self.test.copy()
                imputed[key][key] = np.nan
                imputed[key].loc[:, self.fitted_logs] = self.imputation_train_impts[
                    key
                ].transform(imputed[key][self.fitted_logs])
        else:
            imputed = {}
            for key in self.imputation_train_dfs:
                fitted_logs = set(self.fitted_logs).difference((key,))
                pred = self.imputation_train_impts[key].predict(self.test[fitted_logs])
                imputed[key] = self.test.copy()
                imputed[key].loc[:, key] = pred

        return imputed

    def score(self, score="train"):
        """Evaluate the models against the NANed data from the training set."""
        scores = defaultdict(dict)
        if score == "train":
            imputed_dfs = self._impute_training()
            df = self.train

        elif score == "test":
            imputed_dfs = self._impute_test()
            df = self.test
        else:
            raise ValueError(f"unknown score type: {score}")

        for key, d in imputed_dfs.items():
            if score == "train":
                mask = self.imputation_train_dfs[key].set_nan.values
            elif score == "test":
                mask = ~df[key].isna()
            truth = df.loc[mask, key].values
            test = d.loc[mask, key].values
            se = np.power((truth - test) / truth, 2)
            perc_error_score = np.nanmean(np.power(se, 0.5)) * 100.0
            er = dict(
                perc_error=perc_error_score,
                explained_var=metrics.explained_variance_score(truth, test),
                max_error=metrics.max_error(truth, test),
                mae=metrics.mean_absolute_error(truth, test),
                mse=metrics.mean_squared_error(truth, test),
                r2=metrics.r2_score(truth, test),
            )
            scores[key] = er

        return pd.DataFrame(scores).sort_index(axis=1).T.round(2)


if __name__ == "__main__":
    from lightgbm import LGBMRegressor

    from volve_loader import load_data

    min_leaf_search = [100, 200, 400, 800, 1600]
    max_depth_search = (5, 7, 9, 11)
    bagging_freq = [0, 2, 5, 10, 15]
    bagging_fraction = [1.0, 0.8, 0.6, 0.4, 0.2]

    scores_test = []
    scores_train = []
    scores_test_it = []
    scores_train_it = []
    parameters = []
    i = 0
    for mls in min_leaf_search:
        for mds in max_depth_search:
            for bfreq in bagging_freq:
                for bfrac in bagging_fraction:

                    parameters.append([i, mls, mds, bfreq, bfrac])
                    i += 1
                    train, test = load_data()

                    lgbasc = dict(
                        training_set=[
                            "DT",
                            "DTS",
                            "GR",
                            "NPHI",
                            "PEF",
                            "RHOB",
                            "RM",
                            "RS",
                            "RD",
                            "ZONE",
                        ],
                        estimator=LGBMRegressor,
                        imputer_init=dict(
                            random_state=456,
                            n_jobs=8,
                            max_depth=mds,
                            min_child_samples=mls,
                            bagging_fraction=bfrac,
                            bagging_freq=bfreq,
                        ),
                        kwargs=dict(
                            random_state=456,
                            max_iter=20,
                            tol=0.01,
                            imputation_order="ascending",
                        ),
                    )

                    config = lgbasc

                    itImp = LogImputer(
                        train,
                        test,
                        config["estimator"],
                        iterative=True,
                        log10logs=["RM", "RS", "RD"],
                        imputer_kwargs=config["imputer_init"],
                    )
                    itImp.encode(["WELL_ID"])
                    itImp.scale()
                    itImp.fit(
                        target_logs=["DT", "DTS", "RHOB"],
                    )

                    scores_train_it.append(itImp.score())
                    scores_test_it.append(itImp.score("test"))

                    ###########################################################
                    train, test = load_data()

                    Imp = LogImputer(
                        train,
                        test,
                        LGBMRegressor,
                        imputer_kwargs=dict(
                            random_state=456,
                            max_depth=mds,
                            min_child_samples=mls,
                            bagging_fraction=bfrac,
                            bagging_freq=bfreq,
                        ),
                        log10logs=["RM", "RD", "RS"],
                    )
                    Imp.encode(["WELL_ID"])
                    Imp.scale()
                    Imp.fit(
                        target_logs=["DT", "DTS", "RHOB"],
                    )

                    scores_train.append(Imp.score())
                    scores_test.append(Imp.score("test"))

    #    break
    parameters_df = pd.DataFrame(
        parameters, columns=["ind", "mls", "mds", "bfreq", "bfrac"]
    )
    scores_train_df = pd.concat({i: df for i, df in enumerate(scores_train)})
    scores_test_df = pd.concat({i: df for i, df in enumerate(scores_test)})
    scores_train_it_df = pd.concat({i: df for i, df in enumerate(scores_train_it)})
    scores_test_it_df = pd.concat({i: df for i, df in enumerate(scores_test_it)})

    parameters_df.to_csv("hypertuning_param2.csv")
    scores_train_df.to_csv("hypertuning_scores_train2.csv")
    scores_train_it_df.to_csv("hypertuning_scores_train_it2.csv")
    scores_test_df.to_csv("hypertuning_scores_test2.csv")
    scores_test_it_df.to_csv("hypertuning_scores_test_it2.csv")
