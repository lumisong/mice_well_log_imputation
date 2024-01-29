"""Load Volve Data for input to log_imputer
"""

import pandas as pd


def load_data():
    """
    加载数据并进行预处理。

    Returns:
        train (DataFrame): 训练数据集，不包含测试井。
        test (DataFrame): 测试数据集，只包含测试井。
        
    结构：
    1. 辅助部分
    源数据的数值型到字符型的映射，以及列名的重命名。作为可视化部分内容。辅助理解数据。
    zone_map：映射字典，将ZONE列的int值映射为ZONE_NAME列的str值。
    col_rename_map：别名系统，将源数据中的列名重命名为更通用的列名。
    2. 加载部分
    3. 预处理部分
    预处理分为两部分，一部分是筛选，一部分是类型转换。（还有别名等辅助部分）
    4. 测试井人工指定，数据划分部分
    """
    
    # The zone map can convert the ZONE log of ints to named Zones.
    zone_map = {
        0: "Seabed",
        1: "NORDLAND",
        2: "Utsira",
        3: "HORDALAND",
        4: "Ty",
        5: "SHETLAND",
        6: "Ekofisk",
        7: "Hod",
        8: "Draupne",
        9: "Heather Shale",
        10: "Heather Sand",
        11: "Hugin C",
        12: "Hugin B3",
        13: "Hugin B2",
        14: "Hugin B1",
        15: "Hugin A",
        16: "Sleipner",
        17: "Skagerrak",
        18: "Smith Bank",
    }
    # 源数据要求：ZONE列为int类型
    # zone map映射，目的是将ZONE列的int值映射为ZONE_NAME列的str值，如0映射为Seabed
    # 后续代码中，使用映射内容进行热力图绘制，展示了不同ZONE的特征缺失分布情况。

    # Lets rename logs in the HDF5 input to more common terms.
    col_rename_map = {
        "ZONE_NO": "ZONE",
        "DTE": "DT",
        "DTSE": "DTS",
        "DRHOE": "DRHO",
        "GRE": "GR",
        "NPHIE": "NPHI",
        "PEFE": "PEF",
        "RHOBE": "RHOB",
        "RME": "RM",
        "RSE": "RS",
        "RDE": "RD",
        "WELL": "WELL_ID",
    }
    # 别名系统

    # 加载hdf5文件，重命名列名，筛选ZONE>=4的数据
    data = pd.read_hdf("data/volve_ml_logs.hdf5").rename(col_rename_map, axis=1)
    # look at deeper zones -> shallow zones poorly sampled/not of interest
    # 查看更深的地层，因为浅层地层采样不足或者不感兴趣。
    # query()方法，筛选ZONE>=4的数据，即深层地层数据。得到的数据经过了筛选，只有深层地层数据。
    # （）内的内容为筛选条件，即ZONE>=4，使用字符串形式表示。
    data = data.query("ZONE>=4")
    data["ZONE"] = data["ZONE"].astype(int) # 将ZONE列的数据类型转换为int，那么源类型是什么呢？
    # data["ZONE_NAME"] = data["ZONE"].map(zone_map)

    # 测试井人工指定
    test_wells = ["F-4", "F-12", "F-1", "F-15D"]
    train = data[~data.WELL_ID.isin(test_wells)].copy()
    test = data[data.WELL_ID.isin(test_wells)].copy()

    return train, test
