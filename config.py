import json

TRAIN_DATA_SUBDIR = "data"

#训练集的数据文件总数(文件名后缀从1~TOTAL_SAMPLE_FILES)
# 默认参数值
DEFAULT_SETTINGS = {
    "TOTAL_SAMPLE_FILES": 5000,
    "FEATURES_SET": ["Open", "Close", "Low", "High"],
    "N_STEPS": 20,
    "FEATURE_OFFSET": 2,
    "AUTO_MARK": True,
    "MODEL_TYPE": "SimpleRNN",
    "EPOCHS": 35,
    "BATCH_SIZE": 32,
    "SAMPLE_FILE_PATTERN": "highlow-{}.txt",
    "RAW_DATA_FILE": "HEdata.csv",
    "TEST_FILE_NAME": "test.txt",
    "MODEL_FILE_NAME": "model.keras",
    "FLOATING_POINT_ADJUSTMENT": True,
    "PREDICT_SOURCE_FILE": "HEdata.csv",
    "TEST_YEAR": 2023,
    "TEST_MONTH": 8,
    "TEST_DATE": 8
}

# 加载保存的设置
def load_settings():
    try:
        with open("settings.json", "r") as file:
            settings = json.load(file)
    except FileNotFoundError:
        settings = DEFAULT_SETTINGS
    return settings

# 保存设置到JSON文件
def save_settings(settings):
    with open("settings.json", "w") as file:
        json.dump(settings, file)