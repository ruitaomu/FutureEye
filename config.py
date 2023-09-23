#训练集的数据文件总数(文件名后缀从1~TOTAL_SAMPLE_FILES)
TOTAL_SAMPLE_FILES = 5000

FEATURES_SET = ["Open", "Close", "Low", "High"]
N_STEPS = 20
FEATURE_OFFSET = 2

SAMPLE_FILE_PATTERN = "data/highlow-{}.txt"
RAW_DATA_FILE = "HEdata.csv"
TEST_FILE_NAME="test.txt"
MODEL_FILE_NAME="model.keras"