import numpy as np
import pandas as pd
import config

df = pd.read_csv(config.RAW_DATA_FILE)

# 将'Date'列转换为日期时间类型
df["Date"] = pd.to_datetime(df["Date"])

# 指定时间范围
start_date = '2021-09-02 09:30:00'
end_date = '2021-09-02 16:00:00'

# 选择指定时间范围内的数据行
selected_rows = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

# 打印选定的数据行
print(selected_rows)

file_name = config.TEST_FILE_NAME

selected_rows.to_csv(file_name, index=False)
print(f"Test data file is generated: {file_name}")