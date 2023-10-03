import numpy as np
import pandas as pd
import config

def make(year, month, date):
    settings = config.load_settings()
    df = pd.read_csv(settings["PREDICT_SOURCE_FILE"])

    # 将'Date'列转换为日期时间类型
    df["Date"] = pd.to_datetime(df["Date"])

    # 指定时间范围
    start_date = f"{year}-{month}-{date} 09:30:00"
    end_date = f"{year}-{month}-{date} 16:00:00"

    # 选择指定时间范围内的数据行
    selected_rows = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

    # 打印选定的数据行
    print(selected_rows)

    file_name = settings["TEST_FILE_NAME"]

    selected_rows.to_csv(file_name, index=False)
    print(f"Test data file is generated: {file_name}")
    return selected_rows
    
    
if __name__ == "__main__":
    make(2021,3,20)