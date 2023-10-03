import os
import pandas as pd

# 指定目录路径
directory = 'testdata'

# 获取目录中所有.csv文件的路径
csv_files = [file for file in os.listdir(directory) if file.endswith('.csv')]

# 创建一个空的DataFrame
df = pd.DataFrame()

# 读取每个csv文件并将其内容添加到DataFrame中
for file in csv_files:
    file_path = os.path.join(directory, file)
    temp_df = pd.read_csv(file_path)
    df = pd.concat([df, temp_df], ignore_index=True)

# 根据date_time字段进行排序
df.sort_values(by='date_time', inplace=True)

# 将DataFrame保存为新的csv文件
output_file = 'mergedoutput.csv'
df.to_csv(output_file, index=False)

print("文件已成功导入到DataFrame并排序，结果已保存到output.csv文件中。")