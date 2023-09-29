import random
import config
import os

data_subdir = config.TRAIN_DATA_SUBDIR
settings = config.load_settings()

# 打开文本文件
with open(settings["RAW_DATA_FILE"], "r") as file:
    # 读取第一行并保存
    first_line = file.readline()

    lines = file.readlines()
    total_lines = len(lines)

    # 检查当前目录下是否有名为"data"的子目录
    if not os.path.exists(data_subdir):
        # 如果不存在，则创建"data"子目录
        os.makedirs(data_subdir)
    else:
        # 如果存在，则清空该目录下的所有文件
        file_list = os.listdir(data_subdir)
        for file_name in file_list:
            file_path = os.path.join(data_subdir, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)

    file_pattern = settings["SAMPLE_FILE_PATTERN"]
    # 设置要保存的文件的数量
    num_files = settings["TOTAL_SAMPLE_FILES"]
    from_file_num = 1

    for i in range(from_file_num, from_file_num + num_files):
        # 随机选择一个合适的起始行数
        start_line = random.randint(0, total_lines - 160)

        # 读取随机选择的行和连续的160行
        selected_lines = lines[start_line : start_line + 160]

        # 构建当前文件的文件名
        file_name = file_pattern.format(i)
        file_name = os.path.join(data_subdir, file_name)

        # 将第一行和选取的行保存到新的文件
        with open(file_name, "w") as output_file:
            # 写入第一行到新文件
            output_file.write(first_line)
            output_file.writelines(selected_lines)


    print(f"{num_files} sample files generated.")