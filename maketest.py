import random

# 打开文本文件
with open("input.txt", "r") as file:
    # 读取第一行并保存
    first_line = file.readline()

    lines = file.readlines()
    total_lines = len(lines)

    # 随机选择一个合适的起始行数
    start_line = random.randint(0, total_lines - 160)

    # 读取随机选择的行和连续的160行
    selected_lines = lines[start_line : start_line + 160]

    # 构建当前文件的文件名
    file_name = "test.txt"

    # 将第一行和选取的行保存到新的文件
    with open(file_name, "w") as output_file:
        # 写入第一行到新文件
        output_file.write(first_line)
        output_file.writelines(selected_lines)

    print(f"Test data file is generated: {file_name}")