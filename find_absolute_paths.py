import os
# 导入 os 模块，用于文件和目录操作，例如遍历目录和处理文件路径。

import re
# 导入 re 模块，用于正则表达式匹配，帮助识别代码中的绝对路径。

from pathlib import Path
# 从 pathlib 模块导入 Path 类，用于跨平台路径处理（虽然本脚本未直接使用 Path，但导入以备扩展）。

def find_absolute_paths(project_dir):
    # 定义函数 find_absolute_paths，用于在指定项目目录中查找绝对路径。
    # 参数 project_dir: 项目目录的路径（字符串），通常是当前目录（"."）。

    # 正则表达式：匹配以 /home, /usr, /etc, /tmp 等开头的绝对路径
    path_pattern = re.compile(r'"/(home|usr|etc|tmp|opt|var|root)[^/ ]*([/][^/ ]*)+/?(?:[^/ ]|$)')
    # 定义正则表达式 path_pattern，用于匹配绝对路径：
    # - `"`：匹配路径开头的双引号（假设路径被引号包裹）。
    # - `/(home|usr|etc|tmp|opt|var|root)`：匹配以 /home, /usr, /etc, /tmp, /opt, /var, /root 开头的路径。
    # - `[^/ ]*`：匹配非 / 和非空格的任意字符（组成路径的段）。
    # - `([/][^/ ]*)+`：匹配一个或多个由 / 分隔的路径段。
    # - `/?`：匹配路径末尾的可选 /。
    # - `(?:[^/ ]|$)`：确保路径后是空格或行尾，避免匹配不完整的路径。

    # 存储结果：{文件名: [(行号, 路径)]}
    results = {}
    # 创建空字典 results，用于存储查找结果，格式为 {文件路径: [(行号, 绝对路径), ...]}。

    # 遍历项目目录中的所有 .py 文件
    for root, _, files in os.walk(project_dir):
        # 使用 os.walk 递归遍历 project_dir 目录：
        # - root: 当前目录路径。
        # - _: 忽略子目录列表（使用 _ 表示未使用）。
        # - files: 当前目录中的文件列表。

        for file in files:
            # 遍历当前目录中的每个文件。

            if file.endswith('.py'):  # 只处理 Python 文件
                # 检查文件是否以 .py 结尾，只处理 Python 脚本文件。

                file_path = os.path.join(root, file)
                # 使用 os.path.join 构造文件的完整路径，结合 root 和文件名 file。

                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    # 打开文件 file_path，模式为只读（'r'），使用 UTF-8 编码，errors='ignore' 忽略编码错误。

                    lines = f.readlines()
                    # 读取文件所有行，存储为列表 lines，每行是一个字符串。

                    for line_num, line in enumerate(lines, 1):
                        # 遍历每行，enumerate 从 1 开始计数，提供行号（line_num）和行内容（line）。

                        # 跳过注释行
                        if line.strip().startswith('#'):
                            # 如果行去掉首尾空格后以 # 开头，说明是注释行，跳过以避免误匹配注释中的路径。

                            continue
                            # 跳过当前循环，处理下一行。

                        matches = path_pattern.findall(line)
                        # 使用正则表达式 path_pattern 在当前行中查找所有匹配的绝对路径，存储在 matches 列表中。

                        if matches:
                            # 如果找到匹配的路径（matches 非空）。

                            if file_path not in results:
                                # 如果当前文件路径 file_path 不在 results 字典中。

                                results[file_path] = []
                                # 为该文件路径初始化一个空列表，用于存储匹配的路径和行号。

                            for match in matches:
                                # 遍历正则表达式找到的每个匹配路径。

                                cleaned_match = match.strip('"\' ')
                                # 清理匹配的路径，去掉首尾的双引号、单引号和空格，得到干净的路径字符串。

                                results[file_path].append((line_num, cleaned_match))
                                # 将 (行号, 清理后的路径) 元组添加到 results 中对应文件路径的列表。

    # 输出结果
    if not results:
        # 如果 results 字典为空，说明没有找到任何绝对路径。

        print("未找到绝对路径！")
        # 输出提示信息，表示没有找到符合条件的绝对路径。

    else:
        # 如果 results 非空，说明找到了绝对路径。

        for file_path, paths in results.items():
            # 遍历 results 字典中的每个文件路径和对应的路径列表。

            print(f"\n文件: {file_path}")
            # 打印当前文件路径，前面加换行符以便于阅读。

            for line_num, path in paths:
                # 遍历该文件的每个匹配路径和行号。

                print(f"  行 {line_num}: {path}")
                # 打印行号和对应的绝对路径，格式为“行 X: 路径”。

if __name__ == "__main__":
    # 检查脚本是否作为主程序运行（而不是被导入）。

    project_dir = "."  # 当前项目目录
    # 设置项目目录为当前目录（"."），表示从脚本运行的目录开始搜索。

    find_absolute_paths(project_dir)
    # 调用 find_absolute_paths 函数，传入当前目录，开始搜索绝对路径。