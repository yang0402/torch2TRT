import os
import re
from pathlib import Path

def find_absolute_paths(project_dir):
    # 正则表达式：匹配以 /home, /usr, /etc, /tmp 等开头的绝对路径
    path_pattern = re.compile(r'"/(home|usr|etc|tmp|opt|var|root)[^ ]*"')
    # 匹配以双引号开头，以 /home, /usr 等开头，后面跟任意非空格字符，直到双引号结束。

    # 存储结果：{文件名: [(行号, 路径)]}
    results = {}

    # 遍历项目目录中的所有 .py 文件
    for root, _, files in os.walk(project_dir):
        for file in files:
            if file.endswith('.py'):  # 只处理 Python 文件
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                    for line_num, line in enumerate(lines, 1):
                        # 跳过注释行
                        if line.strip().startswith('#'):
                            continue
                        matches = path_pattern.findall(line)
                        if matches:
                            if file_path not in results:
                                results[file_path] = []
                            for match in matches:
                                # match 是字符串，直接清理引号
                                cleaned_match = match.strip('"\' ')
                                results[file_path].append((line_num, cleaned_match))

    # 输出结果
    if not results:
        print("未找到绝对路径！")
    else:
        for file_path, paths in results.items():
            print(f"\n文件: {file_path}")
            for line_num, path in paths:
                print(f"  行 {line_num}: {path}")

if __name__ == "__main__":
    project_dir = "."  # 当前项目目录
    find_absolute_paths(project_dir)