import os
from collections import defaultdict

# 定义类别名称和索引的映射
class_mapping = {
    'Amaranthus': 0,
    'Chenopodium': 1,
    'Solanum nigrum': 2,
    'Alhagi sparsifolia': 3,
    'Purslane': 4,
    'Convolvulus': 5,
    'Setaria': 6,
    'Amaranthus polygonoides': 7,
    'Corn': 8
}

# 反转映射，从索引到类别名称
index_to_class = {v: k for k, v in class_mapping.items()}

def count_instances(folder_path):
    # 初始化一个字典来统计每个类别的实例数量
    instance_count = defaultdict(int)

    # 遍历文件夹中的所有.txt文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                for line in file:
                    # 提取类别索引（YOLO格式的第一个值）
                    class_index = int(line.strip().split()[0])
                    # 统计类别实例数量
                    instance_count[class_index] += 1

    # 打印统计结果
    print("类别实例统计：")
    for index, count in instance_count.items():
        class_name = index_to_class.get(index, f"未知类别 ({index})")
        print(f"{class_name}: {count} 个实例")

# 使用示例
folder_path = r"E:\datasets\labels\test"  # 替换为你的标签文件夹路径
count_instances(folder_path)