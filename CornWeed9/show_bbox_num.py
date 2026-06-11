import os
import matplotlib.pyplot as plt

# 定义类别名称
class_names = [
    "Amaranthus", "Chenopodium", "Solanum nigrum", "Alhagi sparsifolia",
    "Purslane", "Convolvulus", "Setaria", "Amaranthus polygonoides", "Corn"
]

# 定义标签路径
label_paths = {
    "train": r"E:\CornWeed9\labels\train",
    "val": r"E:\CornWeed9\labels\val",
    "test": r"E:\CornWeed9\labels\test"
}

# 初始化类别计数器
class_counts = {i: 0 for i in range(len(class_names))}

# 遍历所有标签文件
for split, path in label_paths.items():
    for label_file in os.listdir(path):
        with open(os.path.join(path, label_file), 'r') as file:
            for line in file:
                class_index = int(line.split()[0])  # 获取类别索引
                if class_index in class_counts:
                    class_counts[class_index] += 1

# 打印每个类别的实例数量
for class_index, count in class_counts.items():
    print(f"{class_names[class_index]}: {count} instances")

# 绘制类别和 bbox 实例个数的分布图
plt.figure(figsize=(10, 6))
bars = plt.bar(class_names, class_counts.values(), color='skyblue')

# 在每个柱子上显示实例数量
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,  # x 坐标：柱子中心
        height,  # y 坐标：柱子顶部
        f'{int(height)}',  # 显示的文本
        ha='center',  # 水平对齐方式：居中
        va='bottom'  # 垂直对齐方式：底部
    )

plt.xlabel('Class name')
plt.ylabel('Number of Instances')
plt.title('CornWeed9')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()