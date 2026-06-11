import os
import random
import shutil
from collections import defaultdict

# 数据集路径
image_source_dir = r"C:\Users\25053\Desktop\corndataset\images"
label_source_dir = r"C:\Users\25053\Desktop\corndataset\labels\yolo"

# 划分后的数据集保存路径
train_image_dir = r"E:\datasets\images\train"
val_image_dir = r"E:\datasets\images\val"
test_image_dir = r"E:\datasets\images\test"

train_label_dir = r"E:\datasets\labels\train"
val_label_dir = r"E:\datasets\labels\val"
test_label_dir = r"E:\datasets\labels\test"

# 类别映射
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

# 划分比例
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# 创建目标文件夹
os.makedirs(train_image_dir, exist_ok=True)
os.makedirs(val_image_dir, exist_ok=True)
os.makedirs(test_image_dir, exist_ok=True)
os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)
os.makedirs(test_label_dir, exist_ok=True)

# 获取所有图片文件
image_files = [f for f in os.listdir(image_source_dir) if f.endswith('.jpg')]
random.shuffle(image_files)  # 随机打乱

# 统计每个类别的实例数量
class_count = defaultdict(int)
for image_file in image_files:
    label_file = os.path.join(label_source_dir, os.path.splitext(image_file)[0] + '.txt')
    if os.path.exists(label_file):
        with open(label_file, 'r') as f:
            for line in f:
                class_index = int(line.strip().split()[0])
                class_count[class_index] += 1

# 计算每个类别的划分数量
train_counts = {k: int(v * train_ratio) for k, v in class_count.items()}
val_counts = {k: int(v * val_ratio) for k, v in class_count.items()}
test_counts = {k: int(v * test_ratio) for k, v in class_count.items()}

# 初始化每个类别的计数器
current_train_counts = defaultdict(int)
current_val_counts = defaultdict(int)
current_test_counts = defaultdict(int)

# 划分数据集
for image_file in image_files:
    label_file = os.path.join(label_source_dir, os.path.splitext(image_file)[0] + '.txt')
    if not os.path.exists(label_file):
        continue

    # 统计当前图片中的类别
    with open(label_file, 'r') as f:
        classes_in_image = [int(line.strip().split()[0]) for line in f]

    # 计算当前图片对各类别实例数量的贡献
    train_needed = any(current_train_counts[cls] < train_counts[cls] for cls in classes_in_image)
    val_needed = any(current_val_counts[cls] < val_counts[cls] for cls in classes_in_image)
    test_needed = any(current_test_counts[cls] < test_counts[cls] for cls in classes_in_image)

    # 优先分配到最需要的集合
    if train_needed:
        dest_image_dir = train_image_dir
        dest_label_dir = train_label_dir
        for cls in classes_in_image:
            current_train_counts[cls] += 1
    elif val_needed:
        dest_image_dir = val_image_dir
        dest_label_dir = val_label_dir
        for cls in classes_in_image:
            current_val_counts[cls] += 1
    elif test_needed:
        dest_image_dir = test_image_dir
        dest_label_dir = test_label_dir
        for cls in classes_in_image:
            current_test_counts[cls] += 1
    else:
        # 如果所有集合的类别实例数量都已满足，则分配到训练集
        dest_image_dir = train_image_dir
        dest_label_dir = train_label_dir

    # 复制图片和标签文件
    shutil.copy(os.path.join(image_source_dir, image_file), os.path.join(dest_image_dir, image_file))
    shutil.copy(label_file, os.path.join(dest_label_dir, os.path.basename(label_file)))

# 统计训练集、验证集和测试集的类别实例数量
def count_instances_in_split(image_dir, label_dir):
    instance_count = defaultdict(int)
    for image_file in os.listdir(image_dir):
        label_file = os.path.join(label_dir, os.path.splitext(image_file)[0] + '.txt')
        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                for line in f:
                    class_index = int(line.strip().split()[0])
                    instance_count[class_index] += 1
    return instance_count

train_instance_count = count_instances_in_split(train_image_dir, train_label_dir)
val_instance_count = count_instances_in_split(val_image_dir, val_label_dir)
test_instance_count = count_instances_in_split(test_image_dir, test_label_dir)

# 打印结果
print("数据集划分完成！")
print(f"训练集图片数量: {len(os.listdir(train_image_dir))}")
print(f"验证集图片数量: {len(os.listdir(val_image_dir))}")
print(f"测试集图片数量: {len(os.listdir(test_image_dir))}")

print("\n训练集各类别实例数量:")
for cls, count in train_instance_count.items():
    print(f"{list(class_mapping.keys())[list(class_mapping.values()).index(cls)]}: {count}")

print("\n验证集各类别实例数量:")
for cls, count in val_instance_count.items():
    print(f"{list(class_mapping.keys())[list(class_mapping.values()).index(cls)]}: {count}")

print("\n测试集各类别实例数量:")
for cls, count in test_instance_count.items():
    print(f"{list(class_mapping.keys())[list(class_mapping.values()).index(cls)]}: {count}")