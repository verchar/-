import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from concurrent.futures import ThreadPoolExecutor

# 定义类别名称
class_names = [
    "Amaranthus", "Chenopodium", "Solanum nigrum", "Alhagi sparsifolia",
    "Purslane", "Convolvulus", "Setaria", "Amaranthus polygonoides", "Corn"
]

# 定义图片和标签路径
image_paths = {
    "train": r"E:\CornWeed9\images\train",
    "val": r"E:\CornWeed9\images\val",
    "test": r"E:\CornWeed9\images\test"
}

label_paths = {
    "train": r"E:\CornWeed9\labels\train",
    "val": r"E:\CornWeed9\labels\val",
    "test": r"E:\CornWeed9\labels\test"
}

# 定义面积规则（小、中、大、超大）
area_rule = [0, 32**2, 96**2, 256**2, float('inf')]  # 面积阈值

# 初始化存储面积的字典
bbox_area_num = {i: [0] * (len(area_rule) - 1) for i in range(len(class_names))}

# 缓存图片大小
image_size_cache = {}

def process_label_file(label_file, label_path, image_dir):
    """处理单个标签文件，统计边界框的面积分布"""
    # 获取对应的图片文件
    image_file = label_file.replace('.txt', '.jpg')  # 假设图片是jpg格式
    image_path = os.path.join(image_dir, image_file)
    
    # 读取图片的宽度和高度（使用缓存）
    if image_path in image_size_cache:
        img_width, img_height = image_size_cache[image_path]
    else:
        if os.path.exists(image_path):
            image = cv2.imread(image_path)
            img_height, img_width, _ = image.shape
            image_size_cache[image_path] = (img_width, img_height)
        else:
            print(f"Warning: Image {image_path} not found. Skipping.")
            return

    # 读取标签文件
    with open(os.path.join(label_path, label_file), 'r') as file:
        for line in file:
            parts = line.split()
            class_index = int(parts[0])  # 获取类别索引
            width_norm = float(parts[3])  # 归一化的宽度
            height_norm = float(parts[4])  # 归一化的高度

            # 转换为实际像素值
            width_px = width_norm * img_width
            height_px = height_norm * img_height

            # 计算面积
            area = width_px * height_px

            # 根据面积规则统计
            for i in range(len(area_rule) - 1):
                if area_rule[i] <= area < area_rule[i + 1]:
                    bbox_area_num[class_index][i] += 1
                    break

# 使用多线程处理
with ThreadPoolExecutor(max_workers=4) as executor:
    for split, label_path in label_paths.items():
        image_dir = image_paths[split]
        futures = [
            executor.submit(process_label_file, label_file, label_path, image_dir)
            for label_file in os.listdir(label_path)
        ]
        for future in futures:
            future.result()  # 等待所有任务完成

# 定义子函数 show_bbox_area
def show_bbox_area(out_dir, fig_set, area_rule, class_name, bbox_area_num):
    """Display the distribution map of category and bbox instance area based on
    the rules of large, medium and small objects."""
    print('\n\nDrawing bbox_area figure:')
    # Set the direct distance of each label and the width of each histogram
    # Set the required labels and colors
    positions = np.arange(0, 2 * len(class_name), 2)
    width = 0.4
    labels = ['Small', 'Medium', 'Large', 'Huge']
    colors = ['#438675', '#F7B469', '#6BA6DA', '#913221']

    # Draw designs
    fig = plt.figure(
        figsize=(fig_set['figsize'][0], fig_set['figsize'][1]), dpi=300)
    for i in range(len(area_rule) - 1):
        area_num = [bbox_area_num[idx][i] for idx in range(len(class_name))]
        plt.bar(
            positions + width * i,
            area_num,
            width,
            label=labels[i],
            color=colors[i])
        for idx, (x, y) in enumerate(zip(positions.tolist(), area_num)):
            plt.text(
                x + width * i,
                y,
                y,
                ha='center',
                fontsize=fig_set['fontsize'] - 1)

    # Draw titles, labels and so on
    plt.xticks(rotation=fig_set['xticks_angle'])
    plt.xticks(positions + width * ((len(area_rule) - 2) / 2), class_name)
    plt.ylabel('Class Area')
    plt.xlabel('Class Name')
    plt.title(
        'Area and number of large, medium and small objects of each class')

    # Set and Draw Legend
    patches = [
        mpatches.Patch(color=colors[i], label=f'{labels[i]:s}')
        for i in range(len(area_rule) - 1)
    ]
    ax = plt.gca()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.8])
    ax.legend(loc='upper center', handles=patches, ncol=len(area_rule) - 1)

    # Save figure
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_name = fig_set['out_name']
    fig.savefig(
        f'{out_dir}/{out_name}_bbox_area.jpg',
        bbox_inches='tight',
        pad_inches=0.1)  # Save Image
    plt.close()
    print(f'End and save in {out_dir}/{out_name}_bbox_area.jpg')


# 定义输出目录和图形设置
out_dir = r"E:\CornWeed9\output"
fig_set = {
    'figsize': (12, 6),  # 图形大小
    'xticks_angle': 45,  # x 轴标签旋转角度
    'fontsize': 8,       # 字体大小
    'out_name': 'CornWeed9'  # 输出文件名
}

# 调用子函数显示分布图
show_bbox_area(out_dir, fig_set, area_rule, class_names, bbox_area_num)