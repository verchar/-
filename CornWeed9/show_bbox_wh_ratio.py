import os
import cv2
import matplotlib.pyplot as plt
from statistics import median
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

# 初始化存储宽高比的字典
class_bbox_ratio = {i: [] for i in range(len(class_names))}

# 缓存图片大小
image_size_cache = {}

def process_label_file(label_file, label_path, image_dir):
    """处理单个标签文件，提取边界框的宽高比"""
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

            # 计算宽高比
            if height_px != 0:  # 避免除以零
                ratio = width_px / height_px
                if class_index in class_bbox_ratio:
                    class_bbox_ratio[class_index].append(ratio)

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

# 定义子函数 show_bbox_wh_ratio
def show_bbox_wh_ratio(out_dir, fig_set, class_name, class_bbox_ratio):
    """Display the distribution map of category and bbox instance width and
    height ratio."""
    print('\n\nDrawing bbox_wh_ratio figure:')
    # Draw designs
    fig, ax = plt.subplots(
        figsize=(fig_set['figsize'][0], fig_set['figsize'][1]), dpi=300)

    # Set the position of the map and label on the x-axis
    positions = list(range(0, 6 * len(class_name), 6))
    ax.violinplot(
        list(class_bbox_ratio.values()),
        positions,
        showmeans=True,
        showmedians=True,
        widths=5)

    # Draw titles, labels and so on
    plt.xticks(rotation=fig_set['xticks_angle'])
    plt.ylabel('Ratio of width to height of bbox')
    plt.xlabel('Class name')
    plt.title('Width to height ratio distribution of class and bbox instances')

    # Draw the max, min and median of wide data in violin chart
    for i in range(len(class_bbox_ratio)):
        plt.text(
            positions[i],
            median(class_bbox_ratio[i]),
            f'{"%.2f" % median(class_bbox_ratio[i])}',
            ha='center',
            fontsize=fig_set['fontsize'])
        plt.text(
            positions[i],
            max(class_bbox_ratio[i]),
            f'{"%.2f" % max(class_bbox_ratio[i])}',
            ha='center',
            fontsize=fig_set['fontsize'])
        plt.text(
            positions[i],
            min(class_bbox_ratio[i]),
            f'{"%.2f" % min(class_bbox_ratio[i])}',
            ha='center',
            fontsize=fig_set['fontsize'])

    # Set the position of the map and label on the x-axis
    plt.setp(ax, xticks=positions, xticklabels=class_name)

    # Save figure
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_name = fig_set['out_name']
    fig.savefig(
        f'{out_dir}/{out_name}_bbox_ratio.jpg',
        bbox_inches='tight',
        pad_inches=0.1)  # Save Image
    plt.close()
    print(f'End and save in {out_dir}/{out_name}_bbox_ratio.jpg')


# 定义输出目录和图形设置
out_dir = r"E:\CornWeed9\output"
fig_set = {
    'figsize': (12, 6),  # 图形大小
    'xticks_angle': 45,  # x 轴标签旋转角度
    'fontsize': 8,       # 字体大小
    'out_name': 'CornWeed9'  # 输出文件名
}

# 调用子函数显示分布图
show_bbox_wh_ratio(out_dir, fig_set, class_names, class_bbox_ratio)