import os

# 指定图片文件夹路径
image_folder = "/home/ge107552201346/datasets/datatest/images/train"   # 修改文件夹位置  # # # # ## # # #

# 获取文件夹中的所有图片文件
image_files = [f for f in os.listdir(image_folder) if f.endswith((".jpg", ".png", ".jpeg", ".gif"))]

# 打开train.txt文件以写入模式
with open("train.txt", "w") as train_file:
    # 遍历图片文件并写入完整路径和文件名  # 修改文件名称  # # # #  ## # # # #
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        train_file.write(image_path + '\n')

print("train.txt 文件已生成。")   #  修改文件名称
