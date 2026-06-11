import os
import csv
import argparse
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--indir', type=str, required=True, help='Path to the directory containing YOLO format .txt files')
    parser.add_argument('-l', '--imgdir', type=str, required=True, help='Path to the directory containing images')
    parser.add_argument('-o', '--outcsv', type=str, default='dataset.csv', help='Output CSV file')
    args = parser.parse_args()
    return args

def get_file_list(directory, postfix):
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(postfix)]

def get_image_size(image_path):
    with Image.open(image_path) as img:
        return img.size  # 返回 (width, height)

def convert_yolo_to_csv(txt_files, img_dir, out_csv, class_map):
    with open(out_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image_path', 'xmin', 'ymin', 'xmax', 'ymax', 'class'])

        for txt_file in txt_files:
            img_file = os.path.splitext(os.path.basename(txt_file))[0] + '.jpg'
            img_path = os.path.join(img_dir, img_file)
            
            image_width, image_height = get_image_size(img_path)

            with open(txt_file, 'r') as f:
                lines = f.readlines()

            for line in lines:
                cls_id, x_center, y_center, width, height = map(float, line.strip().split())
                cls = class_map[int(cls_id)]
                
                # 转换为整数坐标
                xmin = round((x_center - width / 2) * image_width)
                xmax = round((x_center + width / 2) * image_width)
                ymin = round((y_center - height / 2) * image_height)
                ymax = round((y_center + height / 2) * image_height)
                
                writer.writerow([img_path, xmin, ymin, xmax, ymax, cls])

if __name__ == "__main__":
    args = parse_args()
    txt_files = get_file_list(args.indir, '.txt')
    
    # 定义类别映射
    class_map = {
        0: 'Amaranthus',
        1: 'Chenopodium',
        2: 'Solanum nigrum',
        3: 'Alhagi sparsifolia',
        4: 'Purslane',
        5: 'Convolvulus'
    }

    convert_yolo_to_csv(txt_files, args.imgdir, args.outcsv, class_map)



# CUDA_VISIBLE_DEVICES=6 python yolocsv.py --indir /home/ge107552201346/datasets/oursdatasets/labels/train --imgdir /home/ge107552201346/datasets/oursdatasets/images/train --outcsv train.csv
# CUDA_VISIBLE_DEVICES=6 python yolocsv.py --indir /home/ge107552201346/datasets/oursdatasets/labels/val --imgdir /home/ge107552201346/datasets/oursdatasets/images/val --outcsv val.csv
# CUDA_VISIBLE_DEVICES=6 python yolocsv.py --indir /home/ge107552201346/datasets/oursdatasets/labels/test --imgdir /home/ge107552201346/datasets/oursdatasets/images/test --outcsv test.csv