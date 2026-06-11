import os

def rename_xml_files(folder_path, start_index):
    # 获取文件夹下所有的.xml文件
    xml_files = [f for f in os.listdir(folder_path) if f.endswith('.xml')]
    # 对文件进行排序（如果需要按文件名顺序处理）
    xml_files.sort()

    # 重命名每一个文件
    for idx, xml_file in enumerate(xml_files):
        # 新文件名，使用4位数，不足前面补0，起始编号为start_index
        new_name = f"{str(start_index + idx).zfill(4)}.xml"
        # 源文件路径
        src = os.path.join(folder_path, xml_file)
        # 目标文件路径
        dst = os.path.join(folder_path, new_name)
        
        # 重命名文件
        os.rename(src, dst)
        print(f"Renamed: {xml_file} -> {new_name}")

# 使用示例
folder_path = r"C:\Users\25053\Desktop\corndataset\labels\pascal voc"  # 替换为你的文件夹路径
rename_xml_files(folder_path, start_index=1)  # 设置起始编号为1856