from ultralytics import YOLO

if __name__ == '__main__':
    print("开始转换基线模型...")
    # 注意：把下面括号里的单引号内的内容，替换成你刚刚右键复制的真实路径！
    model_base = YOLO(r'D:\Corn\cvnet_CornWeed9\CornWeed9\runs\detect\runs\detect\Corn_Weed_Paper\v8n_baseline\weights\best.pt')
    model_base.export(format='onnx', opset=11, simplify=True)
    
    print("开始转换本文改进模型...")
    # 同样的操作，去左边找到 v8n_ECSA_WIoU 里的 best.pt，右键复制路径替换到下面！
    model_ours = YOLO(r'D:\Corn\cvnet_CornWeed9\CornWeed9\runs\detect\runs\detect\Corn_Weed_Paper\v8n_ECSA_WIoU\weights\best.pt')
    model_ours.export(format='onnx', opset=11, simplify=True)
    
    print("全部转换完成！")