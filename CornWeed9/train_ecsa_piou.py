from ultralytics import YOLO

if __name__ == '__main__':
    # 依然加载包含 ECSA 模块的图纸
    model = YOLO('yolov8s_ecsa.yaml')
    
    model.train(
        data='CornWeed9.yaml',
        epochs=300,
        patience=50,
        batch=16,          
        imgsz=640,
        workers=4,         
        amp=True,          
        cache='disk',
        device=0,
        optimizer='SGD',   # 保持最稳的优化器
        project='runs/detect',
        name='v8s_ECSA_PIoU_train'  # <--- 保存在全新的文件夹，绝不覆盖之前的成果！
    )