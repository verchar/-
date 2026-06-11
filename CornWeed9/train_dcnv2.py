from ultralytics import YOLO

if __name__ == '__main__':
    # 加载带有可变形卷积的新配置文件
    model = YOLO('yolov8_dcnv2.yaml')
    
    # 开始训练
    model.train(
        data='CornWeed9.yaml',
        epochs=300,
        patience=50,
        batch=16,          
        imgsz=640,
        workers=4,         
        amp=True,         # <--- 【关键修改】这里的 T 必须大写！
        cache='disk',
        device=0,
        project='runs/detect',
        name='v8s_DCNv2_train'
    )