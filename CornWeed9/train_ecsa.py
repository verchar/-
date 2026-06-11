from ultralytics import YOLO

if __name__ == '__main__':
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
        optimizer='SGD',   # <--- 【关键修复】强制指定经典优化器，干掉有Bug的MuSGD！
        project='runs/detect',
        name='v8s_ECSA_train'
    )