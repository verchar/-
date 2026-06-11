from ultralytics import YOLO

if __name__ == '__main__':
    # 1. 依然使用我们建好的 v8n + ECSA 架构图纸
    model = YOLO('yolov8n_ecsa.yaml') 
    
    # 2. 灌入官方预训练权重
    model.load('yolov8n.pt')
    
    # 3. 开始终极训练
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
        optimizer='SGD',   # 依然是绝对公平的 SGD
        project='runs/detect/Corn_Weed_Paper',  
        name='v8n_ECSA_WIoU'  # 新的战绩保存文件夹
    )