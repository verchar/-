from ultralytics import YOLO

if __name__ == '__main__':
    # 加载官方原装预训练权重，系统会自动匹配原装的 YOLOv8n 结构
    model = YOLO('yolov8n.pt') 
    
    # 开始训练
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
        optimizer='SGD',   
        project='runs/detect/Corn_Weed_Paper',  # <--- 【关键修改】精准指向你的专属基线文件夹
        name='v8n_baseline'                     # <--- 文件夹名称
    )