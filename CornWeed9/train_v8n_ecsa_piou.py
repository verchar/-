from ultralytics import YOLO

if __name__ == '__main__':
    # 1. 加载我们之前写好的、带有 ECSA 模块的 v8n 图纸
    model = YOLO('yolov8n_ecsa.yaml') 
    
    # 2. 核心操作：把官方 yolov8n.pt 的预训练知识“灌”进这个新骨架里
    model.load('yolov8n.pt')
    
    # 3. 开始训练
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
        optimizer='SGD',   # 保持 SGD 绝对公平
        project='runs/detect/Corn_Weed_Paper',  # 存在同一个目录下
        name='v8n_ECSA_PIoU'                    # 文件夹起名叫这个
    )