from ultralytics import YOLO

if __name__ == '__main__':
    # 1. 加载包含 ECSA 注意力机制的 v8n 图纸
    model = YOLO('yolov8n_ecsa.yaml') 
    
    # 2. 灌入官方预训练权重
    model.load('yolov8n.pt')
    
    # 3. 开始训练 (此时底层用的是官方 CIoU)
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
        optimizer='SGD',   # 依然保持 SGD 绝对公平
        project='runs/detect/Corn_Weed_Paper',  
        name='v8n_ECSA'    # 新的文件夹：纯 ECSA 测试
    )