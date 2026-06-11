from ultralytics import YOLO

if __name__ == '__main__':
    # 1. 直接加载官方原装结构与权重（不带 ECSA）
    model = YOLO('yolov8n.pt') 
    
    # 2. 此时底层的算分器已经是你激活的 PIoU 了
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
        optimizer='SGD',   # 保持最稳妥的 SGD 绝对公平
        project='runs/detect/Corn_Weed_Paper',  
        name='v8n_PIoU'    # 新的文件夹：纯 PIoU 测试
    )