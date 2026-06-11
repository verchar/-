from ultralytics import YOLO

if __name__ == '__main__':
    # 直接加载官方原装结构与权重（此时没有 ECSA！）
    model = YOLO('yolov8n.pt') 
    
    # 此时底层的算分器已经是你激活的 WIoU 了
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
        project='runs/detect/Corn_Weed_Paper',  
        name='v8n_WIoU'    # 新的文件夹：纯 WIoU 测试
    )