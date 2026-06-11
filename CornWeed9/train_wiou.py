from ultralytics import YOLO

if __name__ == '__main__':
    # 放弃 DCNv2，直接加载最稳定的原版网络结构
    # 但请放心，它的“大脑”已经被我们换成了 WIoU！
    model = YOLO('yolov8s.yaml') 
    
    model.train(
        data='CornWeed9.yaml',
        epochs=300,
        patience=50,
        batch=16,          
        imgsz=640,
        workers=8,         # 没有 DCN 捣乱，4个线程绝对稳
        amp=True,          # 混合精度全开，加速！
        cache='disk',
        device=0,
        project='runs/detect',
        name='v8s_WIoU_train'  # 我们的新实验文件夹
    )