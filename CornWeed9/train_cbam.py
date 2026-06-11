from ultralytics import YOLO

if __name__ == '__main__':
    # 依然是加载咱们带有 CBAM 的图纸，并灌入官方基础权重重新开始
    model = YOLO('yolov8_cbam.yaml').load('yolov8s.pt')

    # 开始终极拉练
    results = model.train(
        data='CornWeed9.yaml', 
        epochs=300,          # 【核心修改】拉满 300 轮！
        patience=50,         # 【核心修改】加入早停机制：如果连续 50 轮没进步，就提前交卷
        imgsz=640, 
        batch=16, 
        workers=0, 
        device=0,
        project='runs/detect/Corn_Weed_Paper', 
        name='v8s_CBAM_300epochs'  # 【核心修改】换个新文件夹名字，千万别把之前 100 轮的覆盖了！
    )