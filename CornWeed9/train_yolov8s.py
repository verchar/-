from ultralytics import YOLO

if __name__ == '__main__':
    print("⏳ 正在请求核武库，准备下载/加载 YOLOv8s 预训练权重...")
    model = YOLO('yolov8s.pt') 
    
    print("🔥 弹药装填完毕，开启防过拟合机制，正式点火！")
    results = model.train(
        data='CornWeed9.yaml',  # 【战术变更】因为脚本和yaml在同一个文件夹，直接写名字！
        epochs=300,          
        patience=50,         
        imgsz=640,           
        batch=8,            
        device='0',          
        workers=4,           
        project='Corn_Weed_Paper',  
        name='v8s_baseline'         
    )
    
    print("✅ 报告总指挥：极限基线训练已完成！")