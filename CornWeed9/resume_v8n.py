from ultralytics import YOLO

if __name__ == '__main__':
    # 核心修改点 1：路径指向你中断那个文件夹里的 last.pt
    # 路径根据你刚才跑的文件夹改，比如：
    model = YOLO('runs/detect/Corn_Weed_Paper/v8n_ECSA_PIoU/weights/last.pt')
    
    # 核心修改点 2：直接调用 train 并设置 resume=True
    # 注意：这时候不需要再传 data, epochs 等参数，它会自动从 last.pt 里读取之前的设置
    model.train(resume=True)