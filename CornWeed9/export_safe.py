from ultralytics import YOLO

if __name__ == '__main__':
    print("🚀 正在使用安全模式转换基线模型 (作为对照组)...")
    # 绝对精准的套娃相对路径
    model_base = YOLO('runs/detect/runs/detect/Corn_Weed_Paper/v8n_baseline/weights/best.pt')
    model_base.export(
        format='onnx', 
        opset=13, 
        simplify=False, 
        half=False
    )
    
    print("🚀 正在使用安全模式转换你的终极改进模型...")
    # 绝对精准的套娃相对路径
    model_ours = YOLO('runs/detect/runs/detect/Corn_Weed_Paper/v8n_ECSA_WIoU/weights/best.pt')
    model_ours.export(
        format='onnx', 
        opset=13, 
        simplify=False, 
        half=False
    )
    
    print("✨ 全部导出完成！快去 weights 文件夹里拿新鲜出炉的 .onnx 吧！")