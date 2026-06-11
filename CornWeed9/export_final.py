from ultralytics import YOLO

if __name__ == '__main__':
    print("🚀 正在使用【移动端最强兼容模式】导出您的心血模型 v8n_ECSA_WIoU ...")
    
    # 填入你终极模型的路径
    model = YOLO('runs/detect/runs/detect/Corn_Weed_Paper/v8n_ECSA_WIoU/weights/best.pt')
    
    # 🌟 这里的参数是专门对抗“满屏渔网图”的终极解法
    model.export(
        format='onnx', 
        opset=11,          # 【关键】降级到所有手机都完美支持的 11 算子集
        simplify=True,     # 【关键】必须开启精简！把那些会导致手机崩溃的冗余算子全部折叠掉
        imgsz=640,         # 严格锁定 640x640 尺寸
        half=False,        # 坚决不使用 fp16 半精度，防止天玑处理器数值溢出变成 NaN
        dynamic=False      # 关闭动态尺寸，让计算图极其稳定
    )
    
    print("✨ 导出完成！")