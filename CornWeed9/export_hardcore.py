import os
from ultralytics import YOLO

def export_hardcore_model():
    print("🚀 正在启动【铁血锁定模式】导出 ONNX 模型...")
    
    # ⚠️ 已经为你填入了精准的绝对路径 (注意前面的 r 不能丢)
    pt_path = r'D:\Corn\cvnet_CornWeed9\CornWeed9\runs\detect\runs\detect\Corn_Weed_Paper\v8n_ECSA_WIoU\weights\best.pt'
    
    if not os.path.exists(pt_path):
        print(f"❌ 找不到权重文件，请检查路径是否正确: {pt_path}")
        return

    # 加载模型
    model = YOLO(pt_path)
    print(f"📦 正在处理并导出: {pt_path}")
    
    # 🚀 最严苛的导出参数设置
    export_path = model.export(
        format='onnx',       
        imgsz=640,           # 【锁定】死锁输入分辨率为 640x640
        half=False,          # 【锁定】绝对禁用 FP16 半精度！强制 FP32 计算
        dynamic=False,       # 【锁定】绝对禁用动态轴
        simplify=True,       # 【优化】极其重要：精简图结构
        opset=12             # 【锁定】降级并锁定算子集版本为 12
    )
    
    print("\n" + "="*50)
    print(f"✨ 铁血版 ONNX 导出成功！")
    print(f"📂 你的新文件已经生成在: {export_path}")
    print("👉 请立刻去这个文件夹，把新生成的 .onnx 文件复制到 Android Studio 的 assets 文件夹中！")
    print("="*50)

if __name__ == '__main__':
    export_hardcore_model()