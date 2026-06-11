import os
from ultralytics import YOLO

def export_v8s_baseline():
    print("🚀 正在启动【高稳定模式】导出 YOLOv8s Baseline ONNX 模型...")
    
    # ==========================================
    # ⚠️ 请填入你的 v8s_baseline 训练出来的 best.pt 绝对路径
    # 注意前面的 r 不要删掉，防止 Windows 路径转义报错
    # ==========================================
    pt_path = r'D:\Corn\cvnet_CornWeed9\CornWeed9\runs\detect\Corn_Weed_Paper\v8s_baseline3\weights\best.pt' # <--- 修改这里！
    
    if not os.path.exists(pt_path):
        print(f"❌ 找不到权重文件，请检查路径是否正确: {pt_path}")
        return

    # 加载基线模型
    model = YOLO(pt_path)
    print(f"📦 正在处理并导出: {pt_path}")
    
    # 🚀 最严苛、最稳健的移动端导出参数设置
    export_path = model.export(
        format='onnx',       
        imgsz=640,           # 死锁输入分辨率
        half=False,          # 绝对禁用 FP16 半精度！强制 FP32 计算
        dynamic=False,       # 绝对禁用动态轴
        simplify=True,       # 极致精简图结构
        opset=12             # 锁定算子集版本为 Android 最兼容的 12
    )
    
    print("\n" + "="*50)
    print(f"✨ YOLOv8s Baseline ONNX 导出成功！")
    print(f"📂 你的新文件已经生成在: {export_path}")
    print("👉 请立刻去这个文件夹，把新生成的 .onnx 文件复制到 Android Studio 中！")
    print("="*50)

if __name__ == '__main__':
    export_v8s_baseline()