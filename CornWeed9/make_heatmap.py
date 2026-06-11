from ultralytics import YOLO
import os

# 1. 加载我们带有 CBAM 注意力机制的最强大脑
model = YOLO('yolov8_cbam_best.pt')

# 2. 从你的测试集里挑一张最典型、最清晰的杂草图片（注意修改为你真实的图片路径！）
# 比如：img_path = 'images/test/0008.jpg'
img_path = 'images/test/0008.jpg'

if not os.path.exists(img_path):
    print(f"❌ 找不到图片：{img_path}，请检查路径！")
else:
    print(f"🚀 正在提取深度神经网络底层特征热力图...")
    
    # 3. 核心黑科技：开启 visualize=True 隐藏参数
    # 它会强行剖开 YOLO 的每一层网络，把你那张图片在各个层级的“热力特征”全部保存下来！
    results = model(img_path, visualize=True)
    
    print("🎉 特征图提取成功！快去左侧的 runs/detect/ 文件夹下寻找最新生成的 predict 文件夹！")