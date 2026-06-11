from ultralytics import YOLO

# 1. 直接加载你刚刚导出的那个 43MB 的 ONNX 文件！
# （请替换为真实的 best.onnx 路径）
onnx_model_path = r"D:\Corn\cvnet_CornWeed9\CornWeed9\runs\detect\Corn_Weed_Paper\v8s_baseline3\weights\best.onnx"
model = YOLO(onnx_model_path)

# 2. 随便找一张你电脑上的玉米地测试照片
# （请替换为真实的图片路径）
test_image_path = r"D:\Corn\cvnet_CornWeed9\CornWeed9\images\test\0008.jpg" 

print("🚀 正在使用 ONNX 模型直接推理...")
# 3. 运行推理并显示结果
results = model(test_image_path, show=True)