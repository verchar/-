from ultralytics import YOLO

# 加载你的魔改模型
model = YOLO('v8n_ECSA_WIoU.pt')

print("🚀 正在启动 TFLite 转换...")

# 核心修改：关闭int8量化，避免下载校准文件，补充移动端适配参数
model.export(
    format='tflite',
    imgsz=640,
    half=True,        # 保留fp16量化，适配移动端推理
    int8=False,       # 关键：关闭int8量化，彻底跳过校准文件下载
    simplify=True,    # 简化中间模型，减少转换报错
    batch=1,          # 固定batch为1，适配移动端固定输入
    dynamic=False,    # 关闭动态shape，提升端侧兼容性
    nms=False         # 可选：关闭内置NMS，如需端侧NMS再开启
)

print("🎉 转换完成！")
