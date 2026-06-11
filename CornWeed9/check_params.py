from ultralytics import YOLO

# 加载您修改了 ECSA 的那个 yaml 配置文件
# 注意：这里一定要填您自己改过的那个 yaml 文件的绝对或相对路径
model_ecsa = YOLO('yolov8n_ecsa.yaml') 

print("=== 改进模型 YOLOv8n+ECSA 信息 ===")
model_ecsa.info()