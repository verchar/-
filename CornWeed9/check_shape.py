import onnx

def check_model_shape(model_path):
    try:
        model = onnx.load(model_path)
        input_shape = [dim.dim_value for dim in model.graph.input[0].type.tensor_type.shape.dim]
        output_shape = [dim.dim_value for dim in model.graph.output[0].type.tensor_type.shape.dim]
        print(f"📦 模型: {model_path}")
        print(f"   📥 输入图像要求: {input_shape}")
        print(f"   📤 输出矩阵格式: {output_shape}\n")
        
    except Exception as e:
        print(f"读取 {model_path} 失败: {e}")

print("=== 🔍 模型格式对比探测器 ===\n")

# ⚠️ 请换成你电脑上真实的路径！
check_model_shape(r"D:\Corn\cvnet_CornWeed9\CornWeed9\yolov8_cbam_best.onnx")   # 那个运行完美的模型
check_model_shape(r"D:\Corn\cvnet_CornWeed9\CornWeed9\runs\detect\Corn_Weed_Paper\v8s_baseline3\weights\best.onnx") # 那个画渔网图的模型