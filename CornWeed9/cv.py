import matplotlib.pyplot as plt
import numpy as np

# 示例数据
recall = np.linspace(0, 1, 100)
precision = np.exp(-5 * recall) + 0.1 * np.random.rand(100)  # 模拟精度-召回率曲线

# 绘制精度-召回率曲线
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label="Precision-Recall Curve", color="blue", linewidth=2)

# 添加标题和标签
plt.title("Precision-Recall Curve", fontsize=16)
plt.xlabel("Recall", fontsize=14)
plt.ylabel("Precision", fontsize=14)

# 设置网格
plt.grid(True, linestyle="--", alpha=0.6)

# 添加图例
plt.legend(loc="upper right", fontsize=12)

# 保存图像
plt.savefig("precision_recall_curve.jpg", dpi=300, bbox_inches="tight")
plt.show()