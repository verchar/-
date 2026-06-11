import sys
import time
import psutil
import cv2
import numpy as np
import os
from PyQt5 import QtWidgets, QtGui, QtCore
from ultralytics import YOLO
# 核心技术：引入 Pillow 的中文字体支持
from PIL import Image, ImageDraw, ImageFont


# ====== 🚀 新增这一行：导入我们的 AI 专家面板 ======
from ai_advisor import AIAdvisorWidget
import matplotlib
matplotlib.use('Qt5Agg') # 声明使用 PyQt5 后端
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# 🚀 解决 Matplotlib 中文显示乱码的问题（必须加！）
plt.rcParams['font.sans-serif'] = ['SimHei'] # Windows 自带的黑体
plt.rcParams['axes.unicode_minus'] = False


# ==================== 核心黑科技：YOLO 专属多线程工作类 ====================
class DetectionThread(QtCore.QThread):
    # 定义传菜管道
    change_pixmap_signal = QtCore.pyqtSignal(np.ndarray)
    update_stats_signal = QtCore.pyqtSignal(dict)
    finished_signal = QtCore.pyqtSignal()
    fps_signal = QtCore.pyqtSignal(float)

    def __init__(self, model, source_type, source_path=None):
        super().__init__()
        self.model = model
        self.source_type = source_type
        self.source_path = source_path
        self._run_flag = True
        self.conf_threshold = 0.4  # 默认阈值

        # 🚀 终极防误诊魔法：记忆相册 (记录每个ID的历史身份)
        self.track_history = {}

        # 核心映射字典与 V1.4 固定高对比度调色板 (直接搬进后厨)
        self.class_names_cn = {
            0: '反枝苋', 1: '灰藜', 2: '龙葵', 3: '骆驼刺',
            4: '马齿苋', 5: '田旋花', 6: '狗尾草', 7: '合被苋', 8: '玉米幼苗'
        }
        self.colors = [
            (0, 0, 255), (255, 0, 255), (255, 0, 0), (0, 255, 255),
            (255, 255, 0), (0, 165, 255), (255, 0, 128), (128, 0, 255), (0, 255, 0)
        ]

        # ==================== V2.1.1 升级：字体瘦身引擎 ====================
        # 中文字体。请根据系统实际情况路径！
        font_paths = [
            "C:\\Windows\\Fonts\\simhei.ttf",  # 黑体 (最稳)
            "C:\\Windows\\Fonts\\arial.ttf",   # Arial Unicode (有些中文不支持)
            "simhei.ttf"  # 如果在当前目录有该字体
        ]
        
        self.chinese_font = None
        for path in font_paths:
            if os.path.exists(path):
                # 【强化升级】将字体大小从 60 降至 40，降低乱杂感，避免遮挡
                self.chinese_font = ImageFont.truetype(path, 40) 
                break
        
        # 如果没找到任何中文字体，退化为原生（无法显示中文）
        if not self.chinese_font:
            print("警告：未找到支持中文的 simhei.ttf 字体，标签仍将无法显示中文！")
            self.chinese_font = ImageFont.load_default()

    def run(self):
        # 线程的核心运行逻辑
        if self.source_type == 'image':
            frame = cv2.imread(self.source_path)
            if frame is not None:
                self.process_frame(frame)
        
        elif self.source_type in ['video', 'webcam']:
            # 打开视频流或摄像头 (0 表示默认摄像头)
            cap_source = 0 if self.source_type == 'webcam' else self.source_path
            cap = cv2.VideoCapture(cap_source)
            
            while cap.isOpened() and self._run_flag:
                ret, frame = cap.read()
                if not ret:
                    break  # 视频播放结束
                
                self.process_frame(frame)
                
                # 稍微休眠一下，同时维持大约 30 帧的播放速度
                self.msleep(30) 
            
            cap.release()
        
        # 任务结束，发送停止信号
        self.finished_signal.emit()

    def stop(self):
        # 外部按下停止按钮时，切断循环
        self._run_flag = False

    # ========================== V2.2 核心强化：雷达追踪与防误诊渲染引擎 ==========================
    def process_frame(self, frame):
        start_time = time.time() # 🚀 记录开始时间
        # 🚀 杀手锏 1：智能分流推理！(引入防崩溃容灾机制)
        if self.source_type in ['video', 'webcam']:
            try:
                # 优先尝试开启 ByteTrack 工业级追踪！persist=True 让它记住上一帧
                results = self.model.track(frame, conf=self.conf_threshold, persist=True, tracker="bytetrack.yaml", verbose=False)[0]
            except Exception as e:
                # ⚠️ 终极防弹衣：如果答辩现场 Windows 再次拦截底层库，绝不闪退！
                # 而是瞬间在后台打印警告，并无缝降级为普通检测引擎，保证画面继续播放！
                print(f"⚠️ 追踪引擎被系统拦截，已无缝切换至标准检测引擎: {e}")
                results = self.model(frame, conf=self.conf_threshold, verbose=False)[0]
        else:
            # 静态图片直接使用普通预测
            results = self.model(frame, conf=self.conf_threshold, verbose=False)[0]

        weed_counts = {}  # 统计字典
        img_h, img_w, _ = frame.shape
        
        box_thickness = 3 
        text_padding = 6 

        # OpenCV -> Pillow 
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        draw = ImageDraw.Draw(pil_image)

        # 解析模型吐出的结果
        if results.boxes is not None:
            for box in results.boxes:
                # 获取初始类别和置信度
                cls_id = int(box.cls[0].item())
                conf = box.conf[0].item()
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # 🚀 杀手锏 2：多数服从多数投票机制（防误诊）
                track_id = -1
                if box.id is not None:
                    # 如果这棵草被成功追踪到了，获取它的专属身份证号
                    track_id = int(box.id[0].item())
                
                if track_id != -1:
                    # 如果是个新朋友，给它建个档案
                    if track_id not in self.track_history:
                        self.track_history[track_id] = []
                    
                    # 把这一帧认出的类别存进它的个人档案里
                    self.track_history[track_id].append(cls_id)
                    
                    # 档案袋里最多只存最近的 15 帧的记忆
                    if len(self.track_history[track_id]) > 15:
                        self.track_history[track_id].pop(0)
                    
                    # 【核心魔法】：在这最近的 15 帧里，大家觉得它最可能是啥？
                    # 比如：[玉米, 玉米, 玉米, 田旋花, 玉米] -> 投票选出"玉米"
                    voted_cls_id = max(set(self.track_history[track_id]), key=self.track_history[track_id].count)
                    
                    # 强制纠正！用投票选出的真身覆盖掉模型这一帧可能眼花的错误
                    cls_id = voted_cls_id

                # 获取中文标签与专属颜色
                cn_name = self.class_names_cn.get(cls_id, f"未知_{cls_id}")
                class_color = self.colors[cls_id] 
                color_rgb = (class_color[2], class_color[1], class_color[0]) 

                # 画检测框
                draw.rectangle([x1, y1, x2, y2], outline=color_rgb, width=box_thickness)
                
                # 统计数量
                weed_counts[cn_name] = weed_counts.get(cn_name, 0) + 1

                # 标签文本准备 (如果有追踪ID，可以顺便打印出来看效果，这里为了画面整洁就不打了)
                label_text = f"{cn_name} {conf:.2f}"
                
                # 获取文本在 Pillow 中的实际占用像素大小
                text_bbox = draw.textbbox((0, 0), label_text, font=self.chinese_font)
                text_w = text_bbox[2] - text_bbox[0]
                text_h = text_bbox[3] - text_bbox[1]
                
                total_label_height = text_h + text_padding * 2
                final_bg_y_start = 0
                
                if y1 < total_label_height:
                    # 顶部边缘动态变位
                    final_bg_y_start = y1 + box_thickness
                else:
                    # 默认定位
                    final_bg_y_start = y1 - total_label_height

                # 绘制多色本地化中文标签背景与文字
                bg_rect_coords = (x1, final_bg_y_start, x1 + text_w + text_padding * 2, final_bg_y_start + total_label_height)
                draw.rectangle(bg_rect_coords, fill=color_rgb)
                draw.text((x1 + text_padding, final_bg_y_start + text_padding), label_text, font=self.chinese_font, fill=(255, 255, 255))

        # 把画好框和字的完整图，转回给界面显示
        processed_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        self.change_pixmap_signal.emit(processed_image)
        self.update_stats_signal.emit(weed_counts)
        # ====== 🚀 新增：计算并发送当前帧率 ======
        end_time = time.time()
        fps = 1.0 / ((end_time - start_time) + 1e-5) # 加上1e-5防止除以0
        self.fps_signal.emit(fps)


class DynamicPieChart(FigureCanvas):
    def __init__(self, parent=None, width=4, height=3, dpi=80):
        # 创建画布，背景设为透明色以融合 UI
        fig = Figure(figsize=(width, height), dpi=dpi)
        fig.patch.set_facecolor('#f4f4f4') # 浅灰背景，契合你的左侧面板
        self.axes = fig.add_subplot(111)
        super(DynamicPieChart, self).__init__(fig)
        self.setParent(parent)
        # ====== 🚀 新增这 2 行：美化初始状态 ======
        self.axes.axis('off')     # 把那两根丑陋的默认坐标轴隐藏掉
        self.update_chart({})     # 强行喂给它一个空数据，触发提示文字
        
    def update_chart(self, stats_dict):
        try:
            self.axes.clear() # 清空上一帧
            
            # 🚀 防弹衣 1：过滤掉数量为 0 的杂草，防止画饼图报错！
            valid_stats = {k: v for k, v in stats_dict.items() if v > 0}
            
            if not valid_stats:
                # 🚀 放大：空状态提示字从 12 放大到了 15
                self.axes.text(0.5, 0.5, '画面中暂无作物/杂草', 
                               horizontalalignment='center', verticalalignment='center',
                               fontsize=15, color='gray')
                self.axes.axis('off')
                self.draw() # 🚀 强力马达：强制立刻重绘
                return

            labels = list(valid_stats.keys())
            sizes = list(valid_stats.values())
            
            colors = []
            for label in labels:
                if '玉米' in label:
                    colors.append('#2ecc71') 
                else:
                    colors.append('#e74c3c') 

            explode = [0.05] * len(labels)
            
            # 🚀 放大：饼图标签和百分比数字从 10 放大到了 14
            self.axes.pie(sizes, explode=explode, labels=labels, colors=colors, 
                          autopct='%1.1f%%', shadow=True, startangle=140, 
                          textprops={'fontsize': 14, 'fontweight': 'bold'})
            
            self.axes.axis('equal') 
            self.draw() # 🚀 强力马达：强制立刻重绘
            
        except Exception as e:
            # 如果再报错，终端一定会打印出来！
            print(f"❌ 饼图绘制崩溃被抓住了: {e}")




# ==================== 主界面 UI 类（完美保留 V2.1 的所有并发和滑块功能） ====================
class WeedDetectionUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        # 1. 软件界面设定与图标
        self.setWindowTitle("农田杂草智能检测系统 V2.2 - [极速追踪与防误诊引擎]")
        self.resize(1200, 750) 

        # 2. 唤醒咱们的“最强大脑”！
        print("正在加载  改进模型...")
        if os.path.exists('v8n_ECSA.pt'):
            self.model = YOLO('v8n_ECSA.pt')
        else:
            print("警告：未找到 v8n_ECSA.pt，将加载基础预训练权重。")
            self.model = YOLO('yolov8n.pt')
        print("模型加载成功！")

        # 3. 初始化变量
        self.current_pixmap = None 
        self.thread = None 

        self.init_ui()

    def init_ui(self):
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QHBoxLayout(central_widget)

        left_layout = QtWidgets.QVBoxLayout()
        left_layout.setContentsMargins(10, 20, 10, 20)
        left_layout.setSpacing(15)
        
        btn_style = "color: white; border-radius: 8px; font-weight: bold; font-size: 16px;"
        
        self.btn_load_img = QtWidgets.QPushButton("🖼️ 静态图像检测")
        self.btn_load_img.setMinimumHeight(45)
        self.btn_load_img.setStyleSheet(f"background-color: #2980b9; {btn_style}")
        self.btn_load_img.clicked.connect(self.detect_image)

        self.btn_load_vid = QtWidgets.QPushButton("🎞️ 视频流检测")
        self.btn_load_vid.setMinimumHeight(45)
        self.btn_load_vid.setStyleSheet(f"background-color: #8e44ad; {btn_style}")
        self.btn_load_vid.clicked.connect(self.detect_video)

        self.btn_webcam = QtWidgets.QPushButton("🎥 实时摄像头接入")
        self.btn_webcam.setMinimumHeight(45)
        self.btn_webcam.setStyleSheet(f"background-color: #d35400; {btn_style}")
        self.btn_webcam.clicked.connect(self.detect_webcam)

        self.btn_stop = QtWidgets.QPushButton("⏹ 强制停止任务")
        self.btn_stop.setMinimumHeight(45)
        self.btn_stop.setStyleSheet(f"background-color: #c0392b; {btn_style}")
        self.btn_stop.clicked.connect(self.stop_thread)
        self.btn_stop.setEnabled(False) 

        slider_layout = QtWidgets.QVBoxLayout()
        self.label_conf = QtWidgets.QLabel("⚖️ 实时置信度阈值 (Conf): 0.40")
        self.label_conf.setStyleSheet("font-weight: bold; font-size: 14px; color: #2c3e50;")
        
        self.slider_conf = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_conf.setRange(10, 90) 
        self.slider_conf.setValue(40)     
        self.slider_conf.valueChanged.connect(self.change_conf_value) 
        
        slider_layout.addWidget(self.label_conf)
        slider_layout.addWidget(self.slider_conf)

        self.text_result = QtWidgets.QTextEdit()
        self.text_result.setReadOnly(True)
        self.text_result.setStyleSheet("background-color: #ecf0f1; border: 1px solid #bdc3c7; font-family: Microsoft YaHei; font-size: 15px; padding: 10px;")
        self.text_result.setPlaceholderText("检测结果统计将显示在这里...")
        # ====== 🚀 新增/修改下面这两行 ======
        # 1. 用 CSS 语法强制放大字体到 15px，并加粗
        self.text_result.setStyleSheet("font-size: 15px; font-weight: bold; font-family: 'Microsoft YaHei'; background-color: transparent; border: none;")
        
        # 2. 强制限制它的最大高度！(200 像素足够显示 6-7 行字了，剩下的空间全留给饼图)
        self.text_result.setMaximumHeight(200) 
        # ==================================
        
        left_layout.addWidget(self.text_result)

        left_layout.addWidget(self.btn_load_img)
        left_layout.addWidget(self.btn_load_vid)
        left_layout.addWidget(self.btn_webcam)
        left_layout.addWidget(self.btn_stop)
        left_layout.addLayout(slider_layout) 
        left_layout.addWidget(self.text_result)

        # ====== 动态数据可视化饼图 ======
        self.pie_chart = DynamicPieChart(self, width=3, height=3, dpi=80)
        
        # 🚀 把饼图的最小高度撑大！(从 250 改成 350 或者更大)
        self.pie_chart.setMinimumHeight(350) 
        
        left_layout.addWidget(self.pie_chart)
    

        # ====== 🚀 新增：边缘节点算力监控大屏 ======
        self.resource_label = QtWidgets.QLabel("🖥️ 边缘节点算力监控待命...\nFPS: 0.0 | CPU: 0.0% | RAM: 0.0%")
        # 黑底绿字，黑客监控风格！
        self.resource_label.setStyleSheet("background-color: #2c3e50; color: #2ecc71; padding: 10px; border-radius: 6px; font-weight: bold; font-family: Consolas, monospace; font-size: 14px;")
        left_layout.addWidget(self.resource_label)

        # 启动一个定时器，每 1 秒钟去偷看一次电脑的 CPU 和内存
        self.resource_timer = QtCore.QTimer()
        self.resource_timer.timeout.connect(self.update_hardware_stats)
        self.resource_timer.start(1000) # 1000毫秒 = 1秒
        self.current_fps = 0.0 # 存当前FPS的变量
        # ==========================================
        
        self.label_image = QtWidgets.QLabel("请在左侧选择数据输入源启动系统")
        self.label_image.setAlignment(QtCore.Qt.AlignCenter)
        self.label_image.setStyleSheet("background-color: #2c3e50; color: #ecf0f1; font-size: 20px; border-radius: 10px; font-weight: bold;")
        self.label_image.setMinimumSize(750, 550) 

        # 这是你原来的两行代码
        main_layout.addLayout(left_layout, 1)
        main_layout.addWidget(self.label_image, 4)

        # ====== 🚀 新增以下 4 行：把 AI 面板镶嵌到最右侧 ======
        self.ai_panel = AIAdvisorWidget()
        main_layout.addWidget(self.ai_panel, 2) # 2 是宽度比例，让它和画面保持和谐
        
        # 绑定 AI 面板上的“一键诊断”按钮
        self.ai_panel.auto_diagnose_btn.clicked.connect(self.trigger_ai_diagnosis)
        # ======================================================

     # ====== 🚀 新增这个方法：自动抓取数据给 AI ======
    def trigger_ai_diagnosis(self):
        # 1. 瞬间抓取左侧 self.text_result 里的实时文本！
        current_stats = self.text_result.toPlainText()
        
        # 2. 加个小保险：如果还没开始检测，给个提示
        if "总株数" not in current_stats:
            current_stats = "目前农田画面暂无识别数据，请先启动图片或视频检测。"
            
        # 3. 把抓到的数据喂给右边的 AI 面板，开始干活！
        self.ai_panel.start_diagnosis(current_stats)
    # ==============================================   

    def change_conf_value(self, value):
        conf_val = value / 100.0
        self.label_conf.setText(f"⚖️ 实时置信度阈值 (Conf): {conf_val:.2f}")
        if self.thread is not None and self.thread.isRunning():
            self.thread.conf_threshold = conf_val

    def detect_image(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择农田图片", "", "Images (*.png *.jpg *.jpeg)")
        if file_path:
            self.start_thread('image', file_path)

    def detect_video(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择无人机巡检视频", "", "Videos (*.mp4 *.avi *.mkv)")
        if file_path:
            self.start_thread('video', file_path)

    def detect_webcam(self):
        reply = QtWidgets.QMessageBox.question(self, '接入摄像头', '即将唤醒本地摄像头，请确保设备已连接。是否继续？',
                                     QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            self.start_thread('webcam')

    def start_thread(self, source_type, path=None):
        self.stop_thread() 
        
        self.text_result.setText("引擎已启动，正在初始化计算管线...")
        self.btn_stop.setEnabled(True) 

        self.thread = DetectionThread(self.model, source_type, path)
        self.thread.conf_threshold = self.slider_conf.value() / 100.0
        
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.update_stats_signal.connect(self.update_stats)
        self.thread.finished_signal.connect(self.thread_finished)
        self.thread.fps_signal.connect(self.update_fps_value)
        self.thread.start()

    def stop_thread(self):
        if self.thread is not None and self.thread.isRunning():
            self.thread.stop()
            self.thread.wait() 

    def thread_finished(self):
        self.btn_stop.setEnabled(False)
        current_text = self.text_result.toPlainText()
        self.text_result.setText(current_text + "\n\n[系统提示] 检测管线已安全关闭。")

    @QtCore.pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        try:
            # 强制捕获所有可能导致画面卡死的隐形错误
            rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_img.shape
            bytes_per_line = ch * w
            qt_img = QtGui.QImage(rgb_img.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
            self.current_pixmap = QtGui.QPixmap.fromImage(qt_img)
            self.update_image_view()
        except Exception as e:
            print(f"❌ 隐形 Bug 被抓住了 (图像转换环节): {e}")

    @QtCore.pyqtSlot(dict)
    def update_stats(self, weed_counts):
        report = "【实时监控统计】 V2.2 雷达追踪版\n\n"
        if not weed_counts:
            report += "视野内暂无目标或玉米。\n"
        else:
            total_count = sum(weed_counts.values())
            report += f"总株数: {total_count} 株\n-----------------\n"
            for name, count in weed_counts.items():
                report += f"🌱 {name}: {count} 株\n"
                
        self.text_result.setText(report)

        # 🚀 组合拳 3：变量名必须和上面接收的参数名一模一样！
        if hasattr(self, 'pie_chart'):
            self.pie_chart.update_chart(weed_counts)  # <--- 就是这里改了！

    @QtCore.pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        # 🚀 探头 1：看看信号到底传过来没有！
        print(f"🚨 [探头1] 成功接收到后台画面！尺寸: {cv_img.shape}") 
        
        try:
            rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_img.shape
            bytes_per_line = ch * w
            
            # ⚠️ 终极修复：必须加上 .copy()！
            # 防止 PyQt5 底层偷偷回收 numpy 数组的内存，导致图片变空气！
            qt_img = QtGui.QImage(rgb_img.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888).copy()
            
            self.current_pixmap = QtGui.QPixmap.fromImage(qt_img)
            self.update_image_view()
            
        except Exception as e:
            print(f"❌ 隐形 Bug 被抓住了 (图像转换环节): {e}")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'label_image'):
            self.update_image_view()
    

    # ... 上面是你 WeedDetectionUI 类里的其他方法 ...

    def update_image_view(self):
        try:
            if hasattr(self, 'current_pixmap') and self.current_pixmap is not None:
                width = max(self.label_image.width(), 1)
                height = max(self.label_image.height(), 1)
                
                scaled_pixmap = self.current_pixmap.scaled(
                    width, 
                    height, 
                    QtCore.Qt.KeepAspectRatio, 
                    QtCore.Qt.SmoothTransformation
                )
                self.label_image.setPixmap(scaled_pixmap)
                
                # 🚀 探头 2：看看画面到底贴上去了没！
                print("✅ [探头2] 画面贴图成功！") 
                
        except Exception as e:
            print(f"❌ 隐形 Bug 被抓住了 (贴图渲染环节): {e}")

    # ========================================================
    # 🚀 把这 2 个方法粘在这里！注意 def 前面是 4 个空格的缩进！
    @QtCore.pyqtSlot(float)
    def update_fps_value(self, fps):
        self.current_fps = fps

    def update_hardware_stats(self):
        # 读取 CPU 和 内存占用率
        cpu_usage = psutil.cpu_percent(interval=None)
        ram_usage = psutil.virtual_memory().percent
        
        # 更新到界面上
        status_text = f"🖥️ 边缘计算资源实时监控\n⚡ FPS: {self.current_fps:.1f} | 🧠 CPU: {cpu_usage}% | 💾 内存: {ram_usage}%"
        self.resource_label.setText(status_text)
        
        # 如果 FPS 掉到 10 以下，或者 CPU 飙升到 90% 以上，文字变红报警
        if (self.current_fps > 0 and self.current_fps < 10) or cpu_usage > 90:
            self.resource_label.setStyleSheet("background-color: #2c3e50; color: #e74c3c; padding: 10px; border-radius: 6px; font-weight: bold; font-family: Consolas, monospace; font-size: 14px;")
        else:
            self.resource_label.setStyleSheet("background-color: #2c3e50; color: #2ecc71; padding: 10px; border-radius: 6px; font-weight: bold; font-family: Consolas, monospace; font-size: 14px;")
    # ========================================================

    # 这是你原来的代码，保持不动
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'label_image'):
            self.update_image_view()

if __name__ == '__main__':
    if os.name == 'nt':
        if 'PATH' in os.environ:
            os.environ['PATH'] = 'C:\\Windows\\Fonts;' + os.environ['PATH']
            
    app = QtWidgets.QApplication(sys.argv)
    window = WeedDetectionUI()
    window.show()
    sys.exit(app.exec_())