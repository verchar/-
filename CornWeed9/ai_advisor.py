# ai_advisor.py
import os
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, 
                             QLineEdit, QPushButton, QLabel)
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QFont, QTextCursor
from openai import OpenAI

class OllamaWorkerThread(QThread):
    chunk_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, chat_history):
        super().__init__()
        self.chat_history = chat_history

    def run(self):
        try:
            # ====== 🚀 核心修复：强行给系统塞一个假的 API_KEY，堵住 openai 库的嘴 ======
            os.environ["OPENAI_API_KEY"] = "ollama_local"
            
            client = OpenAI(
                base_url="http://localhost:11434/v1", 
                api_key="ollama_local", 
            )
            # =====================================================================
            
            completion = client.chat.completions.create(
                model="qwen3-vl:4b",  # ⚠️ 再次确认这里是你 Ollama 里真实下载的模型名哦！
                messages=self.chat_history,
                stream=True
            )
            for chunk in completion:
                if chunk.choices[0].delta.content is not None:
                    self.chunk_signal.emit(chunk.choices[0].delta.content)
            self.finished_signal.emit()
            
        except Exception as e:
            self.error_signal.emit(f"本地 Ollama 连接失败，请检查服务是否开启: {str(e)}")


class AIAdvisorWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(350) 
        self.setMaximumWidth(420)
        
        self.system_prompt = (
            "你是一个拥有丰富经验的农业植保专家，说话专业、严谨、精炼，能够针对玉米农田中的各种杂草问题提供深入的危害分析和实用的解决方案"
            "当用户向你提供玉米农田杂草情况时，你需要给出具体的危害分析，并提供安全环保的除草建议或打药配方，且不以表格的形式输出内容。"
        )
        self.chat_history = [{"role": "system", "content": self.system_prompt}]
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5) 
        
        self.status_label = QLabel("🟢 本地 AI 专家待命")
        self.status_label.setStyleSheet("color: #27ae60; font-weight: bold; font-size: 14px;")
        layout.addWidget(self.status_label)

        self.auto_diagnose_btn = QPushButton("🤖 获取当前画面数据并诊断")
        self.auto_diagnose_btn.setStyleSheet("padding: 10px; background-color: #f39c12; color: white; font-weight: bold; border-radius: 4px;")
        layout.addWidget(self.auto_diagnose_btn)

        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setFont(QFont("Microsoft YaHei", 10))
        self.text_edit.setStyleSheet("background-color: #f8f9fa; padding: 8px; border-radius: 5px;")
        layout.addWidget(self.text_edit)

        input_layout = QHBoxLayout()
        self.input_box = QLineEdit()
        self.input_box.setPlaceholderText("继续追问专家...")
        self.input_box.returnPressed.connect(self.send_user_message)
        input_layout.addWidget(self.input_box)

        self.send_btn = QPushButton("发送")
        self.send_btn.setStyleSheet("padding: 6px 12px; background-color: #2980b9; color: white; border-radius: 4px;")
        self.send_btn.clicked.connect(self.send_user_message)
        input_layout.addWidget(self.send_btn)

        layout.addLayout(input_layout)

    def start_diagnosis(self, detection_summary):
        self.auto_diagnose_btn.setEnabled(False) 
        self.status_label.setText("🔄 专家正在深度思考中...")
        
        prompt = f"这是当前画面最新的农田检测统计结果：\n{detection_summary}\n\n请帮我分析这些杂草的具体危害，并给出除草建议。"
        self.text_edit.append(f"<br><b style='color:#2980b9;'>🧑‍🌾 抓取当前数据：</b><br>{detection_summary}")
        self.text_edit.append("<br><b style='color:#27ae60;'>🤖 专家诊断：</b><br>")
        
        self.chat_history.append({"role": "user", "content": prompt})
        self.call_ollama()

    def send_user_message(self):
        user_text = self.input_box.text().strip()
        if not user_text: return
        self.input_box.clear()
        self.input_box.setEnabled(False)
        self.send_btn.setEnabled(False)

        self.text_edit.append(f"<br><b style='color:#2980b9;'>🧑‍🌾 我：</b><br>{user_text}")
        self.text_edit.append("<br><b style='color:#27ae60;'>🤖 专家：</b><br>")
        self.chat_history.append({"role": "user", "content": user_text})
        self.call_ollama()

    def call_ollama(self):
        self.current_reply = ""
        self.worker = OllamaWorkerThread(self.chat_history)
        self.worker.chunk_signal.connect(self.update_content)
        self.worker.error_signal.connect(self.show_error)
        self.worker.finished_signal.connect(self.analysis_finished)
        self.worker.start()

    def update_content(self, text):
        self.current_reply += text
        cursor = self.text_edit.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.text_edit.setTextCursor(cursor)
        self.text_edit.insertPlainText(text)
        self.text_edit.ensureCursorVisible()

    def show_error(self, err_msg):
        self.text_edit.append(f"<br><b style='color:red;'>{err_msg}</b>")
        self.finish_ui_restore()

    def analysis_finished(self):
        self.chat_history.append({"role": "assistant", "content": self.current_reply})
        self.finish_ui_restore()
        
    def finish_ui_restore(self):
        self.status_label.setText("🟢 本地 AI 专家待命")
        self.input_box.setEnabled(True)
        self.send_btn.setEnabled(True)
        self.auto_diagnose_btn.setEnabled(True)
        self.input_box.setFocus()