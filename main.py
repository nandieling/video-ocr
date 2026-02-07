import sys
import os
import cv2
import time
import datetime
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QPushButton, QLabel, QSlider, 
                               QFileDialog, QLineEdit, QProgressBar, QMessageBox, QGroupBox)
from PySide6.QtCore import Qt, QTimer, QThread, Signal, QRect, QPoint, QSize
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QBrush, QAction

# 尝试导入 RapidOCR
try:
    from rapidocr_onnxruntime import RapidOCR
except ImportError:
    print("错误: 未安装 rapidocr_onnxruntime。请运行: pip install rapidocr_onnxruntime")
    sys.exit(1)

# ===========================
# 工具函数
# ===========================
def ms_to_srt_time(ms):
    """将毫秒转换为 SRT 时间格式 00:00:00,000"""
    seconds = ms / 1000
    td = datetime.timedelta(seconds=seconds)
    fin = str(td)
    # 处理 timedelta 可能不包含毫秒的情况
    if '.' in fin:
        main, micro = fin.split('.')
        return f"{main},{micro[:3]}"
    else:
        return f"{fin},000"

# ===========================
# 后台工作线程：OCR 处理
# ===========================
class OCRWorker(QThread):
    progress_signal = Signal(int, str)  # 进度(%), 剩余时间信息
    finished_signal = Signal(str)       # 完成信号，返回文件路径
    error_signal = Signal(str)          # 错误信号

    def __init__(self, video_path, output_path, roi, threshold=0.6):
        super().__init__()
        self.video_path = video_path
        self.output_path = output_path
        self.roi = roi  # (x, y, w, h) 针对原视频分辨率的坐标
        self.is_running = True
        self.threshold = threshold # 置信度阈值

    def run(self):
        try:
            # 初始化 RapidOCR
            ocr_engine = RapidOCR()
            
            cap = cv2.VideoCapture(self.video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0: fps = 25.0
            
            # 采样间隔：为了速度，不一定每帧都识别。
            # 假设字幕至少持续 0.5 秒，FPS=24，则每 5-8 帧检查一次是安全的。
            skip_frames = 5 
            
            subtitles = []
            # 当前正在记录的字幕片段
            current_sub = {"text": "", "start": 0, "end": 0}
            
            start_time = time.time()
            
            # 解构 ROI
            rx, ry, rw, rh = self.roi
            
            frame_idx = 0
            processed_count = 0
            
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 仅在采样帧进行处理
                if frame_idx % skip_frames == 0:
                    # 裁剪图像 (注意边界检查)
                    h_vid, w_vid = frame.shape[:2]
                    x1, y1 = max(0, int(rx)), max(0, int(ry))
                    x2, y2 = min(w_vid, int(rx + rw)), min(h_vid, int(ry + rh))
                    
                    if x2 > x1 and y2 > y1:
                        roi_img = frame[y1:y2, x1:x2]
                        
                        # 调用 RapidOCR
                        # result 格式: [[[[x1,y1],...], "text", score], ...]
                        result, _ = ocr_engine(roi_img)
                        
                        detected_text = ""
                        if result:
                            # 拼接所有检测到的文本行
                            texts = []
                            for line in result:
                                # RapidOCR score may arrive as str; coerce safely
                                try:
                                    score = float(line[2])
                                except (TypeError, ValueError, IndexError):
                                    score = 0.0
                                if score > self.threshold:
                                    texts.append(line[1])
                            detected_text = " ".join(texts).strip()
                        
                        current_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                        
                        # --- 字幕逻辑核心 ---
                        # 简单去重：如果文字和上一段一样，延长结束时间
                        if detected_text:
                            if current_sub["text"] == detected_text:
                                current_sub["end"] = current_ms
                            else:
                                # 文字变了，保存上一句（如果存在）
                                if current_sub["text"]:
                                    subtitles.append(current_sub)
                                # 开启新的一句
                                current_sub = {"text": detected_text, "start": current_ms, "end": current_ms}
                        else:
                            # 当前没文字，如果之前有，说明字幕消失了，结束上一句
                            if current_sub["text"]:
                                subtitles.append(current_sub)
                                current_sub = {"text": "", "start": 0, "end": 0}
                    
                    # --- 进度计算 ---
                    processed_count += 1
                    elapsed = time.time() - start_time
                    
                    # 估算
                    total_steps = total_frames // skip_frames
                    if processed_count > 0:
                        avg_time = elapsed / processed_count
                        remain_steps = total_steps - processed_count
                        remain_sec = remain_steps * avg_time
                        
                        # 格式化剩余时间
                        m, s = divmod(int(remain_sec), 60)
                        h, m = divmod(m, 60)
                        time_str = f"{h:02d}:{m:02d}:{s:02d}"
                        
                        percent = int((frame_idx / total_frames) * 100)
                        self.progress_signal.emit(percent, f"剩余时间: {time_str}")

                frame_idx += 1
            
            # 循环结束后，保存最后一句
            if current_sub["text"]:
                subtitles.append(current_sub)
            
            cap.release()
            
            # 如果是点击取消退出的
            if not self.is_running:
                return

            # 生成文件
            self.write_srt(subtitles)
            self.finished_signal.emit(self.output_path)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error_signal.emit(str(e))

    def write_srt(self, subs):
        with open(self.output_path, 'w', encoding='utf-8') as f:
            idx = 1
            for s in subs:
                # 过滤极短的闪烁误识别 (例如小于 0.1秒)
                if s["end"] - s["start"] < 100:
                    continue
                
                start_str = ms_to_srt_time(s["start"])
                end_str = ms_to_srt_time(s["end"])
                f.write(f"{idx}\n")
                f.write(f"{start_str} --> {end_str}\n")
                f.write(f"{s['text']}\n\n")
                idx += 1

    def stop(self):
        self.is_running = False


# ===========================
# 自定义视频显示控件 (支持画框)
# ===========================
class VideoLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        # 初始 ROI 框 (显示坐标系)
        self.roi_rect = QRect(100, 200, 400, 80) 
        
        self.drag_mode = None # None, 'move', 'resize_tl', 'resize_tr', ...
        self.last_pos = QPoint()
        self.handle_size = 10
        
        # 视频缩放比例 (实际分辨率 / 显示分辨率)
        self.scale_w = 1.0
        self.scale_h = 1.0

    def get_real_roi(self):
        """获取映射回视频真实分辨率的 ROI"""
        x = self.roi_rect.x() * self.scale_w
        y = self.roi_rect.y() * self.scale_h
        w = self.roi_rect.width() * self.scale_w
        h = self.roi_rect.height() * self.scale_h
        return (int(x), int(y), int(w), int(h))

    def paintEvent(self, event):
        super().paintEvent(event) # 绘制视频帧
        
        if self.pixmap():
            painter = QPainter(self)
            
            # 绘制半透明填充
            painter.setBrush(QBrush(QColor(0, 255, 0, 40))) # 绿色，透明度
            painter.setPen(QPen(Qt.green, 2, Qt.SolidLine))
            painter.drawRect(self.roi_rect)
            
            # 绘制四个角的控制点
            painter.setBrush(Qt.white)
            painter.setPen(Qt.black)
            
            r = self.roi_rect
            handles = [r.topLeft(), r.topRight(), r.bottomLeft(), r.bottomRight()]
            
            for p in handles:
                painter.drawRect(p.x() - 5, p.y() - 5, 10, 10)
            
            # 绘制提示文字
            painter.setPen(Qt.yellow)
            painter.drawText(r.topLeft() + QPoint(5, -5), "字幕区域 (拖动调整)")

    def mousePressEvent(self, event):
        pos = event.position().toPoint()
        r = self.roi_rect
        
        # 简单的碰撞检测
        tl = QRect(r.left()-5, r.top()-5, 10, 10)
        br = QRect(r.right()-5, r.bottom()-5, 10, 10)
        
        if tl.contains(pos):
            self.drag_mode = 'resize_tl'
        elif br.contains(pos):
            self.drag_mode = 'resize_br'
        elif r.contains(pos):
            self.drag_mode = 'move'
        else:
            self.drag_mode = None
            
        self.last_pos = pos

    def mouseMoveEvent(self, event):
        pos = event.position().toPoint()
        
        # 改变鼠标光标
        r = self.roi_rect
        tl = QRect(r.left()-5, r.top()-5, 10, 10)
        br = QRect(r.right()-5, r.bottom()-5, 10, 10)
        
        if tl.contains(pos) or br.contains(pos):
            self.setCursor(Qt.SizeFDiagCursor)
        elif r.contains(pos):
            self.setCursor(Qt.SizeAllCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

        if not self.drag_mode:
            return

        dx = pos.x() - self.last_pos.x()
        dy = pos.y() - self.last_pos.y()
        
        if self.drag_mode == 'move':
            self.roi_rect.translate(dx, dy)
        elif self.drag_mode == 'resize_br':
            # 限制最小尺寸
            new_w = max(20, self.roi_rect.width() + dx)
            new_h = max(20, self.roi_rect.height() + dy)
            self.roi_rect.setWidth(new_w)
            self.roi_rect.setHeight(new_h)
        elif self.drag_mode == 'resize_tl':
            new_x = self.roi_rect.x() + dx
            new_y = self.roi_rect.y() + dy
            new_w = max(20, self.roi_rect.width() - dx)
            new_h = max(20, self.roi_rect.height() - dy)
            # 只有在宽高合法时才更新
            if new_w > 20 and new_h > 20:
                self.roi_rect.setRect(new_x, new_y, new_w, new_h)
        
        # 边界限制 (简单处理，不让框完全跑出去)
        self.roi_rect = self.roi_rect.intersected(self.rect())
        
        self.last_pos = pos
        self.update()

    def mouseReleaseEvent(self, event):
        self.drag_mode = None


# ===========================
# 主界面
# ===========================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RapidOCR 视频硬字幕提取器")
        self.resize(1000, 750)
        
        self.video_path = ""
        self.cap = None
        self.total_frames = 0
        self.fps = 0
        self.output_dir = os.getcwd()
        self.output_filename = "output.srt"
        
        self.ocr_thread = None
        
        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # 1. 视频显示区域
        self.video_label = VideoLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: #222; border: 1px solid #555;")
        self.video_label.setMinimumSize(640, 360)
        self.video_label.setText("请点击下方按钮添加视频")
        self.video_label.setStyleSheet("QLabel { color : #888; background-color: #222; font-size: 20px; }")
        main_layout.addWidget(self.video_label, 1) # 权重1，自动拉伸

        # 2. 时间轴区域
        time_layout = QHBoxLayout()
        self.lbl_curr_time = QLabel("00:00:00")
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.sliderPressed.connect(self.slider_pressed)
        self.slider.sliderReleased.connect(self.slider_released)
        self.slider.valueChanged.connect(self.slider_value_changed)
        self.lbl_total_time = QLabel("00:00:00")
        
        time_layout.addWidget(self.lbl_curr_time)
        time_layout.addWidget(self.slider)
        time_layout.addWidget(self.lbl_total_time)
        main_layout.addLayout(time_layout)

        # 3. 设置区域
        setting_group = QGroupBox("保存设置")
        setting_layout = QHBoxLayout()
        
        self.btn_set_path = QPushButton("设置保存目录")
        self.btn_set_path.clicked.connect(self.choose_save_dir)
        self.le_save_path = QLineEdit(os.path.join(self.output_dir, self.output_filename))
        
        setting_layout.addWidget(self.btn_set_path)
        setting_layout.addWidget(self.le_save_path)
        setting_group.setLayout(setting_layout)
        main_layout.addWidget(setting_group)

        # 4. 控制按钮区域
        btn_layout = QHBoxLayout()
        
        self.btn_add = QPushButton("添加视频")
        self.btn_add.clicked.connect(self.open_video)
        
        self.btn_run = QPushButton("确定 / 开始提取")
        self.btn_run.clicked.connect(self.start_ocr)
        self.btn_run.setEnabled(False)
        self.btn_run.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 5px;")
        
        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.clicked.connect(self.cancel_ocr)
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.setStyleSheet("background-color: #f44336; color: white; padding: 5px;")

        btn_layout.addWidget(self.btn_add)
        btn_layout.addWidget(self.btn_run)
        btn_layout.addWidget(self.btn_cancel)
        main_layout.addLayout(btn_layout)
        
        # 5. 进度条
        progress_layout = QHBoxLayout()
        self.pbar = QProgressBar()
        self.pbar.setValue(0)
        self.lbl_status = QLabel("就绪")
        
        progress_layout.addWidget(self.pbar)
        progress_layout.addWidget(self.lbl_status)
        main_layout.addLayout(progress_layout)

        self.is_sliding = False

    # --- 逻辑功能 ---

    def open_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "Video Files (*.mp4 *.mkv *.avi *.mov)")
        if file_path:
            self.video_path = file_path
            self.load_video(file_path)
            
            # 默认保存到视频所在目录
            self.output_dir = os.path.dirname(os.path.abspath(file_path))

            # 自动生成默认输出路径
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            self.output_filename = f"{base_name}.srt"
            self.le_save_path.setText(os.path.join(self.output_dir, self.output_filename))
            
            self.btn_run.setEnabled(True)
            self.lbl_status.setText("视频已加载。请调整上方绿色框以选中字幕区域。")

    def load_video(self, path):
        if self.cap:
            self.cap.release()
            
        self.cap = cv2.VideoCapture(path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        self.slider.setRange(0, self.total_frames)
        self.slider.setValue(0)
        self.slider.setEnabled(True)
        
        # 更新总时间
        seconds = int(self.total_frames / self.fps) if self.fps > 0 else 0
        self.lbl_total_time.setText(str(datetime.timedelta(seconds=seconds)))
        
        # 显示第一帧
        self.show_frame(0)

    def show_frame(self, frame_idx):
        if not self.cap: return
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if ret:
            # BGR 转 RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            # 缩放以适应 Label
            lbl_w = self.video_label.width()
            lbl_h = self.video_label.height()
            
            # 保持比例缩放
            pixmap = QPixmap.fromImage(q_img)
            scaled_pixmap = pixmap.scaled(lbl_w, lbl_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            # 更新 VideoLabel 的缩放比例，以便计算 ROI
            if scaled_pixmap.width() > 0:
                self.video_label.scale_w = w / scaled_pixmap.width()
                self.video_label.scale_h = h / scaled_pixmap.height()
            
            self.video_label.setPixmap(scaled_pixmap)
            
            # 更新当前时间文字
            seconds = int(frame_idx / self.fps) if self.fps > 0 else 0
            self.lbl_curr_time.setText(str(datetime.timedelta(seconds=seconds)))

    def choose_save_dir(self):
        d = QFileDialog.getExistingDirectory(self, "选择保存目录")
        if d:
            self.output_dir = d
            self.le_save_path.setText(os.path.join(self.output_dir, self.output_filename))

    def slider_pressed(self):
        self.is_sliding = True

    def slider_released(self):
        self.is_sliding = False
        self.show_frame(self.slider.value())

    def slider_value_changed(self, val):
        if self.is_sliding: # 只有拖动时才实时更新预览，避免性能问题
            self.show_frame(val)

    def start_ocr(self):
        if not self.video_path: return
        
        out_path = self.le_save_path.text()
        roi = self.video_label.get_real_roi() # 获取针对原视频分辨率的 ROI
        
        if roi[2] < 10 or roi[3] < 10:
            QMessageBox.warning(self, "警告", "请先调整字幕选框！")
            return

        # UI 状态更新
        self.btn_run.setEnabled(False)
        self.btn_add.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.pbar.setValue(0)
        self.lbl_status.setText("正在初始化 RapidOCR...")

        # 启动线程
        self.ocr_thread = OCRWorker(self.video_path, out_path, roi)
        self.ocr_thread.progress_signal.connect(self.update_progress)
        self.ocr_thread.finished_signal.connect(self.ocr_finished)
        self.ocr_thread.error_signal.connect(self.ocr_error)
        self.ocr_thread.start()

    def cancel_ocr(self):
        if self.ocr_thread and self.ocr_thread.isRunning():
            self.ocr_thread.stop()
            self.lbl_status.setText("正在取消...")
            self.btn_cancel.setEnabled(False)

    def update_progress(self, val, msg):
        self.pbar.setValue(val)
        self.lbl_status.setText(msg)

    def ocr_finished(self, saved_path):
        self.pbar.setValue(100)
        self.lbl_status.setText("完成")
        QMessageBox.information(self, "成功", f"字幕提取成功！\n文件已保存至:\n{saved_path}")
        self.reset_ui()

    def ocr_error(self, err):
        self.lbl_status.setText("发生错误")
        QMessageBox.critical(self, "错误", f"提取过程中出错:\n{err}")
        self.reset_ui()

    def reset_ui(self):
        self.btn_run.setEnabled(True)
        self.btn_add.setEnabled(True)
        self.btn_cancel.setEnabled(False)

# ===========================
# 主入口
# ===========================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
