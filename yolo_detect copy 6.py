import sys
import os
import json
import time
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QPushButton, 
                             QSlider, QVBoxLayout, QHBoxLayout, QGridLayout, 
                             QLineEdit, QFileDialog, QMessageBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
import cv2
from ultralytics import YOLO

# --- ส่วนจัดการไฟล์ Config ---
CONFIG_FILE = 'config.json'

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}

def save_config(data):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(data, f, indent=4)

# --- Worker Thread สำหรับการประมวลผล YOLO ---
class Worker(QThread):
    image_update = pyqtSignal(QPixmap)
    status_update = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, model_path, source, initial_thresh, is_image_source):
        super().__init__()
        self.model_path = model_path
        self.source = source
        self.threshold = initial_thresh
        self.is_image_source = is_image_source
        self._is_running = True
        self.video_writer = None # ADDED: สำหรับจัดการการบันทึกวิดีโอ
        self.is_recording = False # ADDED: สถานะการบันทึก
        self.recording_request = None # ADDED: คำสั่งเริ่ม/หยุดบันทึกจาก GUI

    def run(self):
        try:
            self.status_update.emit("Loading YOLO model...")
            model = YOLO(self.model_path)
            labels = model.names
            self.status_update.emit(f"Model loaded. Opening source: {self.source}")

            if self.is_image_source:
                self.process_single_image(model, labels)
            else:
                self.process_video_stream(model, labels)
                
        except Exception as e:
            self.status_update.emit(f"Error: {e}")
        finally:
            # ADDED: ตรวจสอบให้แน่ใจว่าหยุดบันทึกเมื่อ thread จบการทำงาน
            if self.is_recording:
                self.stop_recording()
            self.finished.emit()

    def process_single_image(self, model, labels):
        frame = cv2.imread(self.source)
        if frame is None:
            raise IOError(f"Cannot read image file: {self.source}")

        results = model(frame, verbose=False)
        detections = results[0].boxes
        
        for i in range(len(detections)):
            conf = detections[i].conf.item()
            if conf > self.threshold:
                xyxy = detections[i].xyxy.cpu().numpy().squeeze().astype(int)
                xmin, ymin, xmax, ymax = xyxy
                classidx = int(detections[i].cls.item())
                classname = labels[classidx]
                label = f'{classname}: {int(conf*100)}%'
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        base, ext = os.path.splitext(self.source)
        output_path = f"{base}_output{ext}"
        cv2.imwrite(output_path, frame)
        self.status_update.emit(f"Output saved to: {output_path}")
        self.display_frame(frame)

    def process_video_stream(self, model, labels):
        try:
            source_index = int(self.source)
            cap = cv2.VideoCapture(source_index)
        except ValueError:
            cap = cv2.VideoCapture(self.source)
        
        if not cap.isOpened():
            raise IOError(f"Cannot open source: {self.source}")

        self.status_update.emit("Processing...")
        while self._is_running:
            ret, frame = cap.read()
            if not ret:
                break 

            # ADDED: ตรวจสอบและจัดการคำสั่งบันทึกวิดีโอ
            self.handle_recording_request(frame)

            results = model(frame, verbose=False)
            detections = results[0].boxes
            object_count = 0
            
            for i in range(len(detections)):
                conf = detections[i].conf.item()
                if conf > self.threshold:
                    object_count += 1
                    xyxy = detections[i].xyxy.cpu().numpy().squeeze().astype(int)
                    xmin, ymin, xmax, ymax = xyxy
                    classidx = int(detections[i].cls.item())
                    classname = labels[classidx]
                    label = f'{classname}: {int(conf*100)}%'
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            cv2.putText(frame, f'Objects: {object_count}', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

            # ADDED: แสดงสถานะการบันทึกบนวิดีโอ
            if self.is_recording:
                cv2.circle(frame, (frame.shape[1] - 30, 30), 10, (0, 0, 255), -1)
                cv2.putText(frame, 'REC', (frame.shape[1] - 70, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # ADDED: เขียนเฟรมลงไฟล์วิดีโอถ้ากำลังบันทึก
            if self.is_recording and self.video_writer is not None:
                self.video_writer.write(frame)

            self.display_frame(frame)
            time.sleep(0.01)
        cap.release()

    def display_frame(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.image_update.emit(pixmap)

    def stop(self):
        self._is_running = False

    # --- ADDED: ฟังก์ชันทั้งหมดที่เกี่ยวกับการบันทึกวิดีโอ ---
    def set_recording_state(self, state: bool):
        """รับคำสั่งจาก GUI เพื่อเริ่มหรือหยุดบันทึก"""
        self.recording_request = state

    def handle_recording_request(self, frame):
        """ตรวจสอบและดำเนินการตามคำสั่งบันทึก"""
        if self.recording_request is not None:
            if self.recording_request is True and not self.is_recording:
                self.start_recording(frame)
            elif self.recording_request is False and self.is_recording:
                self.stop_recording()
            self.recording_request = None

    def start_recording(self, frame):
        """เริ่มต้นการบันทึกวิดีโอ"""
        output_dir = 'outputs'
        os.makedirs(output_dir, exist_ok=True)
        timestamp = int(time.time())
        file_name = f"record_{timestamp}.avi"
        save_path = os.path.join(output_dir, file_name)
        
        h, w, _ = frame.shape
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.video_writer = cv2.VideoWriter(save_path, fourcc, 20.0, (w, h))
        self.is_recording = True
        self.status_update.emit(f"Recording started, saving to {save_path}")

    def stop_recording(self):
        """หยุดการบันทึกวิดีโอ"""
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        self.is_recording = False
        self.status_update.emit("Recording stopped.")
    # --------------------------------------------------------

# --- หน้าต่างหลักของโปรแกรม ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO Object Detection GUI")
        self.setGeometry(100, 100, 1000, 800)
        self.config = load_config()
        self.worker = None
        self.current_pixmap = None

        # --- สร้าง Widgets ---
        self.model_label = QLabel("YOLO Model Path:")
        self.model_path_input = QLineEdit(self.config.get("model_path", ""))
        self.model_browse_button = QPushButton("Browse...")
        self.source_label = QLabel("Source (File/Webcam Index):")
        self.source_path_input = QLineEdit(self.config.get("source_path", "0"))
        self.source_browse_button = QPushButton("Browse File...")
        self.thresh_label = QLabel("Confidence Threshold:")
        self.thresh_slider = QSlider(Qt.Orientation.Horizontal)
        self.thresh_slider.setRange(0, 100)
        initial_thresh = int(self.config.get("threshold", 0.5) * 100)
        self.thresh_slider.setValue(initial_thresh)
        self.thresh_value_label = QLabel(f"{initial_thresh/100:.2f}")
        self.start_button = QPushButton("Start Detection")
        self.stop_button = QPushButton("Stop Detection")
        self.capture_button = QPushButton("Capture Frame")
        self.record_button = QPushButton("Start Recording") # ADDED
        self.record_button.setCheckable(True) # ADDED: ทำให้เป็นปุ่ม Toggle
        self.stop_button.setEnabled(False)
        self.capture_button.setEnabled(False)
        self.record_button.setEnabled(False) # ADDED
        self.image_display_label = QLabel("Press 'Start Detection' to begin")
        self.image_display_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_display_label.setStyleSheet("background-color: black; color: white;")
        self.status_label = QLabel("Ready")

        # --- จัด Layout ---
        control_layout = QGridLayout()
        control_layout.addWidget(self.model_label, 0, 0)
        control_layout.addWidget(self.model_path_input, 0, 1, 1, 2)
        control_layout.addWidget(self.model_browse_button, 0, 3)
        control_layout.addWidget(self.source_label, 1, 0)
        control_layout.addWidget(self.source_path_input, 1, 1, 1, 2)
        control_layout.addWidget(self.source_browse_button, 1, 3)
        control_layout.addWidget(self.thresh_label, 2, 0)
        control_layout.addWidget(self.thresh_slider, 2, 1)
        control_layout.addWidget(self.thresh_value_label, 2, 2, 1, 2)
        
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.capture_button)
        button_layout.addWidget(self.record_button) # ADDED
        
        main_layout = QVBoxLayout()
        main_layout.addLayout(control_layout)
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.image_display_label, 1)
        main_layout.addWidget(self.status_label)
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # --- เชื่อมต่อ Signals กับ Slots ---
        self.model_browse_button.clicked.connect(self.browse_model_file)
        self.source_browse_button.clicked.connect(self.browse_source_file)
        self.thresh_slider.valueChanged.connect(self.update_threshold)
        self.start_button.clicked.connect(self.start_detection)
        self.stop_button.clicked.connect(self.stop_detection)
        self.capture_button.clicked.connect(self.capture_frame)
        self.record_button.clicked.connect(self.toggle_recording) # ADDED

    def browse_model_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select YOLO Model", "", "PyTorch Model (*.pt)")
        if file_path:
            self.model_path_input.setText(file_path)

    def browse_source_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Source File", "", "Video/Image Files (*.mp4 *.avi *.jpg *.jpeg *.png *.bmp)")
        if file_path:
            self.source_path_input.setText(file_path)
            
    def update_threshold(self, value):
        thresh_float = value / 100.0
        self.thresh_value_label.setText(f"{thresh_float:.2f}")
        if self.worker:
            self.worker.threshold = thresh_float

    def start_detection(self):
        model_path = self.model_path_input.text()
        source = self.source_path_input.text()
        threshold = self.thresh_slider.value() / 100.0

        if not os.path.exists(model_path):
            QMessageBox.critical(self, "Error", "Model file not found!")
            return

        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        is_image = any(source.lower().endswith(ext) for ext in image_extensions)
        
        self.start_button.setEnabled(False)
        if not is_image:
            self.stop_button.setEnabled(True)
            self.capture_button.setEnabled(True)
            self.record_button.setEnabled(True) # MODIFIED
        
        self.worker = Worker(model_path, source, threshold, is_image)
        self.worker.image_update.connect(self.update_image)
        self.worker.status_update.connect(self.update_status)
        self.worker.finished.connect(self.detection_finished)
        self.worker.start()

    def stop_detection(self):
        self.status_label.setText("Stopping...")
        if self.worker:
            self.worker.stop()
        self.stop_button.setEnabled(False)
        self.capture_button.setEnabled(False)
        self.record_button.setEnabled(False) # MODIFIED
        self.record_button.setChecked(False) # MODIFIED: รีเซ็ตสถานะปุ่ม
        self.record_button.setText("Start Recording") # MODIFIED
        self.start_button.setEnabled(True)

    def detection_finished(self):
        self.status_label.setText("Finished.")
        if not self.worker.is_image_source:
             self.image_display_label.setText("Processing finished. Press 'Start' to begin again.")
        self.stop_button.setEnabled(False)
        self.capture_button.setEnabled(False)
        self.record_button.setEnabled(False) # MODIFIED
        self.record_button.setChecked(False) # MODIFIED
        self.record_button.setText("Start Recording") # MODIFIED
        self.start_button.setEnabled(True)

    def update_image(self, pixmap):
        self.current_pixmap = pixmap
        self.image_display_label.setPixmap(pixmap.scaled(
            self.image_display_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))

    def update_status(self, message):
        self.status_label.setText(message)

    def capture_frame(self):
        if self.current_pixmap is None:
            self.status_label.setText("No active stream to capture.")
            return
        output_dir = 'outputs'
        os.makedirs(output_dir, exist_ok=True)
        timestamp = int(time.time())
        file_name = f"capture_{timestamp}.png"
        save_path = os.path.join(output_dir, file_name)
        if self.current_pixmap.save(save_path):
            self.status_label.setText(f"Frame captured and saved to {save_path}")
        else:
            self.status_label.setText(f"Failed to save frame to {save_path}")

    # ADDED: ฟังก์ชันใหม่สำหรับจัดการปุ่ม Record
    def toggle_recording(self, checked):
        if self.worker:
            self.worker.set_recording_state(checked)
            if checked:
                self.record_button.setText("Stop Recording")
            else:
                self.record_button.setText("Start Recording")

    def closeEvent(self, event):
        self.config['model_path'] = self.model_path_input.text()
        self.config['source_path'] = self.source_path_input.text()
        self.config['threshold'] = self.thresh_slider.value() / 100.0
        save_config(self.config)
        self.stop_detection()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())