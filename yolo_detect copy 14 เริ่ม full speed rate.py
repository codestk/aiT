import sys
import os
import json
import time
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QPushButton, 
                             QSlider, QVBoxLayout, QHBoxLayout, QGridLayout, 
                             QLineEdit, QFileDialog, QMessageBox, QCheckBox, QComboBox)
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

    # MODIFIED: เพิ่ม target_fps
    def __init__(self, model_path, source, initial_thresh, is_image_source, autosave_enabled, target_fps):
        super().__init__()
        self.model_path = model_path
        self.source = source
        self.threshold = initial_thresh
        self.is_image_source = is_image_source
        self.autosave_enabled = autosave_enabled
        self.target_fps = target_fps # ADDED
        self._is_running = True
        self.video_writer = None
        self.is_recording = False
        self.recording_request = None
        self.is_paused = False
        
        self.session_timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.detection_summary = {}

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
            if self.is_recording:
                self.stop_recording()
            self.finished.emit()

    def process_single_image(self, model, labels):
        frame = cv2.imread(self.source)
        if frame is None:
            raise IOError(f"Cannot read image file: {self.source}")
        
        original_frame = frame.copy()
        object_found = False

        results = model(frame, verbose=False)
        detections = results[0].boxes
        
        for i in range(len(detections)):
            conf = detections[i].conf.item()
            if conf > self.threshold:
                object_found = True
                xyxy = detections[i].xyxy.cpu().numpy().squeeze().astype(int)
                xmin, ymin, xmax, ymax = xyxy
                classidx = int(detections[i].cls.item())
                classname = labels[classidx]
                self.detection_summary[classname] = self.detection_summary.get(classname, 0) + 1
                label = f'{classname}: {int(conf*100)}%'
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if object_found and self.autosave_enabled:
            base, ext = os.path.splitext(self.source)
            output_path_detected = os.path.join('outputs', f"{os.path.basename(base)}_detected{ext}")
            output_path_original = os.path.join('outputs', f"{os.path.basename(base)}_original{ext}")
            os.makedirs('outputs', exist_ok=True)
            
            cv2.imwrite(output_path_detected, frame)
            cv2.imwrite(output_path_original, original_frame)
            self.status_update.emit(f"Detection saved to outputs folder")
        elif object_found:
            self.status_update.emit("Object detected (Auto-save is off).")
        else:
            self.status_update.emit("No objects detected in the image.")

        self.display_frame(frame)

    def process_video_stream(self, model, labels):
        try:
            source_index = int(self.source)
            cap = cv2.VideoCapture(source_index)
        except ValueError:
            cap = cv2.VideoCapture(self.source)
        
        if not cap.isOpened():
            raise IOError(f"Cannot open source: {self.source}")

        # ADDED: Logic ควบคุม Frame Rate
        target_delay = 1.0 / self.target_fps if self.target_fps > 0 else 0

        self.status_update.emit("Processing...")
        while self._is_running:
            start_time = time.time()

            if self.is_paused:
                time.sleep(0.1)
                continue

            ret, frame = cap.read()
            if not ret:
                break 

            original_frame = frame.copy()
            object_found_in_frame = False

            self.handle_recording_request(frame)

            results = model(frame, verbose=False)
            detections = results[0].boxes
            object_count = 0
            
            for i in range(len(detections)):
                conf = detections[i].conf.item()
                if conf > self.threshold:
                    object_found_in_frame = True
                    object_count += 1
                    xyxy = detections[i].xyxy.cpu().numpy().squeeze().astype(int)
                    xmin, ymin, xmax, ymax = xyxy
                    classidx = int(detections[i].cls.item())
                    classname = labels[classidx]
                    self.detection_summary[classname] = self.detection_summary.get(classname, 0) + 1
                    label = f'{classname}: {int(conf*100)}%'
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            if object_found_in_frame and self.autosave_enabled:
                self.save_detection_images(original_frame, frame)

            cv2.putText(frame, f'Objects: {object_count}', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

            if self.is_recording:
                cv2.circle(frame, (frame.shape[1] - 30, 30), 10, (0, 0, 255), -1)
                cv2.putText(frame, 'REC', (frame.shape[1] - 70, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if self.is_recording and self.video_writer is not None:
                self.video_writer.write(frame)

            self.display_frame(frame)
            
            # ADDED: Logic ควบคุม Frame Rate
            proc_time = time.time() - start_time
            delay = max(0, target_delay - proc_time)
            time.sleep(delay)

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

    def set_pause_state(self, paused: bool):
        self.is_paused = paused

    def save_detection_images(self, original_frame, detected_frame):
        output_dir = os.path.join('outputs', 'detections', self.session_timestamp)
        os.makedirs(output_dir, exist_ok=True)
        timestamp = int(time.time() * 1000)
        detected_path = os.path.join(output_dir, f"detected_{timestamp}.png")
        cv2.imwrite(detected_path, detected_frame)

    def set_recording_state(self, state: bool):
        self.recording_request = state

    def handle_recording_request(self, frame):
        if self.recording_request is not None:
            if self.recording_request is True and not self.is_recording:
                self.start_recording(frame)
            elif self.recording_request is False and self.is_recording:
                self.stop_recording()
            self.recording_request = None

    def start_recording(self, frame):
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
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        self.is_recording = False
        self.status_update.emit("Recording stopped.")

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
        
        self.autosave_checkbox = QCheckBox("Auto-save Detections")
        self.autosave_checkbox.setChecked(self.config.get("autosave", False))
        
        # ADDED: Processing Rate ComboBox
        self.fps_label = QLabel("Processing Rate:")
        self.fps_combo = QComboBox()
        self.fps_combo.addItems(["Full Speed", "20 FPS", "15 FPS", "10 FPS", "5 FPS"])
        self.fps_combo.setCurrentIndex(self.config.get("fps_index", 0))

        self.start_button = QPushButton("Start Detection")
        self.stop_button = QPushButton("Stop Detection")
        self.pause_button = QPushButton("Pause")
        self.pause_button.setCheckable(True)
        self.capture_button = QPushButton("Capture Frame")
        self.record_button = QPushButton("Start Recording")
        self.record_button.setCheckable(True)
        self.stop_button.setEnabled(False)
        self.pause_button.setEnabled(False)
        self.capture_button.setEnabled(False)
        self.record_button.setEnabled(False)
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
        control_layout.addWidget(self.thresh_value_label, 2, 2)
        control_layout.addWidget(self.autosave_checkbox, 2, 3)
        control_layout.addWidget(self.fps_label, 3, 0) # ADDED
        control_layout.addWidget(self.fps_combo, 3, 1) # ADDED
        
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.pause_button)
        button_layout.addWidget(self.capture_button)
        button_layout.addWidget(self.record_button)
        
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
        self.pause_button.clicked.connect(self.toggle_pause)
        self.capture_button.clicked.connect(self.capture_frame)
        self.record_button.clicked.connect(self.toggle_recording)

    def browse_model_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select YOLO Model", "", "PyTorch Model (*.pt)")
        if file_path:
            self.model_path_input.setText(file_path)

    def browse_source_file(self):
        filter_str = "All Supported Files (*.mp4 *.avi *.mov *.mkv *.wmv *.jpg *.jpeg *.png *.bmp);;" \
                     "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv);;" \
                     "Image Files (*.jpg *.jpeg *.png *.bmp)"
        
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Source File", "", filter_str)
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
        autosave_enabled = self.autosave_checkbox.isChecked()
        
        # ADDED: Get target FPS from ComboBox
        fps_text = self.fps_combo.currentText()
        if fps_text == "Full Speed":
            target_fps = 0 # 0 means no delay
        else:
            target_fps = int(fps_text.split(" ")[0])

        if not os.path.exists(model_path):
            QMessageBox.critical(self, "Error", "Model file not found!")
            return

        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        is_image = any(source.lower().endswith(ext) for ext in image_extensions)
        
        self.start_button.setEnabled(False)
        if not is_image:
            self.stop_button.setEnabled(True)
            self.pause_button.setEnabled(True)
            self.capture_button.setEnabled(True)
            self.record_button.setEnabled(True)
        
        # MODIFIED: Pass target_fps to worker
        self.worker = Worker(model_path, source, threshold, is_image, autosave_enabled, target_fps)
        self.worker.image_update.connect(self.update_image)
        self.worker.status_update.connect(self.update_status)
        self.worker.finished.connect(self.detection_finished)
        self.worker.start()

    def stop_detection(self):
        self.status_label.setText("Stopping...")
        if self.worker:
            self.worker.stop()
        
    def detection_finished(self):
        self.write_summary_file()

        if hasattr(self.worker, 'is_image_source') and not self.worker.is_image_source:
                 self.image_display_label.setText("Processing finished. Press 'Start' to begin again.")
        self.stop_button.setEnabled(False)
        self.pause_button.setEnabled(False)
        self.pause_button.setChecked(False)
        self.pause_button.setText("Pause")
        self.capture_button.setEnabled(False)
        self.record_button.setEnabled(False)
        self.record_button.setChecked(False)
        self.record_button.setText("Start Recording")
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
    
    def write_summary_file(self):
        if not self.worker or not self.worker.detection_summary:
            self.status_label.setText("Finished. No objects were detected.")
            return

        summary = self.worker.detection_summary
        output_dir = 'outputs'
        os.makedirs(output_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        file_name = f"Result_{timestamp}.txt"
        save_path = os.path.join(output_dir, file_name)
        total_objects = sum(summary.values())

        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(f"Detection Summary - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*40 + "\n")
            for class_name, count in sorted(summary.items()):
                f.write(f"- {class_name}: {count}\n")
            f.write("="*40 + "\n")
            f.write(f"Total Objects Detected: {total_objects}\n")
        
        self.status_label.setText(f"Finished. Summary saved to {save_path}")

    def toggle_recording(self, checked):
        if self.worker:
            self.worker.set_recording_state(checked)
            if checked:
                self.record_button.setText("Stop Recording")
            else:
                self.record_button.setText("Start Recording")

    def toggle_pause(self, checked):
        if self.worker:
            self.worker.set_pause_state(checked)
            if checked:
                self.pause_button.setText("Resume")
                self.status_label.setText("Paused.")
            else:
                self.pause_button.setText("Pause")
                self.status_label.setText("Processing...")

    def closeEvent(self, event):
        self.config['model_path'] = self.model_path_input.text()
        self.config['source_path'] = self.source_path_input.text()
        self.config['threshold'] = self.thresh_slider.value() / 100.0
        self.config['autosave'] = self.autosave_checkbox.isChecked()
        self.config['fps_index'] = self.fps_combo.currentIndex() # ADDED
        save_config(self.config)
        self.stop_detection()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
