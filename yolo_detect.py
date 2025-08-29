# -*- coding: utf-8 -*-
#
# YOLO Rice Sorting GUI Application
# This script creates a graphical user interface (GUI) for real-time object detection using the YOLO model.
# It can be used to detect and sort objects like good and bad rice grains from a camera stream or video file.
#
# แอปพลิเคชัน GUI สำหรับคัดแยกเมล็ดข้าวด้วย YOLO
# สคริปต์นี้สร้างหน้าต่างโปรแกรม (GUI) สำหรับตรวจจับวัตถุแบบเรียลไทม์โดยใช้โมเดล YOLO
# สามารถนำไปใช้ตรวจจับและคัดแยกวัตถุต่างๆ เช่น เมล็ดข้าวดีและข้าวเสีย จากกล้องหรือไฟล์วิดีโอ

import sys
import os
import json
import time
from datetime import datetime
import threading

# Import winsound for beep sound on Windows.
# The try-except block ensures the application runs on other operating systems.
try:
    import winsound
except ImportError:
    winsound = None

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QPushButton,
                             QSlider, QVBoxLayout, QHBoxLayout, QGridLayout,
                             QLineEdit, QFileDialog, QMessageBox, QCheckBox, QComboBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QPoint, QRect, QObject
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen
import cv2
from ultralytics import YOLO

# --- Configuration File Handling ---
# ส่วนจัดการไฟล์ Config
# The configuration file stores user settings like model path and source path.
CONFIG_FILE = 'config.json'

def load_config():
    """Loads settings from the configuration file."""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}

def save_config(data):
    """Saves current settings to the configuration file."""
    with open(CONFIG_FILE, 'w') as f:
        json.dump(data, f, indent=4)

# --- Custom QLabel for ROI Selection ---
# คลาส QLabel แบบกำหนดเองสำหรับเลือก ROI
# This class handles mouse events to allow the user to draw a rectangle.
class ROISelectorLabel(QLabel):
    roi_selected = pyqtSignal(QRect)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.is_selecting = False
        self.start_point = QPoint()
        self.end_point = QPoint()
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.ArrowCursor)

    def start_selection(self):
        self.is_selecting = True
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.update()

    def mousePressEvent(self, event):
        if self.is_selecting and event.button() == Qt.MouseButton.LeftButton:
            self.start_point = event.pos()
            self.end_point = self.start_point
            self.update()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.is_selecting and event.buttons() == Qt.MouseButton.LeftButton:
            self.end_point = event.pos()
            self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.is_selecting and event.button() == Qt.MouseButton.LeftButton:
            self.is_selecting = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            selection_rect = QRect(self.start_point, self.end_point).normalized()
            self.roi_selected.emit(selection_rect)
            self.update()
        super().mouseReleaseEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        
        # Draw ROI selection if active
        if self.is_selecting:
            pen = QPen(Qt.GlobalColor.blue, 2, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawRect(QRect(self.start_point, self.end_point).normalized())
            
        # Draw center guide lines
        pen_guide = QPen(Qt.GlobalColor.red, 1, Qt.PenStyle.SolidLine)
        painter.setPen(pen_guide)
        
        # Get the size of the displayed pixmap
        pixmap = self.pixmap()
        if pixmap and not pixmap.isNull():
            # Calculate the center of the scaled pixmap within the label
            label_size = self.size()
            scaled_pixmap_size = pixmap.size().scaled(label_size, Qt.AspectRatioMode.KeepAspectRatio)
            offset_x = (label_size.width() - scaled_pixmap_size.width()) / 2
            offset_y = (label_size.height() - scaled_pixmap_size.height()) / 2
            
            center_x = int(offset_x + scaled_pixmap_size.width() / 2)
            center_y = int(offset_y + scaled_pixmap_size.height() / 2)
            
            # Draw horizontal line
            painter.drawLine(int(offset_x), center_y, int(offset_x + scaled_pixmap_size.width()), center_y)
            # Draw vertical line
            painter.drawLine(center_x, int(offset_y), center_x, int(offset_y + scaled_pixmap_size.height()))
        

    def clear_roi(self):
        self.is_selecting = False
        self.update()

# --- Worker Thread for YOLO Processing ---
# Worker Thread สำหรับการประมวลผล YOLO
class Worker(QObject):
    image_update = pyqtSignal(QPixmap)
    status_update = pyqtSignal(str)
    finished = pyqtSignal()
    
    def __init__(self, model_path, source, initial_thresh, is_image_source, autosave_enabled, target_fps, save_original_enabled, beep_enabled, roi=None):
        super().__init__()
        self.model_path = model_path
        self.source = source
        self.threshold = initial_thresh
        self.is_image_source = is_image_source
        self.autosave_enabled = autosave_enabled
        self.save_original_enabled = save_original_enabled
        self.beep_enabled = beep_enabled
        self.target_fps = target_fps
        self.roi = roi
        self._is_running = True
        self.video_writer = None
        self.is_recording = False
        self.recording_request = None
        self.is_paused = False
        
        self.session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.detection_summary = {}
        
        self.fps_buffer = []
        self.avg_fps = 0

        self.latest_frame = None
        self.lock = threading.Lock()

    def run(self):
        """Main loop for processing the video or image source."""
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

    def play_beep(self):
        """Plays a beep sound if enabled and on Windows."""
        if self.beep_enabled and winsound:
            winsound.Beep(800, 100) # Frequency = 800 Hz, Duration = 100 ms

    def process_single_image(self, model, labels):
        """Processes a single image file."""
        frame = cv2.imread(self.source)
        if frame is None:
            raise IOError(f"Cannot read image file: {self.source}")
        
        with self.lock:
            self.latest_frame = frame.copy()

        detected_frame = frame.copy()
        
        if self.roi and all(v >= 0 for v in self.roi):
            x, y, w, h = self.roi
            x = max(0, x)
            y = max(0, y)
            w = min(w, frame.shape[1] - x)
            h = min(h, frame.shape[0] - y)
            cropped_frame = frame[y:y+h, x:x+w]
        else:
            cropped_frame = frame
        
        object_found = False
        results = model(cropped_frame, verbose=False)
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
                
                # Offset bounding box coordinates to match the full frame if ROI is used
                if self.roi and all(v >= 0 for v in self.roi):
                    roi_x, roi_y, _, _ = self.roi
                    xmin += roi_x
                    ymin += roi_y
                    xmax += roi_x
                    ymax += roi_y
                
                # Draw the detection box in green
                cv2.rectangle(detected_frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(detected_frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw the blue ROI box after all detections, so it's on top.
        if self.roi and all(v >= 0 for v in self.roi):
            x, y, w, h = self.roi
            cv2.rectangle(detected_frame, (x, y), (x + w, y + h), (255, 0, 0), 3)

        if object_found:
            self.play_beep()
            if self.autosave_enabled:
                self.save_detection_images(frame, detected_frame, is_single_image=True)
                self.status_update.emit(f"Detection saved.")
            else:
                self.status_update.emit("Object detected (Auto-save is off).")
        else:
            self.status_update.emit("No objects detected in the image.")

        self.display_frame(detected_frame)

    def process_video_stream(self, model, labels):
        """Processes a video stream from a webcam or video file."""
        try:
            source_index = int(self.source)
            cap = cv2.VideoCapture(source_index)
        except ValueError:
            cap = cv2.VideoCapture(self.source)
        
        if not cap.isOpened():
            raise IOError(f"Cannot open source: {self.source}")

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

            with self.lock:
                self.latest_frame = frame.copy()

            detected_frame = frame.copy()
            object_found_in_frame = False

            self.handle_recording_request(detected_frame)
            
            if self.roi and all(v >= 0 for v in self.roi):
                x, y, w, h = self.roi
                x = max(0, x)
                y = max(0, y)
                w = min(w, frame.shape[1] - x)
                h = min(h, frame.shape[0] - y)
                cropped_frame = frame[y:y+h, x:x+w]
            else:
                cropped_frame = frame

            results = model(cropped_frame, verbose=False)
            detections = results[0].boxes
            object_count = 0
            
            # Offset bounding box coordinates to match the full frame if ROI is used
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
                    
                    if self.roi and all(v >= 0 for v in self.roi):
                        roi_x, roi_y, _, _ = self.roi
                        xmin += roi_x
                        ymin += roi_y
                        xmax += roi_x
                        ymax += roi_y

                    # Draw the detection box in green
                    cv2.rectangle(detected_frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    cv2.putText(detected_frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            if object_found_in_frame:
                self.play_beep()
                if self.autosave_enabled:
                    self.save_detection_images(frame, detected_frame)

            # Draw the blue ROI box after all detections, so it's on top.
            if self.roi and all(v >= 0 for v in self.roi):
                x, y, w, h = self.roi
                cv2.rectangle(detected_frame, (x, y), (x + w, y + h), (255, 0, 0), 3)

            # Display FPS and object count on the frame
            cv2.putText(detected_frame, f'Objects: {object_count}', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
            cv2.putText(detected_frame, f'FPS: {self.avg_fps:.2f}', (detected_frame.shape[1] - 150, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # Display recording indicator
            if self.is_recording:
                cv2.circle(detected_frame, (detected_frame.shape[1] - 30, 80), 10, (0, 0, 255), -1)
                cv2.putText(detected_frame, 'REC', (detected_frame.shape[1] - 80, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if self.is_recording and self.video_writer is not None:
                self.video_writer.write(detected_frame)

            self.display_frame(detected_frame)
            
            proc_time = time.time() - start_time
            delay = max(0, target_delay - proc_time)
            time.sleep(delay)

            end_time = time.time()
            if (end_time - start_time) > 0:
                actual_fps = 1 / (end_time - start_time)
                self.fps_buffer.append(actual_fps)
                if len(self.fps_buffer) > 30:
                    self.fps_buffer.pop(0)
                self.avg_fps = sum(self.fps_buffer) / len(self.fps_buffer)

        cap.release()

    def display_frame(self, frame):
        """Converts an OpenCV frame to a QPixmap and emits it for display."""
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.image_update.emit(pixmap)

    def stop(self):
        """Stops the worker thread."""
        self._is_running = False

    def set_pause_state(self, paused: bool):
        """Pauses or resumes the video stream."""
        self.is_paused = paused

    def save_detection_images(self, original_frame, detected_frame, is_single_image=False):
        """Saves detected frames to the 'outputs/detections' folder."""
        output_dir = os.path.join('outputs', 'detections', self.session_timestamp)
        os.makedirs(output_dir, exist_ok=True)
        timestamp = int(time.time() * 1000)
        
        detected_path = os.path.join(output_dir, f"detected_{timestamp}.png")
        cv2.imwrite(detected_path, detected_frame)
        
        if self.save_original_enabled:
            original_path = os.path.join(output_dir, f"detection_{timestamp}_original.png")
            cv2.imwrite(original_path, original_frame)

    def set_recording_state(self, state: bool):
        """Sets the recording state request."""
        self.recording_request = state

    def handle_recording_request(self, frame):
        """Handles starting and stopping of video recording."""
        if self.recording_request is not None:
            if self.recording_request is True and not self.is_recording:
                self.start_recording(frame)
            elif self.recording_request is False and self.is_recording:
                self.stop_recording()
            self.recording_request = None

    def start_recording(self, frame):
        """Starts a new video recording session."""
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
        """Stops the current video recording."""
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        self.is_recording = False
        self.status_update.emit("Recording stopped.")
    
    # NEW: Method to update ROI from the main thread
    def update_roi(self, new_roi):
        self.roi = new_roi


# --- Main GUI Window ---
# หน้าต่างหลักของโปรแกรม
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO Object Detection GUI")
        self.setGeometry(100, 100, 1000, 800)
        self.config = load_config()
        self.video_thread = QThread() 
        self.worker = None
        self.current_pixmap = None
        
        # MODIFIED: Load ROI from config on startup
        self.roi = self.config.get("roi", None)

        # --- Create Widgets ---
        # สร้าง Widgets ต่างๆ
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
        
        self.save_original_checkbox = QCheckBox("Save Original Frame")
        self.save_original_checkbox.setChecked(self.config.get("save_original", True))
        
        # Checkbox for beep sound
        self.beep_checkbox = QCheckBox("Beep on Detection")
        self.beep_checkbox.setChecked(self.config.get("beep", False))
        if not winsound:
            self.beep_checkbox.setEnabled(False)
            self.beep_checkbox.setToolTip("Only available on Windows")

        self.fps_label = QLabel("Processing Rate:")
        self.fps_combo = QComboBox()
        self.fps_combo.addItems(["Full Speed","30 FPS", "25 FPS" ,"20 FPS", "15 FPS", "10 FPS", "5 FPS", "2 FPS", "1 FPS"])
        self.fps_combo.setCurrentIndex(self.config.get("fps_index", 0))

        self.start_button = QPushButton("Start Detection")
        self.stop_button = QPushButton("Stop Detection")
        self.pause_button = QPushButton("Pause")
        self.pause_button.setCheckable(True)
        self.capture_button = QPushButton("Capture Frame")
        self.record_button = QPushButton("Start Recording")
        self.record_button.setCheckable(True)
        
        # ADDED: ROI buttons
        self.clear_roi_button = QPushButton("Clear ROI")
        self.clear_roi_button.setEnabled(False)
        self.set_roi_button = QPushButton("กำหนดพื้นที่ (ROI)")

        # Initially disable buttons that require a running stream
        self.stop_button.setEnabled(False)
        self.pause_button.setEnabled(False)
        self.capture_button.setEnabled(False)
        self.record_button.setEnabled(False)
        
        # MODIFIED: Use the custom ROILabel class
        self.image_display_label = ROISelectorLabel("Press 'Start Detection' to begin")
        self.image_display_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_display_label.setStyleSheet("background-color: black; color: white;")
        self.status_label = QLabel("Ready")

        # --- Layout Arrangement ---
        # จัด Layout
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
        control_layout.addWidget(self.fps_label, 3, 0)
        control_layout.addWidget(self.fps_combo, 3, 1)
        control_layout.addWidget(self.save_original_checkbox, 3, 2)
        control_layout.addWidget(self.beep_checkbox, 3, 3)
        
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.pause_button)
        button_layout.addWidget(self.capture_button)
        button_layout.addWidget(self.record_button)
        button_layout.addWidget(self.set_roi_button)
        button_layout.addWidget(self.clear_roi_button)
        
        main_layout = QVBoxLayout()
        main_layout.addLayout(control_layout)
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.image_display_label, 1)
        main_layout.addWidget(self.status_label)
        
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # --- Connect Signals to Slots ---
        # เชื่อมต่อ Signals กับ Slots
        self.model_browse_button.clicked.connect(self.browse_model_file)
        self.source_browse_button.clicked.connect(self.browse_source_file)
        self.thresh_slider.valueChanged.connect(self.update_threshold)
        self.start_button.clicked.connect(self.start_detection)
        self.stop_button.clicked.connect(self.stop_detection)
        self.pause_button.clicked.connect(self.toggle_pause)
        self.capture_button.clicked.connect(self.capture_frame)
        self.record_button.clicked.connect(self.toggle_recording)
        self.image_display_label.roi_selected.connect(self.set_roi)
        self.set_roi_button.clicked.connect(self.start_roi_selection)
        self.clear_roi_button.clicked.connect(self.clear_roi)

        # MODIFIED: Set the persistent ROI rectangle if loaded from config
        if self.roi:
            x, y, w, h = self.roi
            # The ROISelectorLabel no longer draws the persistent box, so this is no longer needed.
            self.status_label.setText(f"ROI loaded from config: x={x}, y={y}, w={w}, h={h}")
            self.clear_roi_button.setEnabled(True)
        else:
            self.status_label.setText("Ready")


    def start_roi_selection(self):
        # The ROISelectorLabel will draw the dashed line during selection.
        self.image_display_label.start_selection()
        self.status_label.setText("สถานะ: คลิกและลากเพื่อกำหนดพื้นที่ (ROI)")

    def browse_model_file(self):
        """Opens a file dialog to select a YOLO model (.pt file)."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Select YOLO Model", "", "PyTorch Model (*.pt)")
        if file_path:
            self.model_path_input.setText(file_path)

    def browse_source_file(self):
        """Opens a file dialog to select a video or image file."""
        filter_str = "All Supported Files (*.mp4 *.avi *.mov *.mkv *.wmv *.jpg *.jpeg *.png *.bmp);;" \
                     "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv);;" \
                     "Image Files (*.jpg *.jpeg *.png *.bmp)"
        
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Source File", "", filter_str)
        if file_path:
            self.source_path_input.setText(file_path)
            
    def update_threshold(self, value):
        """Updates the confidence threshold value and label."""
        thresh_float = value / 100.0
        self.thresh_value_label.setText(f"{thresh_float:.2f}")
        if self.worker:
            self.worker.threshold = thresh_float

    def start_detection(self):
        """Starts the object detection process in a new worker thread."""
        model_path = self.model_path_input.text()
        source = self.source_path_input.text()
        threshold = self.thresh_slider.value() / 100.0
        autosave_enabled = self.autosave_checkbox.isChecked()
        save_original_enabled = self.save_original_checkbox.isChecked()
        beep_enabled = self.beep_checkbox.isChecked()
        
        fps_text = self.fps_combo.currentText()
        if fps_text == "Full Speed":
            target_fps = 0
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
            self.clear_roi_button.setEnabled(True)
        
        # Clean up previous worker/thread before starting a new one
        if self.worker:
            self.worker.stop()
            self.video_thread.quit()
            self.video_thread.wait()
            self.worker.deleteLater()
        
        self.worker = Worker(model_path, source, threshold, is_image, autosave_enabled, target_fps, save_original_enabled, beep_enabled, self.roi)
        self.worker.moveToThread(self.video_thread)
        self.video_thread.started.connect(self.worker.run)
        self.worker.image_update.connect(self.update_image)
        self.worker.status_update.connect(self.update_status)
        self.worker.finished.connect(self.detection_finished)
        self.video_thread.start()
        
    def stop_detection(self):
        """Stops the worker thread and the detection process."""
        self.status_label.setText("Stopping...")
        if self.worker:
            self.worker.stop()
            self.video_thread.quit()
            self.video_thread.wait()
        
    def detection_finished(self):
        """Resets the GUI state and writes the detection summary file."""
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
        self.clear_roi_button.setEnabled(False)
        self.start_button.setEnabled(True)
        # Clear the worker instance
        self.worker = None

    def update_image(self, pixmap):
        """Updates the display label with the new frame."""
        self.current_pixmap = pixmap
        self.image_display_label.setPixmap(pixmap.scaled(
            self.image_display_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))

    def update_status(self, message):
        """Updates the status bar with current messages."""
        self.status_label.setText(message)

    def set_roi(self, selection_rect):
        """Receives ROI coordinates from the ROILabel and stores them."""
        # Use QPixmap to get the dimensions of the displayed image after scaling
        pixmap = self.image_display_label.pixmap()
        if not pixmap or pixmap.isNull():
            self.status_label.setText("Cannot set ROI: No image displayed.")
            return

        label_size = self.image_display_label.size()
        scaled_pixmap_size = pixmap.size().scaled(label_size, Qt.AspectRatioMode.KeepAspectRatio)
        offset_x = (label_size.width() - scaled_pixmap_size.width()) / 2
        offset_y = (label_size.height() - scaled_pixmap_size.height()) / 2
        
        if scaled_pixmap_size.width() == 0 or scaled_pixmap_size.height() == 0:
            self.status_label.setText("Cannot set ROI: Invalid image size.")
            return

        # Get original frame dimensions from worker for accurate scaling
        if self.worker and self.worker.latest_frame is not None:
             h_orig, w_orig = self.worker.latest_frame.shape[0], self.worker.latest_frame.shape[1]
        else:
             # Fallback if worker is not running, assume a common resolution
             h_orig, w_orig = 1080, 1920
             
        scale_x = w_orig / scaled_pixmap_size.width()
        scale_y = h_orig / scaled_pixmap_size.height()
        
        x1 = int((selection_rect.left() - offset_x) * scale_x)
        y1 = int((selection_rect.top() - offset_y) * scale_y)
        x2 = int((selection_rect.right() - offset_x) * scale_x)
        y2 = int((selection_rect.bottom() - offset_y) * scale_y)
        
        final_x1 = max(0, x1)
        final_y1 = max(0, y1)
        final_x2 = min(w_orig, x2)
        final_y2 = min(h_orig, y2)
        
        self.roi = (final_x1, final_y1, final_x2-final_x1, final_y2-final_y1)
        
        # Update worker's ROI immediately if it's running
        if self.worker:
            self.worker.update_roi(self.roi)
        
        self.status_label.setText(f"ROI selected: x={self.roi[0]}, y={self.roi[1]}, w={self.roi[2]}, h={self.roi[3]}")
        
        # Save ROI to config immediately after selection
        self.save_config_with_roi()
        self.clear_roi_button.setEnabled(True)

    
    def clear_roi(self):
        """Clears the selected ROI."""
        self.roi = None
        self.image_display_label.clear_roi()
        
        # Update worker's ROI immediately if it's running
        if self.worker:
            self.worker.update_roi(None)

        self.status_label.setText("ROI cleared. Detection will cover the full frame.")
        
        # Save cleared ROI to config
        self.save_config_with_roi()
        self.clear_roi_button.setEnabled(False)
    
    def capture_frame(self):
        """Captures and saves the currently displayed frame as an image."""
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
        """Writes a text file summarizing the total objects detected."""
        if not self.worker or not self.worker.detection_summary:
            self.status_label.setText("Finished. No objects were detected.")
            return

        summary = self.worker.detection_summary
        output_dir = 'outputs'
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"Result_{timestamp}.txt"
        save_path = os.path.join(output_dir, file_name)
        total_objects = sum(summary.values())

        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(f"Detection Summary - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*40 + "\n")
            for class_name, count in sorted(summary.items()):
                f.write(f"- {class_name}: {count}\n")
            f.write("="*40 + "\n")
            f.write(f"Total Objects Detected: {total_objects}\n")
        
        self.status_label.setText(f"Finished. Summary saved to {save_path}")

    def toggle_recording(self, checked):
        """Toggles the video recording on and off."""
        if self.worker:
            self.worker.set_recording_state(checked)
            if checked:
                self.record_button.setText("Stop Recording")
            else:
                self.record_button.setText("Start Recording")

    def toggle_pause(self, checked):
        """Pauses or resumes the detection process."""
        if self.worker:
            self.worker.set_pause_state(checked)
            if checked:
                self.pause_button.setText("Resume")
                self.status_label.setText("Paused.")
            else:
                self.pause_button.setText("Pause")
                self.status_label.setText("Processing...")

    def save_config_with_roi(self):
        """Saves the current settings and ROI to the config file."""
        self.config['model_path'] = self.model_path_input.text()
        self.config['source_path'] = self.source_path_input.text()
        self.config['threshold'] = self.thresh_slider.value() / 100.0
        self.config['autosave'] = self.autosave_checkbox.isChecked()
        self.config['fps_index'] = self.fps_combo.currentIndex()
        self.config['save_original'] = self.save_original_checkbox.isChecked()
        self.config['beep'] = self.beep_checkbox.isChecked()
        self.config['roi'] = self.roi # ADDED: Save the ROI tuple
        save_config(self.config)

    def closeEvent(self, event):
        """Saves settings and stops the worker thread when the application is closed."""
        self.save_config_with_roi()
        self.stop_detection()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())




