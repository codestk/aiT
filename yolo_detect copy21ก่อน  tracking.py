# How to force MJPG on your USB camera in this GUI
# -------------------------------------------------
# This version adds a "Codec" dropdown (Auto / MJPG / YUY2 / H264) and applies the
# chosen FOURCC to the camera when opening it (DSHOW→MSMF→ANY fallback). It also
# logs the resulting FOURCC actually set by the driver, along with FPS and
# resolution, so you can confirm MJPG is active.
#
# Notes (Windows):
# - Many UVC cameras only expose certain resolution+FPS+codec combinations.
#   If setting MJPG fails at 1920×1080@60, try 1280×720 or 1920×1080@30.
# - With MSMF some devices ignore CAP_PROP_FOURCC; DSHOW often honors it better.
# - Set resolution before FOURCC (some drivers require that ordering).
# - After setting, always read back CAP_PROP_FOURCC to verify.
#
# ✅ What’s added:
# 1) GUI: Codec dropdown (self.codec_combo) with more options.
# 2) Save/restore codec to config.json
# 3) Camera open: sets resolution → FOURCC → (optional) FPS, then logs actual values
# 4) Helper to convert FOURCC int→string for readable status
# 5) MODIFIED: Camera backend preference changed to DSHOW -> MSMF -> ANY.
# 6) MODIFIED: Desired height is now hardcoded to 120px.
# 7) MODIFIED: Added more codec options to the dropdown.
# 8) MODIFIED: Improved the FOURCC to string conversion to be more robust.


import sys
import os
import json
import time
from datetime import datetime
import threading
import struct # For robust FOURCC conversion

try:
    import winsound
except ImportError:
    winsound = None

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QSlider, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLineEdit, QFileDialog, QMessageBox, QCheckBox, QComboBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QPoint, QRect, QObject
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen
import cv2
from ultralytics import YOLO
import torch
import numpy as np

# ---- Perf tweaks ----
try:
    torch.backends.cudnn.benchmark = True
except Exception:
    pass
try:
    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('high')
except Exception:
    pass

MODEL_CACHE = {}

# MODIFIED: Added more common codecs
CODEC_MAP = {
    "Auto": None,
    "MJPG": "MJPG",
    "YUY2": "YUY2",
    "H264": "H264",
    "DIVX": "DIVX",
    "XVID": "XVID",
    "MP4V": "MP4V",
    "I420": "I420", # YUV420 format
}

# MODIFIED: Made this function more robust to handle non-standard values
def int_fourcc_to_str(val:int) -> str:
    """Converts a FOURCC integer code to a readable string using struct, handles errors."""
    if not isinstance(val, (int, float)) or int(val) == 0:
        return "----"
    try:
        val_int = int(val)
        # Handle potential negative values from some drivers by converting to unsigned 32-bit
        if val_int < 0:
            val_int += (1 << 32)
        # Pack the integer into 4 bytes (little-endian) and decode as ASCII.
        # 'replace' will insert a placeholder for any byte that is not valid ASCII.
        return struct.pack('<I', val_int).decode('ascii', errors='replace').strip()
    except (struct.error, OverflowError):
        # If the int is too large or causes a packing error, show its raw hex value.
        return f"Raw({hex(int(val))})"


def load_yolo_model_cached(model_path):
    print(f"Entering load_yolo_model_cached")
    try:
        # ตรวจสอบว่าโมเดลอยู่ในแคชหรือไม่
        if model_path in MODEL_CACHE:
            print(f"Using cached model for {model_path}")
            return MODEL_CACHE[model_path]
        
        # ตรวจสอบว่าไฟล์โมเดลมีอยู่จริงหรือไม่
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        # ตรวจสอบนามสกุลไฟล์
        if not model_path.lower().endswith('.pt'):
            print(f"Warning: Model file {model_path} does not have .pt extension. Attempting to load anyway.")
        
        print(f"Loading YOLO model from {model_path}...")
        model = YOLO(model_path)
        print(f"YOLO model loaded successfully")
        
        # ตั้งค่า CUDA ถ้ามี
        if torch.cuda.is_available():
            try:
                print(f"Moving model to CUDA...")
                model.to('cuda:0')
                try:
                    model.fuse()
                    print(f"Model layers fused successfully")
                except Exception as e:
                    print(f"Warning: Could not fuse model layers: {e}")
                try:
                    model.model.half()
                    print(f"Model converted to half precision")
                except Exception as e:
                    print(f"Warning: Could not convert model to half precision: {e}")
            except Exception as e:
                print(f"Warning: Could not move model to CUDA: {e}")
        
        # ทดสอบโมเดลด้วยข้อมูลจำลอง
        try:
            print(f"Testing model with dummy input...")
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            _ = model(dummy, verbose=False, device='cuda:0' if torch.cuda.is_available() else 'cpu', half=True if torch.cuda.is_available() else False)
            print(f"Model test successful")
        except Exception as e:
            print(f"Warning: Model test failed: {e}")
            # ไม่ raise ข้อผิดพลาดที่นี่ เพราะโมเดลอาจยังทำงานได้กับข้อมูลจริง
        
        # เก็บโมเดลในแคช
        MODEL_CACHE[model_path] = model
        return model
        
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        raise Exception(f"ไม่สามารถโหลดโมเดล YOLO ได้: {str(e)}")

CONFIG_FILE = 'config.json'

def load_config():
    print(f"Entering load_config")
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}

def save_config(data):
    print(f"Entering save_config")
    with open(CONFIG_FILE, 'w') as f:
        json.dump(data, f, indent=4)

class ROISelectorLabel(QLabel):
    roi_selected = pyqtSignal(QRect)
    roi_polygon_selected = pyqtSignal(list)
    def __init__(self, parent=None):
        #print(f"Entering ROISelectorLabel.__init__")
        super().__init__(parent)
        self.mode = 'none'  # 'none' | 'rect' | 'poly'
        self.is_selecting = False
        self.start_point = QPoint(); self.end_point = QPoint()
        self.poly_points: list[QPoint] = []
        self.preview_point: QPoint|None = None
        self.setMouseTracking(True); self.setCursor(Qt.CursorShape.ArrowCursor)
    def start_selection(self):
        #print(f"Entering ROISelectorLabel.start_selection (rect)")
        self.mode = 'rect'
        self.is_selecting = True; self.setCursor(Qt.CursorShape.CrossCursor); self.update()
    def start_freeform_selection(self):
        #print(f"Entering ROISelectorLabel.start_freeform_selection (poly)")
        self.mode = 'poly'
        self.is_selecting = True
        self.poly_points = []
        self.preview_point = None
        self.setCursor(Qt.CursorShape.CrossCursor); self.update()
    def mousePressEvent(self, e):
        #print(f"Entering ROISelectorLabel.mousePressEvent")
        if self.mode == 'rect' and self.is_selecting and e.button() == Qt.MouseButton.LeftButton:
            self.start_point = e.pos(); self.end_point = self.start_point; self.update()
        elif self.mode == 'poly' and self.is_selecting and e.button() == Qt.MouseButton.LeftButton:
            # Left-click while drawing polygon: add point or auto-close if near start
            click_pt = e.pos()
            self.preview_point = click_pt
            # If close to the first point and we already have at least 3 points, auto-finish
            try:
                close_thresh = 12  # pixels on the label coordinates
                if len(self.poly_points) >= 3:
                    p0 = self.poly_points[0]
                    dx = click_pt.x() - p0.x(); dy = click_pt.y() - p0.y()
                    if (dx*dx + dy*dy) ** 0.5 <= close_thresh:
                        # Finish polygon without adding another vertex on top of p0
                        self.is_selecting = False; self.setCursor(Qt.CursorShape.ArrowCursor)
                        self.roi_polygon_selected.emit(self.poly_points.copy())
                        self.poly_points = []
                        self.preview_point = None
                        self.mode = 'none'
                        self.update()
                        super().mousePressEvent(e)
                        return
            except Exception:
                pass
            # Otherwise keep collecting points
            self.poly_points.append(click_pt); self.update()
        elif self.mode == 'poly' and self.is_selecting and e.button() == Qt.MouseButton.RightButton:
            # Finish polygon on right-click if >= 3 points
            if len(self.poly_points) >= 3:
                self.is_selecting = False; self.setCursor(Qt.CursorShape.ArrowCursor)
                self.roi_polygon_selected.emit(self.poly_points.copy())
                self.poly_points = []
                self.preview_point = None
                self.mode = 'none'
                self.update()
        super().mousePressEvent(e)
    def mouseDoubleClickEvent(self, e):
        # Double-click to finish polygon as well
        if self.mode == 'poly' and self.is_selecting and len(self.poly_points) >= 3:
            self.is_selecting = False; self.setCursor(Qt.CursorShape.ArrowCursor)
            self.roi_polygon_selected.emit(self.poly_points.copy())
            self.poly_points = []
            self.preview_point = None
            self.mode = 'none'
            self.update()
        super().mouseDoubleClickEvent(e)
    def mouseMoveEvent(self, e):
        #print(f"Entering ROISelectorLabel.mouseMoveEvent")
        if self.mode == 'rect' and self.is_selecting and e.buttons() == Qt.MouseButton.LeftButton:
            self.end_point = e.pos(); self.update()
        elif self.mode == 'poly' and self.is_selecting:
            self.preview_point = e.pos(); self.update()
        super().mouseMoveEvent(e)
    def mouseReleaseEvent(self, e):
        #print(f"Entering ROISelectorLabel.mouseReleaseEvent")
        if self.mode == 'rect' and self.is_selecting and e.button() == Qt.MouseButton.LeftButton:
            self.is_selecting = False; self.setCursor(Qt.CursorShape.ArrowCursor)
            self.roi_selected.emit(QRect(self.start_point, self.end_point).normalized())
            self.mode = 'none'
            self.update()
        super().mouseReleaseEvent(e)
    def paintEvent(self, e):
        #print(f"Entering ROISelectorLabel.paintEvent")
        super().paintEvent(e)
        p = QPainter(self)
        if self.mode == 'rect' and self.is_selecting:
            p.setPen(QPen(Qt.GlobalColor.red, 2, Qt.PenStyle.DashLine))
            p.drawRect(QRect(self.start_point, self.end_point).normalized())
        elif self.mode == 'poly' and self.is_selecting:
            p.setPen(QPen(Qt.GlobalColor.green, 2, Qt.PenStyle.SolidLine))
            # Draw points and segments
            for i in range(1, len(self.poly_points)):
                p.drawLine(self.poly_points[i-1], self.poly_points[i])
            for pt in self.poly_points:
                p.drawEllipse(pt, 2, 2)
            # Preview line to current mouse; if cursor is near the start point,
            # hint that the shape will close.
            if self.preview_point is not None and len(self.poly_points) > 0:
                p.setPen(QPen(Qt.GlobalColor.green, 1, Qt.PenStyle.DashLine))
                try:
                    if len(self.poly_points) >= 3:
                        p0 = self.poly_points[0]
                        dx = self.preview_point.x() - p0.x(); dy = self.preview_point.y() - p0.y()
                        if (dx*dx + dy*dy) ** 0.5 <= 12:
                            p.drawLine(self.poly_points[-1], p0)
                        else:
                            p.drawLine(self.poly_points[-1], self.preview_point)
                    else:
                        p.drawLine(self.poly_points[-1], self.preview_point)
                except Exception:
                    p.drawLine(self.poly_points[-1], self.preview_point)
        # Center crosshair
        p.setPen(QPen(Qt.GlobalColor.red, 1))
        pm = self.pixmap()
        if pm and not pm.isNull():
            L = self.size(); S = pm.size().scaled(L, Qt.AspectRatioMode.KeepAspectRatio)
            ox = (L.width()-S.width())/2; oy=(L.height()-S.height())/2
            cx = int(ox+S.width()/2); cy=int(oy+S.height()/2); arm=40
            p.drawLine(cx-arm, cy, cx+arm, cy); p.drawLine(cx, cy-arm, cx, cy+arm)
    def clear_roi(self):
        print(f"Entering ROISelectorLabel.clear_roi")
        self.is_selecting=False; self.mode='none'; self.poly_points=[]; self.preview_point=None; self.update()

class Worker(QObject):
    image_update = pyqtSignal(QPixmap)
    status_update = pyqtSignal(str)
    finished = pyqtSignal()
    def __init__(self, model_path, source, initial_thresh, is_image_source,
                 autosave_enabled, target_fps, save_original_enabled, beep_enabled,
                 roi=None, roi_poly=None, desired_resolution=None, is_webcam_source=False,
                 low_latency_mode=False, codec_choice: str|None=None):
        print(f"Entering Worker.__init__")
        super().__init__()
        self.model_path=model_path; self.source=source; self.threshold=initial_thresh
        self.is_image_source=is_image_source; self.autosave_enabled=autosave_enabled
        self.save_original_enabled=save_original_enabled; self.beep_enabled=beep_enabled
        self.target_fps=target_fps; self.roi=roi; self.roi_poly=roi_poly; self.desired_resolution=desired_resolution
        self.is_webcam_source=is_webcam_source; self.low_latency_mode=low_latency_mode
        self.codec_choice = codec_choice  # 'MJPG'/'YUY2'/'H264'/None
        self._is_running=True; self.video_writer=None; self.is_recording=False
        self.recording_request=None; self.is_paused=False
        self.session_timestamp=datetime.now().strftime('%Y%m%d_%H%M%S')
        self.detection_summary={}; self.fps_buffer=[]; self.avg_fps=0
        self.total_objects_detected = 0  # Cumulative counter for all objects
        self.latest_frame=None; self.lock=threading.Lock()
    def run(self):
        print(f"Entering Worker.run")
        try:
            # แจ้งสถานะการโหลดโมเดล
            self.status_update.emit("กำลังโหลดโมเดล YOLO...")
            
            try:
                # โหลดโมเดล YOLO
                model = load_yolo_model_cached(self.model_path)
                labels = model.names
                self.status_update.emit(f"โหลดโมเดลสำเร็จ กำลังเปิดแหล่งข้อมูล: {self.source}")
                
                # ตั้งค่า CUDA ถ้ามี
                if torch.cuda.is_available():
                    try:
                        model.to('cuda:0')
                        try: 
                            model.fuse()
                        except Exception as e: 
                            print(f"Warning: Could not fuse model layers: {e}")
                        try: 
                            model.model.half()
                        except Exception as e: 
                            print(f"Warning: Could not convert model to half precision: {e}")
                    except Exception as e: 
                        print(f"Warning: Could not move model to CUDA: {e}")
                
                # ประมวลผลภาพหรือวิดีโอ
                if self.is_image_source: 
                    self.process_single_image(model, labels)
                else: 
                    self.process_video_stream(model, labels)
                    
            except FileNotFoundError as e:
                # กรณีไม่พบไฟล์โมเดล
                self.status_update.emit(f"ไม่พบไฟล์โมเดล: {e}")
                print(f"Model file not found: {e}")
                
            except Exception as e:
                # กรณีเกิดข้อผิดพลาดในการโหลดโมเดล
                self.status_update.emit(f"เกิดข้อผิดพลาดในการโหลดโมเดล: {e}")
                print(f"Error loading model: {e}")
                
        except Exception as e:
            # กรณีเกิดข้อผิดพลาดทั่วไป
            error_msg = f"เกิดข้อผิดพลาด: {e}"
            self.status_update.emit(error_msg)
            print(f"Error in Worker.run: {e}")
            
        finally:
            # ปิดการบันทึกวิดีโอถ้ากำลังบันทึกอยู่
            if self.is_recording:
                self.stop_recording()
            # แจ้งว่าการทำงานเสร็จสิ้น
            self.finished.emit()
            print("Worker.run finished.")
    def play_beep(self):
        #print(f"Entering Worker.play_beep")
        if self.beep_enabled and winsound: winsound.Beep(800,100)
    def _apply_camera_settings(self, cap):
        print(f"Entering Worker._apply_camera_settings")
        start_time = time.time()

        # Resolution first
        if self.is_webcam_source and self.desired_resolution:
            try:
                w,h = self.desired_resolution
                print(f"Attempting to set resolution to {w}x{h}")
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(w))
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(h))
                print(f"Set resolution took: {time.time() - start_time:.4f} seconds")
            except Exception as e:
                print(f"Error setting resolution: {e}")
            start_time = time.time() # Reset timer for next operation

        # FOURCC next
        try:
            if self.codec_choice:
                fourcc = cv2.VideoWriter_fourcc(*self.codec_choice)
                print(f"Attempting to set FOURCC to {self.codec_choice}")
                cap.set(cv2.CAP_PROP_FOURCC, fourcc)
                print(f"Set FOURCC took: {time.time() - start_time:.4f} seconds")
        except Exception as e:
            print(f"Error setting FOURCC: {e}")
        start_time = time.time() # Reset timer for next operation

        # FPS last
        try:
            if self.target_fps>0:
                print(f"Attempting to set FPS to {self.target_fps}")
                cap.set(cv2.CAP_PROP_FPS, float(self.target_fps))
                print(f"Set FPS took: {time.time() - start_time:.4f} seconds")
        except Exception as e:
            print(f"Error setting FPS: {e}")
        start_time = time.time() # Reset timer for next operation

        # report actuals
        try:
            rw=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); rh=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            rf=cap.get(cv2.CAP_PROP_FPS); fc=int_fourcc_to_str(int(cap.get(cv2.CAP_PROP_FOURCC)))
            self.status_update.emit(f"Camera set → {rw}x{rh} @{rf:.1f} FPS, FOURCC={fc}")
            print(f"Get actuals and emit status took: {time.time() - start_time:.4f} seconds")
        except Exception as e:
            print(f"Error getting actual camera settings: {e}")
    def process_single_image(self, model, labels):
        print(f"Entering Worker.process_single_image")
        frame=cv2.imread(self.source)
        if frame is None: raise IOError(f"Cannot read image file: {self.source}")
        with self.lock: self.latest_frame=frame.copy()
        detected_frame=frame.copy()
        use_masked=False
        if self.roi_poly and isinstance(self.roi_poly, (list, tuple)) and len(self.roi_poly) >= 3:
            # Apply polygon mask
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            pts = np.array(self.roi_poly, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
            cropped_frame = cv2.bitwise_and(frame, frame, mask=mask)
            use_masked=True
        elif self.roi and all(v>=0 for v in self.roi):
            x,y,w,h=self.roi; x=max(0,x); y=max(0,y); w=min(w, frame.shape[1]-x); h=min(h, frame.shape[0]-y)
            cropped_frame=frame[y:y+h, x:x+w]
        else:
            cropped_frame=frame
        results=model(cropped_frame, verbose=False, device='cuda:0' if torch.cuda.is_available() else 'cpu', half=True if torch.cuda.is_available() else False)
        detections=results[0].boxes; object_found=False
        for i in range(len(detections)):
            conf=detections[i].conf.item()
            if conf>self.threshold:
                object_found=True
                xmin,ymin,xmax,ymax = detections[i].xyxy.cpu().numpy().squeeze().astype(int)
                cls=int(detections[i].cls.item()); name=labels[cls]; label=f"{name}: {int(conf*100)}%"
                if (not use_masked) and self.roi and all(v>=0 for v in self.roi):
                    rx,ry,_,_=self.roi; xmin+=rx; ymin+=ry; xmax+=rx; ymax+=ry
                self.detection_summary[name]=self.detection_summary.get(name,0)+1
                self.total_objects_detected += 1
                cv2.rectangle(detected_frame,(xmin,ymin),(xmax,ymax),(0,0,255),2)
                cv2.putText(detected_frame,label,(xmin,ymin-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
        if self.roi_poly and isinstance(self.roi_poly, (list, tuple)) and len(self.roi_poly) >= 3:
            pts = np.array(self.roi_poly, dtype=np.int32)
            cv2.polylines(detected_frame, [pts], isClosed=True, color=(0,255,0), thickness=2)
        elif self.roi and all(v>=0 for v in self.roi):
            x,y,w,h=self.roi; cv2.rectangle(detected_frame,(x,y),(x+w,y+h),(0,0,255),3)
        # Count objects in this frame
        frame_count = sum(1 for i in range(len(detections)) if detections[i].conf.item() > self.threshold)
        cv2.putText(detected_frame,f'Frame: {frame_count} | Total: {self.total_objects_detected}',(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        self.status_update.emit("Detection saved." if object_found and self.autosave_enabled else ("Object detected (Auto-save is off)." if object_found else "No objects detected in the image."))
        cv2.putText(detected_frame,f'Res: {detected_frame.shape[1]}x{detected_frame.shape[0]}',(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,255),2)
        self.display_frame(detected_frame)
    def process_video_stream(self, model, labels):
        print(f"Entering Worker.process_video_stream")
        cap = None
        try:
            # พยายามเปิดกล้องหรือวิดีโอ
            try:
                idx = int(self.source)
                # ลองใช้ DSHOW ก่อนตามที่ร้องขอ
                cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW); backend = 'DSHOW'
                if not cap.isOpened():
                    cap.release(); cap = cv2.VideoCapture(idx, cv2.CAP_MSMF); backend = 'MSMF'
                if not cap.isOpened():
                    cap.release(); cap = cv2.VideoCapture(idx); backend = 'ANY'

                if cap.isOpened():
                    try:
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    except Exception as e:
                        print(f"Warning: Could not set buffer size: {e}")
                    self._apply_camera_settings(cap)
                    self.status_update.emit(f"เปิดกล้องเว็บแคมโดยใช้ backend: {backend}{' (โหมดความหน่วงต่ำ)' if self.low_latency_mode else ''}")
            except ValueError:
                # ถ้าไม่ใช่ตัวเลข ให้ถือว่าเป็นไฟล์วิดีโอ
                cap = cv2.VideoCapture(self.source)
                if cap.isOpened():
                    self.status_update.emit(f"เปิดไฟล์วิดีโอ: {self.source}")
            
            # ตรวจสอบว่าเปิดแหล่งข้อมูลได้หรือไม่
            if not cap or not cap.isOpened():
                error_msg = f"ไม่สามารถเปิดแหล่งข้อมูล: {self.source}"
                self.status_update.emit(error_msg)
                print(error_msg)
                return
                
            # ตั้งค่าความล่าช้าเป้าหมายตาม FPS
            target_delay = 1.0 / self.target_fps if self.target_fps > 0 else 0
            self.status_update.emit("กำลังประมวลผล...")
            
            # ตัวแปรสำหรับตรวจสอบข้อผิดพลาดต่อเนื่อง
            consecutive_errors = 0
            max_consecutive_errors = 5
            
            # วนลูปประมวลผลวิดีโอ
            while self._is_running:
                try:
                    t0 = time.time()
                    
                    # ตรวจสอบการหยุดชั่วคราว
                    if self.is_paused:
                        time.sleep(0.1)
                        continue
                        
                    # โหมดความหน่วงต่ำ
                    if self.low_latency_mode:
                        try: 
                            cap.grab()
                        except Exception as e: 
                            print(f"Warning: Error in low latency grab: {e}")
                    
                    # อ่านเฟรม
                    ret, frame = cap.read()
                    if not ret:
                        # ถ้าอ่านเฟรมไม่สำเร็จ
                        if isinstance(self.source, str) and self.source.isdigit():
                            # ถ้าเป็นกล้อง ให้ลองเชื่อมต่อใหม่
                            self.status_update.emit("กล้องถูกตัดการเชื่อมต่อ กำลังพยายามเชื่อมต่อใหม่...")
                            cap.release()
                            time.sleep(1)  # รอสักครู่ก่อนลองเชื่อมต่อใหม่
                            
                            # ลองเปิดกล้องใหม่ด้วย backend ต่างๆ
                            idx = int(self.source)
                            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW); backend = 'DSHOW'
                            if not cap.isOpened():
                                cap.release(); cap = cv2.VideoCapture(idx, cv2.CAP_MSMF); backend = 'MSMF'
                            if not cap.isOpened():
                                cap.release(); cap = cv2.VideoCapture(idx); backend = 'ANY'
                                
                            if cap.isOpened():
                                self._apply_camera_settings(cap)
                                self.status_update.emit(f"เชื่อมต่อกล้องใหม่สำเร็จโดยใช้ backend: {backend}")
                                continue
                            else:
                                self.status_update.emit("ไม่สามารถเชื่อมต่อกล้องใหม่ได้")
                                break
                        else:
                            # ถ้าเป็นไฟล์วิดีโอและถึงจุดสิ้นสุดแล้ว
                            self.status_update.emit("ถึงจุดสิ้นสุดของไฟล์วิดีโอแล้ว")
                            break
                    
                    # รีเซ็ตตัวนับข้อผิดพลาดเมื่ออ่านเฟรมได้สำเร็จ
                    consecutive_errors = 0
                    
                    # บันทึกเฟรมล่าสุด
                    with self.lock: 
                        self.latest_frame = frame.copy()
                    
                    # สร้างสำเนาเฟรมสำหรับการตรวจจับ
                    detected_frame = frame.copy()
                    found = False
                    
                    # จัดการคำขอบันทึก
                    self.handle_recording_request(detected_frame)
                    
                    # ประมวลผลตาม ROI ที่กำหนด
                    try:
                        use_masked = False
                        # ใช้ ROI แบบหลายเหลี่ยม
                        if self.roi_poly and isinstance(self.roi_poly, (list, tuple)) and len(self.roi_poly) >= 3:
                            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                            pts = np.array(self.roi_poly, dtype=np.int32)
                            cv2.fillPoly(mask, [pts], 255)
                            source_for_model = cv2.bitwise_and(frame, frame, mask=mask)
                            use_masked = True
                        # ใช้ ROI แบบสี่เหลี่ยม
                        elif self.roi and all(v >= 0 for v in self.roi):
                            x, y, w, h = self.roi
                            # ตรวจสอบว่า ROI อยู่ในขอบเขตของเฟรม
                            x = max(0, x)
                            y = max(0, y)
                            w = min(w, frame.shape[1] - x)
                            h = min(h, frame.shape[0] - y)
                            source_for_model = frame[y:y+h, x:x+w]
                        # ใช้เฟรมทั้งหมด
                        else:
                            source_for_model = frame
                            
                        # ประมวลผลโมเดล
                        results = model(source_for_model, verbose=False, 
                                      device='cuda:0' if torch.cuda.is_available() else 'cpu', 
                                      half=True if torch.cuda.is_available() else False)
                        
                        # ประมวลผลการตรวจจับ
                        detections = results[0].boxes
                        count = 0
                        
                        # วาดกรอบสี่เหลี่ยมล้อมรอบวัตถุที่ตรวจพบ
                        for i in range(len(detections)):
                            try:
                                conf = detections[i].conf.item()
                                if conf > self.threshold:
                                    found = True
                                    count += 1
                                    # ป้องกันการเกิด division by zero หรือข้อผิดพลาดในการแปลงค่า
                                    try:
                                        box_data = detections[i].xyxy.cpu().numpy()
                                        # ตรวจสอบว่า box_data มีข้อมูลและไม่เป็น NaN
                                        if box_data.size > 0 and not np.isnan(box_data).any():
                                            xmin, ymin, xmax, ymax = box_data.squeeze().astype(int)
                                        else:
                                            print(f"Warning: Invalid box data for detection {i}")
                                            continue
                                    except Exception as box_error:
                                        print(f"Error extracting box coordinates: {box_error}")
                                        continue
                                    cls = int(detections[i].cls.item())
                                    name = labels[cls]
                                    label = f"{name}: {int(conf*100)}%"
                                    
                                    # ปรับพิกัดถ้าใช้ ROI
                                    if (not use_masked) and self.roi and all(v >= 0 for v in self.roi):
                                        rx, ry, _, _ = self.roi
                                        xmin += rx
                                        ymin += ry
                                        xmax += rx
                                        ymax += ry
                                        
                                    # อัปเดตสรุปการตรวจจับ
                                    self.detection_summary[name] = self.detection_summary.get(name, 0) + 1
                                    self.total_objects_detected += 1
                                    
                                    # วาดกรอบสี่เหลี่ยมและป้ายกำกับ
                                    cv2.rectangle(detected_frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                                    cv2.putText(detected_frame, label, (xmin, ymin-10), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                            except Exception as e:
                                print(f"Error processing detection {i}: {e}")
                                continue
                                
                        # เล่นเสียงบี๊ปและบันทึกภาพถ้าพบวัตถุ
                        if found:
                            self.play_beep()
                            if self.autosave_enabled: 
                                self.save_detection_images(frame, detected_frame)
                                
                        # วาด ROI บนเฟรม
                        if self.roi_poly and isinstance(self.roi_poly, (list, tuple)) and len(self.roi_poly) >= 3:
                            pts = np.array(self.roi_poly, dtype=np.int32)
                            cv2.polylines(detected_frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                        elif self.roi and all(v >= 0 for v in self.roi):
                            x, y, w, h = self.roi
                            cv2.rectangle(detected_frame, (x, y), (x+w, y+h), (0, 0, 255), 3)
                            
                        # เพิ่มข้อมูลบนเฟรม
                        cv2.putText(detected_frame, f'Frame: {count} | Total: {self.total_objects_detected}', 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        cv2.putText(detected_frame, f'FPS: {self.avg_fps:.2f}', 
                                   (detected_frame.shape[1]-150, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        cv2.putText(detected_frame, f'Res: {detected_frame.shape[1]}x{detected_frame.shape[0]}', 
                                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                                   
                        # แสดงสถานะการบันทึก
                        if self.is_recording:
                            cv2.circle(detected_frame, (detected_frame.shape[1]-30, 80), 10, (0, 0, 255), -1)
                            cv2.putText(detected_frame, 'REC', (detected_frame.shape[1]-80, 85), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                       
                        # บันทึกวิดีโอถ้าเปิดใช้งาน
                        if self.is_recording and self.video_writer is not None:
                            try:
                                self.video_writer.write(detected_frame)
                            except Exception as e:
                                print(f"Error writing to video: {e}")
                                self.stop_recording()
                                self.status_update.emit(f"เกิดข้อผิดพลาดในการบันทึกวิดีโอ: {e}")
                                
                        # แสดงเฟรมที่ประมวลผลแล้ว
                        self.display_frame(detected_frame)
                        
                    except Exception as e:
                        # กรณีเกิดข้อผิดพลาดในการประมวลผล ROI หรือโมเดล
                        print(f"Error processing frame with model: {e}")
                        # แสดงข้อความข้อผิดพลาดบนเฟรม
                        cv2.putText(frame, f"Error: {str(e)[:50]}", (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        self.display_frame(frame)
                    
                    # คำนวณความล่าช้าและ FPS
                    proc = time.time() - t0
                    delay = max(0, target_delay - proc)
                    time.sleep(delay)
                    t1 = time.time()
                    
                    time_diff = t1 - t0
                    if time_diff > 0.0001:  # ป้องกันการหารด้วยศูนย์หรือค่าที่ใกล้ศูนย์มาก
                        fps = 1 / time_diff
                        self.fps_buffer.append(fps)
                        if len(self.fps_buffer) > 30: 
                            self.fps_buffer.pop(0)
                        if len(self.fps_buffer) > 0:  # ป้องกันการหารด้วยศูนย์ถ้า buffer ว่าง
                            self.avg_fps = sum(self.fps_buffer) / len(self.fps_buffer)
                        
                except Exception as e:
                    # กรณีเกิดข้อผิดพลาดในการประมวลผลเฟรม
                    consecutive_errors += 1
                    error_msg = f"เกิดข้อผิดพลาดในการประมวลผลเฟรม: {e}"
                    print(error_msg)
                    
                    # หยุดถ้าเกิดข้อผิดพลาดต่อเนื่องเกินกำหนด
                    if consecutive_errors >= max_consecutive_errors:
                        self.status_update.emit(f"เกิดข้อผิดพลาดต่อเนื่องเกินกำหนด: {error_msg}")
                        break
                    
                    # รอสักครู่ก่อนลองอีกครั้ง
                    time.sleep(0.5)
            
        except Exception as e:
            # กรณีเกิดข้อผิดพลาดทั่วไป
            error_msg = f"เกิดข้อผิดพลาดในการประมวลผลวิดีโอ: {e}"
            self.status_update.emit(error_msg)
            print(error_msg)
            
        finally:
            # ปิดการเชื่อมต่อกล้องหรือวิดีโอ
            if cap is not None:
                cap.release()
    def display_frame(self, frame):
        #print(f"Entering Worker.display_frame")
        rgb=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h,w,ch=rgb.shape; qimg=QImage(rgb.data,w,h,ch*w,QImage.Format.Format_RGB888)
        self.image_update.emit(QPixmap.fromImage(qimg))
    def stop(self):
        print(f"Entering Worker.stop")
        self._is_running=False
    def set_pause_state(self, paused:bool):
        print(f"Entering Worker.set_pause_state")
        self.is_paused=paused
    def save_detection_images(self, original, detected, is_single_image=False):
        #print(f"Entering Worker.save_detection_images")
        # Base output directory per session
        base_out = os.path.join('outputs', 'detections', self.session_timestamp)
        # Separate folders for detected (with boxes) and original frames
        detect_dir = os.path.join(base_out, 'detect')
        original_dir = os.path.join(base_out, 'original')
        os.makedirs(detect_dir, exist_ok=True)
        if self.save_original_enabled:
            os.makedirs(original_dir, exist_ok=True)

        # Use the same filename in both folders to pair images easily
        ts = int(time.time() * 1000)
        filename = f"detection_{ts}.png"

        # Save detected image
        cv2.imwrite(os.path.join(detect_dir, filename), detected)
        # Optionally save original image with the same name
        if self.save_original_enabled:
            cv2.imwrite(os.path.join(original_dir, filename), original)
    def set_recording_state(self, state:bool):
        print(f"Entering Worker.set_recording_state")
        self.recording_request=state
    def handle_recording_request(self, frame):
        #print(f"Entering Worker.handle_recording_request")
        if self.recording_request is not None:
            if self.recording_request and not self.is_recording: self.start_recording(frame)
            elif (not self.recording_request) and self.is_recording: self.stop_recording()
            self.recording_request=None
    def start_recording(self, frame):
        print(f"Entering Worker.start_recording")
        out='outputs'; os.makedirs(out, exist_ok=True)
        ts=int(time.time()); path=os.path.join(out, f"record_{ts}.avi")
        h,w,_=frame.shape; fourcc=cv2.VideoWriter_fourcc(*'XVID')
        self.video_writer=cv2.VideoWriter(path, fourcc, 20.0, (w,h)); self.is_recording=True
        self.status_update.emit(f"Recording started, saving to {path}")
    def stop_recording(self):
        print(f"Entering Worker.stop_recording")
        if self.video_writer:
            self.video_writer.release(); self.video_writer=None
        self.is_recording=False; self.status_update.emit("Recording stopped.")
    def update_roi(self, new_roi):
        print(f"Entering Worker.update_roi")
        self.roi=new_roi
    def update_roi_poly(self, new_poly):
        print(f"Entering Worker.update_roi_poly")
        self.roi_poly=new_poly

class MainWindow(QMainWindow):
    def __init__(self):
        print(f"Entering MainWindow.__init__")
        super().__init__()
        self.setWindowTitle("YOLO Object Detection GUI")
        self.setGeometry(100,100,1000,800)
        self.config=load_config(); self.video_thread=QThread(); self.worker=None
        self.current_pixmap=None; self.roi=self.config.get("roi", None)
        self.roi_poly=self.config.get("roi_poly", None)
        # Widgets
        self.model_label=QLabel("YOLO Model Path:"); self.model_path_input=QLineEdit(self.config.get("model_path",""))
        self.model_browse_button=QPushButton("Browse...")
        self.source_label=QLabel("Source (File/Webcam Index):"); self.source_path_input=QLineEdit(self.config.get("source_path","0"))
        self.source_browse_button=QPushButton("Browse File...")
        self.list_cams_button=QPushButton("List Cameras")
        self.thresh_label=QLabel("Confidence Threshold:"); self.thresh_slider=QSlider(Qt.Orientation.Horizontal)
        self.thresh_slider.setRange(0,100); init_t=int(self.config.get("threshold",0.5)*100)
        self.thresh_slider.setValue(init_t); self.thresh_value_label=QLabel(f"{init_t/100:.2f}")
        self.autosave_checkbox=QCheckBox("Auto-save Detections"); self.autosave_checkbox.setChecked(self.config.get("autosave",False))
        self.save_original_checkbox=QCheckBox("Save Original Frame"); self.save_original_checkbox.setChecked(self.config.get("save_original",True))
        self.beep_checkbox=QCheckBox("Beep on Detection"); self.beep_checkbox.setChecked(self.config.get("beep",False))
        if not winsound: self.beep_checkbox.setEnabled(False); self.beep_checkbox.setToolTip("Only available on Windows")
        self.low_latency_checkbox=QCheckBox("Low Latency (optimize camera startup)"); self.low_latency_checkbox.setChecked(self.config.get("low_latency",True))
        self.fps_label=QLabel("Processing Rate:"); self.fps_combo=QComboBox(); self.fps_combo.addItems(["Full Speed","60 FPS","50 FPS","30 FPS","25 FPS","20 FPS","15 FPS","10 FPS","5 FPS","2 FPS","1 FPS"])
        self.fps_combo.setCurrentIndex(self.config.get("fps_index",0))
        
        self.codec_label=QLabel("Codec:")
        self.codec_combo=QComboBox(); self.codec_combo.addItems(list(CODEC_MAP.keys()))
        self.codec_combo.setCurrentText(self.config.get("codec","Auto"))
        self.resolution_label=QLabel("Resolution:"); self.resolution_combo=QComboBox()
        self.resolution_options=["Source/Native","2592x1440","2304x1296","1920x1080","1600x900","1280x720","1024x576","960x540","800x450","640x480"]
        self.resolution_combo.addItems(self.resolution_options)
        self.resolution_combo.setCurrentIndex(self.config.get("resolution_index",0))
        self.start_button=QPushButton("Start Detection"); self.stop_button=QPushButton("Stop Detection")
        self.pause_button=QPushButton("Pause"); self.pause_button.setCheckable(True)
        self.capture_button=QPushButton("Capture Frame"); self.record_button=QPushButton("Start Recording"); self.record_button.setCheckable(True)
        # Add freeform ROI button (Pen-like)
        self.set_roi_poly_button=QPushButton("วาดเส้น ROI (อิสระ)")
        self.clear_roi_button=QPushButton("Clear ROI"); self.clear_roi_button.setEnabled(False)
        self.set_roi_button=QPushButton("กำหนดพื้นที่ (ROI)")
        self.image_display_label=ROISelectorLabel("Press 'Start Detection' to begin")
        self.image_display_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_display_label.setStyleSheet("background-color: black; color: white;")
        self.status_label=QLabel("Ready")
        # Layout
        grid=QGridLayout()
        grid.addWidget(self.model_label,0,0); grid.addWidget(self.model_path_input,0,1,1,2); grid.addWidget(self.model_browse_button,0,3)
        grid.addWidget(self.source_label,1,0); grid.addWidget(self.source_path_input,1,1,1,1); grid.addWidget(self.source_browse_button,1,2); grid.addWidget(self.list_cams_button,1,3)
        grid.addWidget(self.thresh_label,2,0); grid.addWidget(self.thresh_slider,2,1); grid.addWidget(self.thresh_value_label,2,2); grid.addWidget(self.autosave_checkbox,2,3)
        grid.addWidget(self.fps_label,3,0); grid.addWidget(self.fps_combo,3,1); grid.addWidget(self.save_original_checkbox,3,2); grid.addWidget(self.beep_checkbox,3,3)
        grid.addWidget(self.resolution_label,4,0); grid.addWidget(self.resolution_combo,4,1); grid.addWidget(self.low_latency_checkbox,4,2,1,2)
        grid.addWidget(self.codec_label,5,0); grid.addWidget(self.codec_combo,5,1)
        h=QHBoxLayout();
        for w in [self.start_button,self.stop_button,self.pause_button,self.capture_button,self.record_button,self.set_roi_button,self.set_roi_poly_button,self.clear_roi_button]: h.addWidget(w)
        v=QVBoxLayout(); v.addLayout(grid); v.addLayout(h); v.addWidget(self.image_display_label,1); v.addWidget(self.status_label)
        c=QWidget(); c.setLayout(v); self.setCentralWidget(c)
        # Signals
        self.model_browse_button.clicked.connect(self.browse_model_file)
        self.source_browse_button.clicked.connect(self.browse_source_file)
        self.list_cams_button.clicked.connect(self.list_cameras_dialog)
        self.thresh_slider.valueChanged.connect(self.update_threshold)
        self.start_button.clicked.connect(self.start_detection)
        self.stop_button.clicked.connect(self.stop_detection)
        self.pause_button.clicked.connect(self.toggle_pause)
        self.capture_button.clicked.connect(self.capture_frame)
        self.record_button.clicked.connect(self.toggle_recording)
        self.image_display_label.roi_selected.connect(self.set_roi)
        self.image_display_label.roi_polygon_selected.connect(self.set_roi_polygon)
        self.set_roi_button.clicked.connect(self.start_roi_selection)
        try:
            self.set_roi_poly_button.clicked.connect(self.start_roi_poly_selection)
        except Exception:
            pass
        self.clear_roi_button.clicked.connect(self.clear_roi)
        if self.roi_poly:
            self.status_label.setText(f"ROI (poly) loaded from config: {len(self.roi_poly)} points"); self.clear_roi_button.setEnabled(True)
        elif self.roi:
            x,y,w,h=self.roi; self.status_label.setText(f"ROI loaded from config: x={x}, y={y}, w={w}, h={h}"); self.clear_roi_button.setEnabled(True)
        else:
            self.status_label.setText("Ready")
    # camera listing
    def _try_open_camera_with_backends(self, index:int):
        print(f"Entering MainWindow._try_open_camera_with_backends")
        # MODIFIED: Try DSHOW first as requested.
        for backend,name in [(cv2.CAP_DSHOW,"DSHOW"),(cv2.CAP_MSMF,"MSMF"),(0,"ANY")]:
            try:
                cap=cv2.VideoCapture(index, backend) if backend!=0 else cv2.VideoCapture(index)
                if not cap.isOpened():
                    cap.release(); continue
                try: cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
                except Exception: pass
                ok,_=cap.read()
                if not ok:
                    cap.release(); continue
                return cap,name
            except Exception:
                try: cap.release()
                except Exception: pass
        return None,None
    def list_cameras_dialog(self, max_index:int=10):
        print(f"Entering MainWindow.list_cameras_dialog")
        lines=[]
        for idx in range(max_index):
            cap,backend=self._try_open_camera_with_backends(idx)
            if cap is None: continue
            try:
                rw=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); rh=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                rf=cap.get(cv2.CAP_PROP_FPS); fc=int_fourcc_to_str(int(cap.get(cv2.CAP_PROP_FOURCC)))
            except Exception:
                rw,rh,rf,fc=0,0,0.0,"----"
            lines.append(f"Index {idx}: {rw}x{rh} @ ~{rf:.1f} FPS, FOURCC={fc}, backend={backend}")
            cap.release()
        if not lines:
            QMessageBox.information(self, "List Cameras", f"No cameras detected (0..{max_index-1})"); return
        msg="Available camera indices:\n"+"\n".join(lines)+"\n\n(ใส่เลข index นี้ในช่อง Source ได้เลย)"
        QMessageBox.information(self, "List Cameras", msg)
    # GUI actions
    def start_roi_selection(self):
        print(f"Entering MainWindow.start_roi_selection")
        self.image_display_label.start_selection(); self.status_label.setText("สถานะ: คลิกและลากเพื่อกำหนดพื้นที่ (ROI)")
    def start_roi_poly_selection(self):
        print(f"Entering MainWindow.start_roi_poly_selection")
        self.image_display_label.start_freeform_selection(); self.status_label.setText("สถานะ: คลิกเพิ่มจุดทีละจุด แล้วดับเบิลคลิก/คลิกขวาเพื่อจบเส้น ROI")
    def browse_model_file(self):
        print(f"Entering MainWindow.browse_model_file")
        p,_=QFileDialog.getOpenFileName(self,"Select YOLO Model","","PyTorch Model (*.pt)")
        if p: self.model_path_input.setText(p)
    def browse_source_file(self):
        print(f"Entering MainWindow.browse_source_file")
        filt=("All Supported Files (*.mp4 *.avi *.mov *.mkv *.wmv *.jpg *.jpeg *.png *.bmp);;"
              "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv);;"
              "Image Files (*.jpg *.jpeg *.png *.bmp)")
        p,_=QFileDialog.getOpenFileName(self,"Select Source File","",filt)
        if p: self.source_path_input.setText(p)
    def update_threshold(self, v:int):
        #print(f"Entering MainWindow.update_threshold")
        t=v/100.0; self.thresh_value_label.setText(f"{t:.2f}")
        if self.worker: self.worker.threshold=t
    def start_detection(self):
        print(f"Entering MainWindow.start_detection")
        try:
            # ตรวจสอบพารามิเตอร์ต่างๆ
            model_path=self.model_path_input.text(); source=self.source_path_input.text()
            
            # ตรวจสอบว่ามีการระบุ model_path และ source หรือไม่
            if not model_path.strip():
                QMessageBox.critical(self,"Error","กรุณาระบุไฟล์โมเดล YOLO"); return
            if not source.strip():
                QMessageBox.critical(self,"Error","กรุณาระบุแหล่งที่มาของภาพหรือวิดีโอ"); return
                
            threshold=self.thresh_slider.value()/100.0
            autosave=self.autosave_checkbox.isChecked(); save_orig=self.save_original_checkbox.isChecked()
            beep=self.beep_checkbox.isChecked(); lowlat=self.low_latency_checkbox.isChecked()
            fps_text=self.fps_combo.currentText(); target_fps=0 if fps_text=="Full Speed" else int(fps_text.split(" ")[0])
            # codec
            codec_key=self.codec_combo.currentText(); codec_choice=CODEC_MAP.get(codec_key)
            # webcam + resolution
            is_webcam=False; desired_res=None
            try: 
                idx = int(source)
                is_webcam=True
                # ตรวจสอบว่า index ของกล้องถูกต้องหรือไม่
                if idx < 0:
                    QMessageBox.warning(self,"Warning","ค่า index ของกล้องควรเป็นจำนวนเต็มบวก"); return
            except ValueError: 
                is_webcam=False
                
            rtxt=self.resolution_combo.currentText()
            if is_webcam and rtxt!="Source/Native":
                try:
                    wstr,hstr=rtxt.split("x")
                    desired_res=(int(wstr), int(hstr))
                    
                    # --- MODIFICATION START ---
                    # Hardcode the desired height to 120 as requested.
                    if desired_res is not None:
                        #desired_res = (desired_res[0], 120)
                        desired_res = (desired_res[0], 1000)
                    # --- MODIFICATION END ---

                except Exception as e:
                    print(f"Error parsing resolution: {e}")
                    desired_res=None

            # ตรวจสอบว่าไฟล์โมเดลมีอยู่จริงหรือไม่
            if not os.path.exists(model_path):
                QMessageBox.critical(self,"Error","ไม่พบไฟล์โมเดล! กรุณาตรวจสอบที่อยู่ไฟล์"); return
                
            # ตรวจสอบว่าเป็นไฟล์ภาพหรือไม่
            exts=['.jpg','.jpeg','.png','.bmp']; is_image=any(source.lower().endswith(e) for e in exts)
            if not is_webcam and not is_image and not os.path.exists(source):
                QMessageBox.critical(self,"Error","ไม่พบไฟล์วิดีโอหรือภาพ! กรุณาตรวจสอบที่อยู่ไฟล์"); return
                
            self.start_button.setEnabled(False)
            if not is_image:
                for b in [self.stop_button,self.pause_button,self.capture_button,self.record_button,self.clear_roi_button]: b.setEnabled(True)
                
            # หยุดการทำงานของ worker เดิม (ถ้ามี)
            if self.worker:
                try:
                    self.worker.stop(); self.video_thread.quit(); self.video_thread.wait(3000); self.worker.deleteLater()
                except Exception as e:
                    print(f"Error stopping previous worker: {e}")
                    
            # สร้าง worker ใหม่
            try:
                self.worker=Worker(model_path, source, threshold, is_image, autosave, target_fps, save_orig, beep,
                               self.roi, self.roi_poly, desired_res, is_webcam, lowlat, codec_choice)
                self.worker.moveToThread(self.video_thread)
                self.video_thread.started.connect(self.worker.run)
                self.worker.image_update.connect(self.update_image)
                self.worker.status_update.connect(self.update_status)
                self.worker.finished.connect(self.detection_finished)
                self.status_label.setText("กำลังเริ่มการตรวจจับ...")
                self.video_thread.start()
            except Exception as e:
                self.start_button.setEnabled(True)
                for b in [self.stop_button,self.pause_button,self.capture_button,self.record_button]: b.setEnabled(False)
                QMessageBox.critical(self,"Error",f"เกิดข้อผิดพลาดในการเริ่มการตรวจจับ: {str(e)}")
                print(f"Error starting detection: {e}")
                
        except Exception as e:
            self.start_button.setEnabled(True)
            QMessageBox.critical(self,"Error",f"เกิดข้อผิดพลาดที่ไม่คาดคิด: {str(e)}")
            print(f"Unexpected error in start_detection: {e}")
    def stop_detection(self):
        print(f"Entering MainWindow.stop_detection")
        self.status_label.setText("Stopping...")
        if self.worker and self.video_thread.isRunning():
            self.worker.stop()
            self.video_thread.quit()
            # Wait for the thread to finish. Add a timeout to prevent indefinite blocking.
            if not self.video_thread.wait(5000): # 5-second timeout
                print("Warning: Worker thread did not terminate gracefully. Forcing termination.")
                self.video_thread.terminate()
                self.video_thread.wait() # Wait again after termination
    def detection_finished(self):
        print(f"Entering MainWindow.detection_finished")
        self.write_summary_file()
        if hasattr(self.worker,'is_image_source') and not self.worker.is_image_source:
            self.image_display_label.setText("Processing finished. Press 'Start' to begin again.")
        for b in [self.stop_button,self.pause_button,self.capture_button,self.record_button]: b.setEnabled(False)
        self.pause_button.setChecked(False); self.pause_button.setText("Pause")
        self.record_button.setChecked(False); self.record_button.setText("Start Recording")
        self.clear_roi_button.setEnabled(False); self.start_button.setEnabled(True); self.worker=None
    def update_image(self, pixmap):
        #print(f"Entering MainWindow.update_image")
        self.current_pixmap=pixmap
        self.image_display_label.setPixmap(pixmap.scaled(self.image_display_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
    def update_status(self, msg:str):
        #print(f"Entering MainWindow.update_status")
        self.status_label.setText(msg)
    def set_roi(self, rect:QRect):
        print(f"Entering MainWindow.set_roi")
        pm=self.image_display_label.pixmap()
        if not pm or pm.isNull(): self.status_label.setText("Cannot set ROI: No image displayed."); return
        L=self.image_display_label.size(); S=pm.size().scaled(L, Qt.AspectRatioMode.KeepAspectRatio)
        ox=(L.width()-S.width())/2; oy=(L.height()-S.height())/2
        if S.width()==0 or S.height()==0: self.status_label.setText("Cannot set ROI: Invalid image size."); return
        if self.worker and self.worker.latest_frame is not None:
            h0,w0=self.worker.latest_frame.shape[0], self.worker.latest_frame.shape[1]
        else:
            h0,w0=1080,1920
        sx=w0/S.width(); sy=h0/S.height()
        x1=int((rect.left()-ox)*sx); y1=int((rect.top()-oy)*sy); x2=int((rect.right()-ox)*sx); y2=int((rect.bottom()-oy)*sy)
        x1=max(0,x1); y1=max(0,y1); x2=min(w0,x2); y2=min(h0,y2)
        self.roi=(x1,y1,x2-x1,y2-y1)
        self.roi_poly=None
        if self.worker:
            self.worker.update_roi(self.roi)
            try:
                self.worker.update_roi_poly(None)
            except Exception:
                pass
        self.status_label.setText(f"ROI (rect) selected: x={self.roi[0]}, y={self.roi[1]}, w={self.roi[2]}, h={self.roi[3]}")
        self.save_config_with_roi(); self.clear_roi_button.setEnabled(True)
    def set_roi_polygon(self, points:list):
        print(f"Entering MainWindow.set_roi_polygon")
        pm=self.image_display_label.pixmap()
        if not pm or pm.isNull(): self.status_label.setText("Cannot set ROI: No image displayed."); return
        L=self.image_display_label.size(); S=pm.size().scaled(L, Qt.AspectRatioMode.KeepAspectRatio)
        ox=(L.width()-S.width())/2; oy=(L.height()-S.height())/2
        if S.width()==0 or S.height()==0: self.status_label.setText("Cannot set ROI: Invalid image size."); return
        if self.worker and self.worker.latest_frame is not None:
            h0,w0=self.worker.latest_frame.shape[0], self.worker.latest_frame.shape[1]
        else:
            h0,w0=1080,1920
        sx=w0/S.width(); sy=h0/S.height()
        poly=[]
        for pt in points:
            x=int((pt.x()-ox)*sx); y=int((pt.y()-oy)*sy)
            x=max(0,min(w0-1,x)); y=max(0,min(h0-1,y))
            poly.append([x,y])
        if len(poly) < 3:
            self.status_label.setText("ROI polygon requires at least 3 points."); return
        self.roi_poly = poly
        self.roi = None
        if self.worker:
            self.worker.update_roi(None)
            try:
                self.worker.update_roi_poly(self.roi_poly)
            except Exception:
                pass
        self.status_label.setText(f"ROI (poly) selected: {len(self.roi_poly)} points")
        self.save_config_with_roi(); self.clear_roi_button.setEnabled(True)
    def clear_roi(self):
        print(f"Entering MainWindow.clear_roi")
        self.roi=None; self.roi_poly=None; self.image_display_label.clear_roi()
        if self.worker:
            self.worker.update_roi(None)
            try:
                self.worker.update_roi_poly(None)
            except Exception:
                pass
        self.status_label.setText("ROI cleared. Detection will cover the full frame.")
        self.save_config_with_roi(); self.clear_roi_button.setEnabled(False)
    def capture_frame(self):
        print(f"Entering MainWindow.capture_frame")
        if self.worker is None or self.worker.latest_frame is None: 
            self.status_label.setText("No active stream to capture."); return
        out='outputs'; os.makedirs(out, exist_ok=True); ts=int(time.time()); path=os.path.join(out, f"capture_{ts}.png")
        # Save original frame without labels
        with self.worker.lock:
            original_frame = self.worker.latest_frame.copy()
        if cv2.imwrite(path, original_frame): self.status_label.setText(f"Frame captured and saved to {path}")
        else: self.status_label.setText(f"Failed to save frame to {path}")
    def write_summary_file(self):
        #print(f"Entering MainWindow.write_summary_file")
        if not self.worker or not self.worker.detection_summary:
            self.status_label.setText("Finished. No objects were detected."); return
        summary=self.worker.detection_summary; out='outputs'; os.makedirs(out, exist_ok=True)
        ts=datetime.now().strftime('%Y%m%d_%H%M%S'); path=os.path.join(out, f"Result_{ts}.txt")
        total=sum(summary.values())
        with open(path,'w',encoding='utf-8') as f:
            f.write(f"Detection Summary - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"); f.write("="*40+"\n")
            for name,cnt in sorted(summary.items()): f.write(f"- {name}: {cnt}\n")
            f.write("="*40+"\n"); f.write(f"Total Objects Detected: {total}\n")
        self.status_label.setText(f"Finished. Summary saved to {path}")
    def toggle_recording(self, checked:bool):
        print(f"Entering MainWindow.toggle_recording")
        if self.worker: self.worker.set_recording_state(checked); self.record_button.setText("Stop Recording" if checked else "Start Recording")
    def toggle_pause(self, checked:bool):
        print(f"Entering MainWindow.toggle_pause")
        if self.worker:
            self.worker.set_pause_state(checked)
            if checked: self.pause_button.setText("Resume"); self.status_label.setText("Paused.")
            else: self.pause_button.setText("Pause"); self.status_label.setText("Processing...")
    def save_config_with_roi(self):
        print(f"Entering MainWindow.save_config_with_roi")
        self.config['model_path']=self.model_path_input.text(); self.config['source_path']=self.source_path_input.text()
        self.config['threshold']=self.thresh_slider.value()/100.0; self.config['autosave']=self.autosave_checkbox.isChecked()
        self.config['fps_index']=self.fps_combo.currentIndex(); self.config['save_original']=self.save_original_checkbox.isChecked()
        self.config['beep']=self.beep_checkbox.isChecked(); self.config['roi']=self.roi; self.config['roi_poly']=self.roi_poly
        self.config['resolution_index']=self.resolution_combo.currentIndex(); self.config['low_latency']=self.low_latency_checkbox.isChecked()
        self.config['codec']=self.codec_combo.currentText(); save_config(self.config)
    def closeEvent(self, e):
        print(f"Entering MainWindow.closeEvent")
        self.save_config_with_roi(); self.stop_detection(); super().closeEvent(e)

if __name__ == "__main__":
    print(f"Entering main execution block")
    print('is_available =', torch.cuda.is_available())
    print('cuda ver     =', torch.version.cuda)
    if torch.cuda.is_available():
        print('device name  =', torch.cuda.get_device_name(0))
    app=QApplication(sys.argv); w=MainWindow(); w.show(); sys.exit(app.exec())



     
 
