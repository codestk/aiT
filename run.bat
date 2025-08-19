@echo off
chcp 65001 > nul
title Automatic Full YOLO Workflow

echo =================================================================
echo   Starting Automatic Full YOLO Workflow...
echo =================================================================
echo.

echo [ขั้นตอนที่ 1/4] กำลังลบโฟลเดอร์เก่า (custom_data, data, runs)...

if exist "data" ( rd /s /q "data" )
if exist "runs" ( rd /s /q "runs" )
echo Cleanup complete.
echo.
timeout /t 2 > nul

echo [ขั้นตอนที่ 2/4] กำลังเตรียมข้อมูล (แบ่งข้อมูล และสร้างไฟล์ data.yaml)...
python train_val_split.py --datapath="D:\aiT\custom_data" --train_pct=.9
python.exe dataYaml.py
echo Data preparation complete.
echo.
timeout /t 2 > nul

echo [ขั้นตอนที่ 3/4] กำลังเริ่มการฝึกสอนโมเดล YOLOv8m...
yolo detect train data=data.yaml model=yolo11x.pt epochs=60 imgsz=640
echo Model training complete.
echo
timeout /t 2 > nul

echo [ขั้นตอนที่ 4/4] กำลังค้นหาโมเดลล่าสุดและเริ่มการตรวจจับ...
set "LATEST_RUN="
for /d %%i in (runs\detect\train*) do set "LATEST_RUN=%%i"

if not defined LATEST_RUN (
    echo.
    echo [เกิดข้อผิดพลาด] ไม่พบโฟลเดอร์ผลการฝึกสอน (train)
    echo.
    pause
    exit /b
)

set "MODEL_PATH=%LATEST_RUN%\weights\best.pt"
echo พบโมเดลล่าสุดที่: %MODEL_PATH%
echo.
echo Starting detection...
rem python yolo_detect.py --model="%MODEL_PATH%" --source=usb0 --imgsz=640 --conf=0.25 --save-txt --save-conf --project="runs/detect" --name="latest_detection"
python yolo_detect.py

echo.

echo --- Full Workflow Finished ---
echo.
pause
