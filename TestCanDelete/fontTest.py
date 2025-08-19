import cv2
import numpy as np

# --- สร้างภาพพื้นหลังสีขาว ---
# สร้างภาพขนาด 800x600 pixels, 3 channels (BGR), และเติมด้วยสีขาว (255)
width, height = 800, 600
image = np.full((height, width, 3), 255, dtype=np.uint8)

# --- รายชื่อฟอนต์ทั้งหมดใน OpenCV ---
# สร้าง Dictionary เพื่อเก็บค่าคงที่ของฟอนต์และชื่อของมัน
font_map = {
    "FONT_HERSHEY_SIMPLEX": cv2.FONT_HERSHEY_SIMPLEX,
    "FONT_HERSHEY_PLAIN": cv2.FONT_HERSHEY_PLAIN,
    "FONT_HERSHEY_DUPLEX": cv2.FONT_HERSHEY_DUPLEX,
    "FONT_HERSHEY_COMPLEX": cv2.FONT_HERSHEY_COMPLEX,
    "FONT_HERSHEY_TRIPLEX": cv2.FONT_HERSHEY_TRIPLEX,
    "FONT_HERSHEY_COMPLEX_SMALL": cv2.FONT_HERSHEY_COMPLEX_SMALL,
    "FONT_HERSHEY_SCRIPT_SIMPLEX": cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
    "FONT_HERSHEY_SCRIPT_COMPLEX": cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
}

# --- ตั้งค่าเริ่มต้นสำหรับการวาด ---
start_x = 50
start_y = 50
line_height = 50
font_scale = 1.0
font_thickness = 2
font_color = (0, 0, 0) # สีดำ

# --- วาดข้อความด้วยฟอนต์แต่ละแบบลงบนภาพ ---
# วนลูปผ่านฟอนต์แต่ละตัวใน Dictionary
for font_name, font_constant in font_map.items():
    # วาดชื่อฟอนต์ลงบนภาพ
    cv2.putText(image, 
                f"{font_name}", 
                (start_x, start_y), 
                font_constant, 
                font_scale, 
                font_color, 
                font_thickness)
    
    # เลื่อนตำแหน่ง y ลงมาสำหรับบรรทัดถัดไป
    start_y += line_height

# --- เพิ่มตัวอย่างฟอนต์ตัวเอียง (Italic) ---
start_y += line_height # เว้นบรรทัด
italic_font = cv2.FONT_HERSHEY_SIMPLEX | cv2.FONT_ITALIC
cv2.putText(image, 
            "FONT_HERSHEY_SIMPLEX (Italic)", 
            (start_x, start_y), 
            italic_font, 
            font_scale, 
            font_color, 
            font_thickness)

# --- บันทึกและแสดงผลภาพ ---
output_filename = "font_examples.png"
cv2.imwrite(output_filename, image)

print(f"รูปภาพตัวอย่างฟอนต์ถูกบันทึกเป็นไฟล์ '{output_filename}' เรียบร้อยแล้ว")

# แสดงผลภาพในหน้าต่าง
cv2.imshow("OpenCV Font Examples", image)

# รอการกดปุ่มใดๆ เพื่อปิดหน้าต่าง
cv2.waitKey(0)
cv2.destroyAllWindows()
