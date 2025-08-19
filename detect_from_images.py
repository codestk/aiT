import os
import sys
import argparse
import glob
import time
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP'}

def list_images(path: str):
    p = Path(path)
    if p.is_file():
        if p.suffix in IMG_EXTS:
            return [str(p)]
        else:
            print(f"ไฟล์ไม่ใช่รูปที่รองรับ: {p}")
            return []
    elif p.is_dir():
        files = []
        for ext in IMG_EXTS:
            files.extend([str(x) for x in p.rglob(f"*{ext}")])
        return sorted(files)
    else:
        print(f"ไม่พบพาธ: {path}")
        return []

def parse_resolution(s):
    if s is None:
        return None
    try:
        w, h = s.lower().split('x')
        return int(w), int(h)
    except Exception:
        raise ValueError('รูปแบบ --resolution ต้องเป็น WxH เช่น 1280x720')

def draw_label(img, text, x, y, color=(0, 255, 255)):
    (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    y = max(y, th + 6)
    cv2.rectangle(img, (x, y - th - 6), (x + tw + 2, y + bl - 6), color, cv2.FILLED)
    cv2.putText(img, text, (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, help='พาธโมเดล YOLO (.pt), เช่น best.pt')
    ap.add_argument('--source', required=True, help='ไฟล์รูป หรือ โฟลเดอร์รูป')
    ap.add_argument('--thresh', type=float, default=0.5, help='confidence ขั้นต่ำ (ดีฟอลต์ 0.5)')
    ap.add_argument('--resolution', default=None, help='ปรับขนาดก่อน infer เช่น 1280x720 (ตัวเลือก)')
    ap.add_argument('--outdir', default='outputs', help='โฟลเดอร์บันทึกผล (ภาพ+json)')
    args = ap.parse_args()

    # ตรวจโมเดล
    if not os.path.exists(args.model):
        print('ERROR: ไม่พบไฟล์โมเดลที่ระบุ')
        sys.exit(1)

    # เตรียมรายการภาพ
    images = list_images(args.source)
    if not images:
        print('ERROR: ไม่พบรูปสำหรับประมวลผล')
        sys.exit(1)

    # โหลดโมเดล
    model = YOLO(args.model, task='detect')
    labels = model.names
    resize_to = parse_resolution(args.resolution)

    # เตรียมโฟลเดอร์บันทึกผล
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # สีกรอบ (Tableau-ish)
    bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106),
                   (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

    all_fps = []
    for idx, img_path in enumerate(images, 1):
        t0 = time.perf_counter()

        frame = cv2.imread(img_path)
        if frame is None:
            print(f"[{idx}/{len(images)}] ข้าม (อ่านรูปไม่ได้): {img_path}")
            continue

        if resize_to:
            frame = cv2.resize(frame, resize_to)

        # รันโมเดล
        results = model(frame, verbose=False)
        det = results[0].boxes

        # สรุปผลสำหรับ JSON
        detections_json = []
        obj_count = 0

        for i in range(len(det)):
            conf = float(det[i].conf.item())
            if conf < args.thresh:
                continue

            cls_id = int(det[i].cls.item())
            cls_name = labels.get(cls_id, str(cls_id))

            xyxy = det[i].xyxy.cpu().numpy().squeeze().astype(int)
            xmin, ymin, xmax, ymax = map(int, xyxy.tolist())

            color = bbox_colors[cls_id % len(bbox_colors)]
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            draw_label(frame, f"{cls_name}: {conf*100:.0f}%", xmin, ymin, color)

            obj_count += 1
            detections_json.append({
                "class_id": cls_id,
                "class_name": cls_name,
                "confidence": conf,
                "bbox_xyxy": [int(xmin), int(ymin), int(xmax), int(ymax)]
            })

        t1 = time.perf_counter()
        fps = 1.0 / max(1e-9, (t1 - t0))
        all_fps.append(fps)

        # วาดสรุปบนภาพ
        cv2.putText(frame, f'Objects: {obj_count}  FPS: {fps:.2f}',
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2, cv2.LINE_AA)

        # บันทึกผล
        stem = Path(img_path).stem
        out_img = outdir / f"{stem}_det.jpg"
        out_json = outdir / f"{stem}_det.json"
        cv2.imwrite(str(out_img), frame)
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump({
                "source": img_path,
                "num_detections": obj_count,
                "threshold": args.thresh,
                "detections": detections_json
            }, f, ensure_ascii=False, indent=2)

        print(f"[{idx}/{len(images)}] ✓ {img_path} → {out_img.name}, {out_json.name}  (FPS {fps:.2f})")

    if all_fps:
        print(f"\nAverage FPS over {len(all_fps)} images: {np.mean(all_fps):.2f}")

if __name__ == "__main__":
    main()
