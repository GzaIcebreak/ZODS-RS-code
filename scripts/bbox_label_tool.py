import argparse
import json
import os
from pathlib import Path

import cv2

def draw_bboxes(image_path, category_id, category_name):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    clone = img.copy()
    h, w = img.shape[:2]
    bboxes = []  # [x,y,w,h]
    start_pt = None
    preview = None

    win_name = "BBox Label (L-Drag draw; Backspace delete; s save; q quit)"
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)

    def on_mouse(event, x, y, flags, param):
        nonlocal start_pt, preview, img
        if event == cv2.EVENT_LBUTTONDOWN:
            start_pt = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and start_pt is not None:
            img[:] = clone
            # draw existing bboxes
            for (bx, by, bw, bh) in bboxes:
                cv2.rectangle(img, (int(bx), int(by)), (int(bx + bw), int(by + bh)), (0, 255, 0), 2)
            # draw preview
            x0, y0 = start_pt
            x1, y1 = x, y
            x_min, y_min = min(x0, x1), min(y0, y1)
            x_max, y_max = max(x0, x1), max(y0, y1)
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 255), 1)
        elif event == cv2.EVENT_LBUTTONUP and start_pt is not None:
            x0, y0 = start_pt
            x1, y1 = x, y
            x_min, y_min = min(x0, x1), min(y0, y1)
            x_max, y_max = max(x0, x1), max(y0, y1)
            bw, bh = x_max - x_min, y_max - y_min
            if bw > 0 and bh > 0:
                bboxes.append([float(x_min), float(y_min), float(bw), float(bh)])
            start_pt = None
            # redraw all
            img[:] = clone
            for (bx, by, bw, bh) in bboxes:
                cv2.rectangle(img, (int(bx), int(by)), (int(bx + bw), int(by + bh)), (0, 255, 0), 2)

    cv2.setMouseCallback(win_name, on_mouse)

    while True:
        cv2.imshow(win_name, img)
        key = cv2.waitKey(20) & 0xFF
        if key == ord('q'):
            bboxes = []
            break
        elif key == ord('s'):
            break
        elif key == 8 or key == 127:  # Backspace / DEL
            if bboxes:
                bboxes.pop()
                img[:] = clone
                for (bx, by, bw, bh) in bboxes:
                    cv2.rectangle(img, (int(bx), int(by)), (int(bx + bw), int(by + bh)), (0, 255, 0), 2)

    cv2.destroyAllWindows()
    return bboxes, (w, h)

def save_coco(out_json, image_path, category_id, category_name, bboxes, size_wh):
    w, h = size_wh
    images = [{
        "id": 0,
        "file_name": Path(image_path).name,
        "width": int(w),
        "height": int(h)
    }]
    annotations = []
    for i, (x, y, bw, bh) in enumerate(bboxes, start=1):
        annotations.append({
            "id": i,
            "image_id": 0,
            "category_id": int(category_id),
            "bbox": [x, y, bw, bh],
            "iscrowd": 0
        })
    categories = [{
        "id": int(category_id),
        "name": str(category_name)
    }]
    os.makedirs(Path(out_json).parent, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "images": images,
            "annotations": annotations,
            "categories": categories
        }, f, ensure_ascii=False)
    print(f"Saved: {out_json}")

def main():
    parser = argparse.ArgumentParser("Simple BBox Label Tool -> COCO JSON (Win friendly)")
    parser.add_argument("--image", type=str, required=True, help="path to image")
    parser.add_argument("--out", type=str, required=True, help="output COCO json")
    parser.add_argument("--category_id", type=int, default=1)
    parser.add_argument("--category_name", type=str, default="glass")
    args = parser.parse_args()

    bboxes, size_wh = draw_bboxes(args.image, args.category_id, args.category_name)
    if not bboxes:
        print("No bbox saved. Exit.")
        return
    save_coco(args.out, args.image, args.category_id, args.category_name, bboxes, size_wh)

if __name__ == "__main__":
    main()