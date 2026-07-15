import argparse
import json
import os
from typing import List, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def parse_args():
    parser = argparse.ArgumentParser(description="Generate segmentation masks from bounding boxes using SAM-2.1 (Hiera)")
    parser.add_argument("--input_json", type=str, required=True, help="Path to input COCO json file")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to directory containing images")
    parser.add_argument("--sam2_cfg_file", type=str, required=True, help="SAM-2.1 config yaml (e.g. sam2.1_hiera_b+.yaml)")
    parser.add_argument("--sam2_ckpt_path", type=str, required=True, help="Path to SAM-2.1 checkpoint .pt")
    parser.add_argument("--device", type=str, default="cuda", help="cuda|cpu")
    parser.add_argument("--visualize", action="store_true", help="Visualize and save results")
    parser.add_argument("--multimask_output", action="store_true", help="Generate multiple masks per bbox and select best")
    parser.add_argument("--mask_threshold", type=float, default=0.0, help="Threshold for binarizing mask logits")
    return parser.parse_args()


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    mask_image = mask.reshape(mask.shape[-2:] + (1,)) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box_xywh(box, ax):
    x0, y0, w, h = box
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def mask_to_polygon(ann, mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segmentation = []
    for contour in contours:
        contour = contour.flatten().tolist()
        if len(contour) > 4:
            segmentation.append(contour)
    if not segmentation:
        # fallback to bbox rectangle polygon
        x, y, w, h = ann['bbox']
        segmentation = [[x, y, x+w, y, x+w, y+h, x, y+h]]
    return segmentation


def bbox_xywh_to_xyxy(bbox_xywh: List[float]) -> List[float]:
    """Convert bbox from [x, y, w, h] to [x1, y1, x2, y2] format."""
    x, y, w, h = bbox_xywh
    return [x, y, x + w, y + h]


def visualize(image: np.ndarray, masks: List[np.ndarray], boxes: List[List[float]], save_path: str, polygons=None):
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    ax.imshow(image)
    for m in masks:
        show_mask(m, ax, random_color=True)
    for b in boxes:
        show_box_xywh(b, ax)
    if polygons is not None:
        for polygon_group in polygons:
            for polygon in polygon_group:
                pts = np.array(polygon).reshape(-1, 2)
                ax.plot(pts[:, 0], pts[:, 1], '-r', linewidth=2)
    ax.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')

    print("[SAM2.1] Building model ...")
    model = build_sam2(args.sam2_cfg_file, args.sam2_ckpt_path)
    model.to(device)
    model.eval()
    predictor = SAM2ImagePredictor(model)

    with open(args.input_json, 'r') as f:
        coco = json.load(f)

    # index annotations by image id
    image_to_anns = {}
    for ann in coco['annotations']:
        image_to_anns.setdefault(ann['image_id'], []).append(ann)

    # create out dir
    if args.visualize:
        vis_dir = os.path.splitext(args.input_json)[0] + '_sam21_segm'
        os.makedirs(vis_dir, exist_ok=True)

    print("[SAM2.1] Processing images ...")
    for img_info in tqdm(coco['images']):
        file_name = img_info['file_name']
        img_path = os.path.join(args.image_dir, file_name)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise ValueError(f"Cannot read image: {img_path}")
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        anns = image_to_anns.get(img_info['id'], [])
        boxes = [ann['bbox'] for ann in anns]
        if len(boxes) == 0:
            continue

        # Set image for predictor (only once per image)
        predictor.set_image(img)

        # Generate masks directly from bboxes
        selected_masks = []
        polygons_all = []
        for ann in anns:
            bbox_xywh = ann['bbox']
            x, y, w, h = bbox_xywh
            
            # Validate bbox
            if w <= 0 or h <= 0:
                print(f"   ⚠️  Warning: Invalid bbox {bbox_xywh} (width or height <= 0), skipping")
                # Use bbox rectangle as fallback
                best_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
                y1, y2 = max(0, int(y)), min(img.shape[0], int(y + h))
                x1, x2 = max(0, int(x)), min(img.shape[1], int(x + w))
                if y2 > y1 and x2 > x1:
                    best_mask[y1:y2, x1:x2] = 1
                seg = mask_to_polygon(ann, best_mask)
                ann['segmentation'] = seg
                ann['area'] = float(best_mask.sum())
                selected_masks.append(best_mask)
                polygons_all.append(seg)
                continue
            
            # Convert bbox to xyxy format for SAM2
            bbox_xyxy = bbox_xywh_to_xyxy(bbox_xywh)
            input_box = np.array(bbox_xyxy, dtype=np.float32)
            
            try:
                # Predict mask using bbox as prompt
                # box format: [x1, y1, x2, y2] (single box, not batched)
                masks, iou_predictions, low_res_masks = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box,
                    multimask_output=args.multimask_output,
                )
                
                # masks shape: (C, H, W) where C is number of masks
                # iou_predictions shape: (C,) - quality scores for each mask
                if args.multimask_output and masks.shape[0] > 1:
                    # Select best mask (highest IoU score)
                    best_idx = np.argmax(iou_predictions)
                    best_mask = masks[best_idx]
                else:
                    best_mask = masks[0]
                
                # Binarize mask using threshold
                best_mask = (best_mask > args.mask_threshold).astype(np.uint8)
                
                # Validate mask is not empty
                if best_mask.sum() == 0:
                    print(f"   ⚠️  Warning: Generated empty mask for bbox {bbox_xywh}, using bbox rectangle")
                    # Fallback to bbox rectangle
                    best_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
                    y1, y2 = max(0, int(y)), min(img.shape[0], int(y + h))
                    x1, x2 = max(0, int(x)), min(img.shape[1], int(x + w))
                    if y2 > y1 and x2 > x1:
                        best_mask[y1:y2, x1:x2] = 1
                
            except Exception as e:
                print(f"   ⚠️  Warning: Failed to generate mask for bbox {bbox_xywh}: {e}")
                # Fallback to bbox rectangle
                best_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
                y1, y2 = max(0, int(y)), min(img.shape[0], int(y + h))
                x1, x2 = max(0, int(x)), min(img.shape[1], int(x + w))
                if y2 > y1 and x2 > x1:
                    best_mask[y1:y2, x1:x2] = 1
            
            # Convert mask to polygon segmentation
            seg = mask_to_polygon(ann, best_mask)
            ann['segmentation'] = seg
            ann['area'] = float(best_mask.sum())
            selected_masks.append(best_mask)
            polygons_all.append(seg)
        
        # Reset predictor for next image
        predictor.reset_predictor()

        if args.visualize:
            save_path = os.path.join(vis_dir, f"{os.path.splitext(file_name)[0]}_sam21.png")
            visualize(img, selected_masks, boxes, save_path, polygons=polygons_all)

    out_json = os.path.splitext(args.input_json)[0] + '_with_segm_sam21.json'
    with open(out_json, 'w') as f:
        json.dump(coco, f)
    print(f"[SAM2.1] Done. Saved annotations to: {out_json}")


if __name__ == "__main__":
    main()


