import json
import numpy as np
from pycocotools import mask as mask_utils
from collections import defaultdict
import torch
from PIL import Image
from tqdm import tqdm
import argparse

from zods_rs.dataset.metainfo import METAINFO

class COCOInstToSegmEvaluator:
    def __init__(self, gt_json_path, pred_json_path, confidence_threshold=0.5, class_split=None):
        """Initialize the evaluator with ground truth and prediction paths"""
        self.confidence_threshold = confidence_threshold
        with open(gt_json_path, 'r') as f:
            self.gt_data = json.load(f)
        with open(pred_json_path, 'r') as f:
            self.pred_data = json.load(f)
        
        # Get few-shot class names and ids
        self.cat_names = METAINFO[class_split]
        self.cat_ids = [cat['id'] for cat in self.gt_data['categories'] 
                       if cat['name'] in self.cat_names]
        
        # Create category id to index mapping only for relevant categories
        self.cat_id_to_idx = {cat['id']: idx 
                             for idx, cat in enumerate(self.gt_data['categories'])
                             if cat['name'] in self.cat_names}
        
        # Filter predictions to only include relevant categories
        self.pred_data = [pred for pred in self.pred_data 
                         if pred['category_id'] in self.cat_ids]
        
        # Filter gt_data to only include relevant categories
        self.gt_data['annotations'] = [ann for ann in self.gt_data['annotations'] 
                                      if ann['category_id'] in self.cat_ids]

        # Pre-compute all semantic masks
        self.pred_semantic_masks = {}
        self.gt_semantic_masks = {}
        
        # Create image id to size mapping
        self.image_sizes = {img['id']: (img['height'], img['width']) 
                           for img in self.gt_data['images']}
        
        # Pre-compute all prediction masks
        print("Converting instance predictions to semantic masks...")
        for img_id, (height, width) in tqdm(self.image_sizes.items()):
            self.pred_semantic_masks[img_id] = self._convert_pred_to_semantic(
                img_id, height, width)
            
        # Pre-compute all ground truth masks
        print("Converting ground truth instances to semantic masks...")
        for img_id, (height, width) in tqdm(self.image_sizes.items()):
            self.gt_semantic_masks[img_id] = self._convert_gt_to_semantic(
                img_id, height, width)

    def _convert_pred_to_semantic(self, img_id, height, width):
        """Helper method to convert predictions for one image"""
        semantic_mask = np.zeros((height, width), dtype=np.uint8)
        
        img_preds = [p for p in self.pred_data 
                    if p['image_id'] == img_id 
                    and p['score'] >= self.confidence_threshold]
        
        # 
        img_preds.sort(key=lambda x: x['score'], reverse=True)
        
        for pred in img_preds:
            binary_mask = mask_utils.decode(pred['segmentation'])
            category_idx = self.cat_id_to_idx[pred['category_id']]
            semantic_mask[binary_mask > 0] = category_idx
            
        return semantic_mask

    def _convert_gt_to_semantic(self, img_id, height, width):
        """Helper method to convert ground truth for one image"""
        semantic_mask = np.zeros((height, width), dtype=np.uint8)
        
        img_anns = [ann for ann in self.gt_data['annotations'] 
                   if ann['image_id'] == img_id and ann['iscrowd'] == 0]
        
        for ann in img_anns:
            if isinstance(ann['segmentation'], dict):
                binary_mask = mask_utils.decode(ann['segmentation'])
            else:
                rles = mask_utils.frPyObjects(ann['segmentation'], height, width)
                rle = mask_utils.merge(rles)
                binary_mask = mask_utils.decode(rle)
            
            category_idx = self.cat_id_to_idx[ann['category_id']]
            semantic_mask[binary_mask > 0] = category_idx
            
        return semantic_mask

    def evaluate(self):
        """Evaluate semantic segmentation results"""
        total_inter = defaultdict(int)
        total_union = defaultdict(int)
        
        print("Computing IoU metrics...")
        for img_id in tqdm(self.image_sizes.keys()):
            pred_mask = torch.from_numpy(self.pred_semantic_masks[img_id])
            gt_mask = torch.from_numpy(self.gt_semantic_masks[img_id])
            
            for class_idx in range(len(self.cat_id_to_idx)):
                pred_binary = (pred_mask == class_idx)
                gt_binary = (gt_mask == class_idx)
                
                intersection = (pred_binary & gt_binary).sum().item()
                union = (pred_binary | gt_binary).sum().item()
                
                total_inter[class_idx] += intersection
                total_union[class_idx] += union
        
        # Compute IoU for each class
        ious = []
        for class_idx in range(len(self.cat_id_to_idx)):
            if total_union[class_idx] == 0:
                continue
            iou = total_inter[class_idx] / total_union[class_idx]
            ious.append(iou)
        
        # Compute mean IoU
        miou = np.mean(ious)
        return miou, ious

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate COCO instance to semantic segmentation')
    parser.add_argument('--pred_json', type=str, required=True,
                        help='Path to prediction JSON file')
    parser.add_argument('--class_split', type=str, required=True,
                        help='Class split name as defined in METAINFO')
    args = parser.parse_args()
    
    print(f"\nEvaluating \033[31m{args.pred_json}\033[0m with class split \033[31m{args.class_split}\033[0m")

    evaluator = COCOInstToSegmEvaluator(
        gt_json_path='inst_to_segm/original/instances_val2017.json',
        pred_json_path=args.pred_json, # inst_to_segm/coco_inst_semantic_split_1_1shot_33seed_results.json
        class_split=args.class_split # coco_semantic_split_1
    )
    miou, class_ious = evaluator.evaluate()
    print(f"Mean IoU: {miou:.4f}")
