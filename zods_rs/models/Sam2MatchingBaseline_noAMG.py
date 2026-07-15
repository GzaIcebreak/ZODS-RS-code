import copy
import json
import os
import random
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import warnings

import ot
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from scipy.optimize import linear_sum_assignment
import pycocotools.mask as mask_utils

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torchvision.ops.boxes import batched_nms
from torchvision.transforms import Normalize
import matplotlib.pyplot as plt
try:  # pragma: no cover - optional
    import seaborn as sns
except Exception:  # pragma: no cover
    sns = None

from sklearn.decomposition import PCA as SKPCA

try:  # pragma: no cover - optional dependency
    import umap  # type: ignore
except Exception:  # pragma: no cover
    umap = None

from sam2.build_sam import build_sam2
from sam2.build_sam import build_sam2_video_predictor
from sam2.modeling.sam2_utils import MLP
from sam2.utils.misc import fill_holes_in_mask_scores, concat_points
from sam2.utils.amg import batched_mask_to_box

from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

from modules.uam_uncert_merge import bayes_merge, crf_refine, pixelwise_distribution
from modules.pp_prototype_purify import class_prototypes, class_prototypes_robust, class_prototypes_ot
from utils.prior_maps import margin_prior, norm_prior, combine_priors
from zods_rs.engine import build_merger, build_matcher

from zods_rs.models.encoder_factory import build_encoder

from zods_rs.models.matching_baseline_utils import kmeans, kmeans_decouple
from zods_rs.models.model_utils import concat_all_gather
from zods_rs.utils import print_dict
from zods_rs.models.matching_baseline_utils import vis_pca, vis_kmeans, fast_l2

import time

PRINT_TIMING = False

INSTANCE_COLORS = np.array(
    [
        [255, 99, 71],
        [72, 209, 204],
        [255, 215, 0],
        [60, 179, 113],
        [147, 112, 219],
        [255, 140, 0],
        [64, 224, 208],
        [238, 130, 238],
    ],
    dtype=np.uint8,
)


def _ensure_dir(path: Path) -> None:
    """Create directory for given path if it does not already exist."""

    path.parent.mkdir(parents=True, exist_ok=True)

encoder_predefined_cfgs = {
    "dinov2_large": dict(
        model_size="vit_large",
        img_size=518,
        patch_size=14,
        init_values=1e-5,
        ffn_layer='mlp',
        block_chunks=0,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        feat_dim=1024,
    ),
    "dinov3_large": dict(
        hf_model_name="facebook/dinov3-large"
    ),
}



class Sam2MatchingBaselineNoAMG(nn.Module):
    def __init__(
        self,
        sam2_cfg_file,
        sam2_ckpt_path,
        sam2_infer_cfgs,
        encoder_cfg,
        encoder_ckpt_path,
        memory_bank_cfg,
        dataset_name='coco',
        dataset_imgs_path=None,
        class_names=None,
        online_vis=False,
        vis_thr=0.5,
        uam=None,
        eval=None,
        sem=None,
    ):
        super(Sam2MatchingBaselineNoAMG, self).__init__()

        self.dataset_name = dataset_name
        self.class_names = class_names
        self.dataset_imgs_path = dataset_imgs_path
        self.online_vis = online_vis
        self.vis_thr = vis_thr
        self.points_per_side = sam2_infer_cfgs.get("points_per_side")
        self.testing_point_bs = sam2_infer_cfgs.get("testing_point_bs")
        self.iou_thr = sam2_infer_cfgs.get("iou_thr")
        self.num_out_instance = sam2_infer_cfgs.get("num_out_instance")
        self.nms_thr = sam2_infer_cfgs.get("nms_thr")
        self.kmeans_k = sam2_infer_cfgs.get("kmeans_k")
        self.n_pca_components = sam2_infer_cfgs.get("n_pca_components")
        self.cls_num_per_mask = sam2_infer_cfgs.get("cls_num_per_mask")

        self.with_negative_refs = sam2_infer_cfgs.get("with_negative_refs", False)

        # UAM configuration (training-free uncertainty-aware merging)
        self.uam_cfg = self._build_uam_cfg(uam or {})
        self.uam_enabled = self.uam_cfg.get("enable", False)
        self.uam_merger = build_merger(self.uam_cfg) if self.uam_enabled else None
        self.eval_out_format = (
            (eval or {}).get("out_format") if isinstance(eval, dict) else None
        ) or "mask"
        self._uam_debug = self.uam_cfg.get("debug", {})
        
        # SEM configuration (semantic scale-aware matching)
        self.sem_cfg = self._build_sem_cfg(sem or {})
        self.sem_enabled = self.sem_cfg.get("enable", False)
        self.sem_matcher = build_matcher(self.sem_cfg) if self.sem_enabled else None
        self._sem_debug = self.sem_cfg.get("debug", {})

        self.pp_cfg = self._build_pp_cfg((memory_bank_cfg or {}).get("pp", {}))
        self.pp_enabled = self.pp_cfg.get("enable", False)

        debug_pp = self.pp_cfg.get("debug", {})
        self.pp_debug_cfg = {
            "enable": debug_pp.get("enable", False),
            "save_spectrum": debug_pp.get("save_spectrum", False),
            "save_heatmap": debug_pp.get("save_heatmap", False),
            "save_clusters": debug_pp.get("save_clusters", False),
            "max_classes": int(debug_pp.get("max_classes", 6)),
            "out_dir": Path(debug_pp.get("out_dir", "pp_debug")),
        }

        # Models
        self.sam_transform = Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.predictor = build_sam2_video_predictor(sam2_cfg_file, sam2_ckpt_path)
        self.sam_img_size = 1024

        encoder_name = encoder_cfg.pop("name")
        encoder_args = copy.deepcopy(encoder_predefined_cfgs.get(encoder_name, {}))
        # Keep user-provided overrides
        encoder_args.update(encoder_cfg)

        encoder = build_encoder(encoder_args | {"name": encoder_name}, encoder_ckpt_path)
        encoder_img_size = encoder.img_size
        encoder_patch_size = encoder.patch_size
        encoder_hw = encoder_img_size // encoder_patch_size

        self.encoder_h, self.encoder_w = encoder_hw, encoder_hw
        self.encoder_img_size = encoder_img_size
        self.encoder_patch_size = encoder_patch_size
        self.encoder_dim = encoder.feat_dim

        self.encoder = encoder

        self.predictor.eval()
        self.encoder.model.eval()
        
        # Move encoder to same device as predictor
        if hasattr(self.predictor, 'device'):
            self.encoder.model = self.encoder.model.to(self.predictor.device)

        # Others
        memory_bank_cfg["feat_shape"] = (self.encoder_h * self.encoder_w, self.encoder_dim)
        self._init_memory_bank(memory_bank_cfg)

        self._reset()

    def to(self, *args, **kwargs):
        """Override to() to also move encoder model to the correct device"""
        result = super().to(*args, **kwargs)
        if hasattr(self, 'encoder') and hasattr(self.encoder, 'model'):
            self.encoder.model = self.encoder.model.to(*args, **kwargs)
        return result

    def _init_memory_bank(self, memory_bank_cfg):
        assert memory_bank_cfg.pop("enable")

        self.mem_n_classes = memory_bank_cfg.get("category_num")
        self.mem_length = memory_bank_cfg.get("length")
        self.mem_feat_shape = memory_bank_cfg.get("feat_shape")

        assert len(self.mem_feat_shape) == 2
        _mem_n, _mem_c = self.mem_feat_shape

        self.register_buffer(
            "mem_fill_counts", torch.zeros((self.mem_n_classes,), dtype=torch.long)
        )
        self.register_buffer(
            "mem_feats", torch.zeros((self.mem_n_classes, self.mem_length, _mem_n, _mem_c))
        )
        self.register_buffer(
            "mem_masks", torch.zeros((self.mem_n_classes, self.mem_length, _mem_n))
        )
        self.register_buffer(
            "mem_feats_avg", torch.zeros((self.mem_n_classes, _mem_c))
        )
        self.register_buffer(
            "mem_feats_ins_avg", torch.zeros((self.mem_n_classes, self.mem_length, _mem_c))
        )
        self.register_buffer(
            "mem_feats_covariances", torch.zeros((self.mem_n_classes, _mem_c, _mem_c))
        )
        self.register_buffer(
            "mem_feats_centers", torch.zeros((self.mem_n_classes, self.kmeans_k, _mem_c))
        )
        self.register_buffer(
            "mem_ins_sim_avg", torch.zeros((self.mem_n_classes,))
        )
        self.register_buffer(
            "mem_pca_mean", torch.zeros((self.mem_n_classes, _mem_c))
        )
        self.register_buffer(
            "mem_pca_components", torch.zeros((self.mem_n_classes, self.n_pca_components, _mem_c))
        )
        self.register_buffer(
            "mem_pp_prototypes", torch.zeros((self.mem_n_classes, _mem_c))
        )
        self.mem_pp_subprototypes: Dict[int, List[torch.Tensor]] = {}
        self.mem_pp_debug: Dict[int, Dict[str, Any]] = {}
        self.register_buffer("mem_postprocessed", torch.zeros((1,), dtype=torch.bool))
        self.memory_ready = False

        if self.with_negative_refs:
            self.mem_length_negative = memory_bank_cfg.get("length_negative")
            self.register_buffer(
                "mem_fill_counts_neg", torch.zeros((self.mem_n_classes,), dtype=torch.long)
            )
            self.register_buffer(
                "mem_feats_neg", torch.zeros((self.mem_n_classes, self.mem_length_negative, _mem_n, _mem_c))
            )
            self.register_buffer(
                "mem_masks_neg", torch.zeros((self.mem_n_classes, self.mem_length_negative, _mem_n))
            )
            self.register_buffer(
                "mem_feats_avg_neg", torch.zeros((self.mem_n_classes, _mem_c))
            )
            self.register_buffer(
                "mem_feats_ins_avg_neg", torch.zeros((self.mem_n_classes, self.mem_length_negative, _mem_c))
            )
            self.register_buffer("mem_postprocessed_neg", torch.zeros((1,), dtype=torch.bool))
            self.memory_neg_ready = False

    def _reset(self):
        self.backbone_features = None
        self.backbone_hr_features = None

    def _compute_matched_iou_matrix(self, gt_masks, pred_masks):
        with torch.inference_mode():
            assert len(gt_masks) == pred_masks.shape[0]

            N = len(gt_masks)
            n_pix = gt_masks[0].shape[-2] * gt_masks[0].shape[-1]
            n_points, n_output = pred_masks.shape[1], pred_masks.shape[2]

            matched_ious = []
            machted_inds = []
            for i in range(N):
                gt = gt_masks[i].reshape(1, -1, n_pix)  # [1, n_ins, n_pix]
                n_ins = gt.shape[0]
                pred = pred_masks[i].reshape(-1, 1, n_pix)  # .expand(-1, n_ins, -1)  # [n_points * n_output, n_ins, n_pix]

                intersection = torch.logical_and(gt, pred).to(dtype=torch.float)
                union = torch.logical_or(gt, pred).to(dtype=torch.float)
                iou = intersection.sum(dim=-1) / union.sum(dim=-1)
                matched_iou, matched_ins_inds = iou.max(dim=-1) # [n_points * n_output], [n_points * n_output]
                matched_ious.append(matched_iou)
                machted_inds.append(matched_ins_inds)
        return torch.cat(matched_ious, dim=0), torch.cat(machted_inds, dim=0)

    def _compute_semantic_ios(self, masks_binary, labels, obj_sim, use_semantic=True, rank_score=True):
        n_masks = masks_binary.shape[0]
        masks = masks_binary.reshape(n_masks, -1).to(dtype=torch.float32)
        ios = torch.zeros((n_masks,), device=masks_binary.device, dtype=torch.float32)

        for cat_ind in range(self.mem_n_classes):
            select_idxs = (labels == cat_ind)
            _masks = masks[select_idxs]
            _obj_sim = obj_sim[select_idxs][:, select_idxs]
            n_cat = _masks.shape[0]
            if n_cat == 0:
                continue
            pos_num = _masks.sum(dim=-1).to(dtype=torch.float32)
            inter_num = _masks @ _masks.t()
            inter_num.fill_diagonal_(0.0)
            if rank_score:
                inter_num = torch.tril(inter_num, diagonal=0)
            _ios = (inter_num / pos_num[:, None]) # .max(dim=-1)[0]
            if use_semantic:
                _ios = _ios * _obj_sim
            _ios = _ios.max(dim=-1)[0]
            ios[select_idxs] += _ios
        return ios

    def _compute_ios_batched(self, masks_binary, labels, rank_score=True, batch_size=10):
        n_masks = masks_binary.shape[0]
        masks = masks_binary.reshape(n_masks, -1).to(dtype=torch.float32)
        ios = torch.zeros((n_masks,), device=masks_binary.device, dtype=torch.float32)
        for cat_ind in range(self.mem_n_classes):
            select_idxs = (labels == cat_ind)
            _masks = masks[select_idxs]
            n_cat = _masks.shape[0]
            if n_cat == 0:
                continue
            
            # Process in batches to avoid OOM
            for i in range(0, n_cat, batch_size):
                batch_end = min(i + batch_size, n_cat)
                batch_masks = _masks[i:batch_end]
                
                pos_num = batch_masks.sum(dim=-1).to(dtype=torch.float32)
                
                # Compute inter_num in sub-batches if needed
                inter_num = torch.zeros((batch_masks.shape[0], n_cat), device=masks.device)
                for j in range(0, n_cat, batch_size):
                    j_end = min(j + batch_size, n_cat)
                    inter_num[:, j:j_end] = batch_masks @ _masks[j:j_end].t()
                
                if rank_score:
                    inter_num = torch.tril(inter_num, diagonal=0)
                    
                _ios = inter_num.max(dim=-1)[0] / pos_num
                ios[select_idxs][i:batch_end] += _ios
        return ios

    def _compute_pca_scores(self, tar_feats, binary_masks):
        n_masks = binary_masks.shape[0]

        pca_mean = self.mem_pca_mean.unsqueeze(dim=0)
        pca_components = (
            F.normalize(self.mem_pca_components, p=2, dim=-1)
            .permute(0, 2, 1)
            .unsqueeze(dim=0)
        )   # [1, n_class, c, n_components]

        scores_all = []
        for i in range(n_masks):
            foreground_feats = (
                tar_feats[binary_masks[i]]
                .unsqueeze(dim=1)
                .expand(-1, self.mem_n_classes, -1)
            )   # [n_fore, n_class, c]
            centered_feats = F.normalize(foreground_feats - pca_mean, p=2, dim=-1).unsqueeze(dim=2)
            pca_scores = centered_feats @ pca_components
            pca_scores = (pca_scores.squeeze(dim=2) + 1.0) * 0.5
            pca_scores = pca_scores.max(dim=0, keepdim=True)[0].mean(dim=-1)
            scores_all.append(pca_scores)
        scores_all = torch.cat(scores_all, dim=0)
        return scores_all

    def _compute_query_points(self, tar_feats, matching_size, num_points):
        assert matching_size[0] * matching_size[1] >= num_points
        device = tar_feats.device

        matching_h, matching_w = matching_size
        c = self.encoder_dim

        x, y = torch.meshgrid(
            torch.linspace(0, matching_w - 1, matching_w),
            torch.linspace(0, matching_h - 1, matching_h)
        )
        x = x + 0.5
        y = y + 0.5
        all_points = torch.stack((x.reshape(-1) / matching_w, y.reshape(-1) / matching_h), dim=-1)
        all_points = all_points.to(device=device)

        _tar_feats = tar_feats.reshape(1, self.encoder_h, self.encoder_w, c).permute(0, 3, 1, 2)
        _tar_feats = F.interpolate(
            _tar_feats,
            size=(matching_h, matching_w),
            mode="bilinear",
            align_corners=False,
            antialias=True
        ).squeeze(dim=0).reshape(c, -1).t()
        _tar_feats = F.normalize(_tar_feats, p=2, dim=-1)  # [n, c]

        mem_feats_avg = self.mem_feats_avg  # [n_class, c]
        mem_feats_avg = F.normalize(mem_feats_avg, p=2, dim=-1)

        sim = _tar_feats @ mem_feats_avg.t()
        _, top_inds = torch.topk(sim.max(dim=1)[0], k=num_points)
        query_points = all_points[top_inds].reshape(num_points, 2)
        return query_points

    def _compute_ambiguous_decay(self, sim_global, labels):
        assert self.cls_num_per_mask == 1
        mem_feats_avg = self.mem_feats_avg  # [n_class, c]
        mem_feats_avg = F.normalize(mem_feats_avg, p=2, dim=-1)

        mem_sim_mat = mem_feats_avg @ mem_feats_avg.t()
        mem_sim_mat = (mem_sim_mat + 1.0) * 0.5
        mem_sim_mat.fill_diagonal_(0.5)

        mem_sim_select = mem_sim_mat[labels]
        decay = -1.0 * (sim_global - mem_sim_select).clamp(min=0.0).pow(2.0)  # [n_mask, c]
        weights = (mem_sim_select - 0.5).clamp(min=0.0)
        decay = (decay * weights).sum(dim=-1) / (weights.sum(dim=-1) + 1e-10)

        decay = torch.exp(decay / 0.1**2)
        # weights = (mem_sim_select - 0.5).clamp(min=0.0)
        # decay = (decay * weights).sum(dim=-1) / (weights.sum(dim=-1) + 1e-10)
        return decay

    def _get_oracle_iou(self, lr_masks_all, tar_anns_by_cat, matching_size=None):
        n_masks = lr_masks_all.shape[0]
        lr_masks_all = lr_masks_all.reshape(1, n_masks, *lr_masks_all.shape[-2:])

        if matching_size is not None:
            assert lr_masks_all.shape[-2] == lr_masks_all.shape[-1]
            matching_h, matching_w = matching_size, matching_size
            lr_masks_all = F.interpolate(
                lr_masks_all,
                size=(matching_h, matching_w),
                align_corners=False,
                mode="bilinear",
                antialias=True,
            ).squeeze(dim=1)
        else:
            matching_h, matching_w = lr_masks_all.shape[-2], lr_masks_all.shape[-1]
        lr_masks_all = lr_masks_all > 0

        scores_oracle = torch.zeros((self.mem_n_classes, n_masks), device=lr_masks_all.device)

        for cat_ind in tar_anns_by_cat.keys():
            gt_masks_cat = tar_anns_by_cat[cat_ind]["masks"].to(dtype=torch.float, device=lr_masks_all.device)
            gt_masks_cat = F.interpolate(
                gt_masks_cat.unsqueeze(dim=1),
                size=(matching_h, matching_w),
                mode="nearest"
            ).squeeze(dim=1).bool()
            matched_iou, _ = self._compute_matched_iou_matrix([gt_masks_cat], lr_masks_all)
            matched_iou = matched_iou.reshape(n_masks)
            scores_oracle[cat_ind] += matched_iou
        scores_oracle = scores_oracle.reshape(self.mem_n_classes, n_masks)
        return scores_oracle

    def _get_oracle_refine_prompts(
        self,
        lr_masks_all,
        pred_labels_all,
        tar_anns_by_cat,
        pool_size=7,
        mask_stride=4
    ):
        device = lr_masks_all.device
        n_masks = lr_masks_all.shape[0]

        matching_h, matching_w = lr_masks_all.shape[-2], lr_masks_all.shape[-1]
        lr_masks_all = lr_masks_all > 0

        start = 0.5 / pool_size
        end = 1.0 - start
        intervals = torch.linspace(start, end, pool_size).to(dtype=torch.float32, device=device)
        grid_x, grid_y = torch.meshgrid(intervals, intervals)
        grid_x = grid_x.reshape(-1)
        grid_y = grid_y.reshape(-1)

        refine_points = torch.zeros((n_masks, pool_size**2, 2), dtype=torch.float32, device=device)
        refine_labels = torch.zeros((n_masks, pool_size**2), dtype=torch.float32, device=device)
        do_refine = torch.zeros((n_masks,), dtype=torch.float32, device=device)

        for cat_ind in tar_anns_by_cat.keys():
            gt_masks_cat = tar_anns_by_cat[cat_ind]["masks"].to(dtype=torch.float, device=device)
            gt_masks_cat = F.interpolate(
                gt_masks_cat.unsqueeze(dim=1),
                size=(matching_h, matching_w),
                mode="nearest"
            ).squeeze(dim=1).bool()

            cat_matched_inds = pred_labels_all==cat_ind
            lr_masks_cat = lr_masks_all[cat_matched_inds]
            n_masks_cat = lr_masks_cat.shape[0]
            if n_masks_cat == 0:
                continue
            bboxes_cat = batched_mask_to_box(lr_masks_cat)  # [n_cat, 4]
            lr_masks_cat = lr_masks_cat.unsqueeze(dim=0)
            _, matched_ind = self._compute_matched_iou_matrix([gt_masks_cat], lr_masks_cat)
            matched_ind = matched_ind.reshape(n_masks_cat)

            matched_gt_masks = gt_masks_cat[matched_ind].reshape(n_masks_cat, -1)
            lr_masks_cat = lr_masks_cat.reshape(n_masks_cat, -1)
            is_correct = torch.logical_and(lr_masks_cat, matched_gt_masks).to(dtype=torch.float32)

            x1, y1, x2, y2 = bboxes_cat[:, 0:1], bboxes_cat[:, 1:2], bboxes_cat[:, 2:3], bboxes_cat[:, 3:4]
            xs = x1 + (x2 - x1) * grid_x  # [n_masks_cat, pool_size * pool_size]
            ys = y1 + (y2 - y1) * grid_y  # [n_masks_cat, pool_size * pool_size]

            xs = xs.clamp(min=0, max=matching_w-1)
            ys = ys.clamp(min=0, max=matching_h-1)
            sample_pos = ys.to(dtype=torch.long) * matching_h + xs.to(dtype=torch.long)  # [n_masks_cat, pool_size * pool_size]
            is_correct_sampled = torch.gather(is_correct, 1, sample_pos)   # [n_masks_cat, pool_size * pool_size]

            do_refine[cat_matched_inds] += 1
            refine_points[cat_matched_inds] += torch.stack((xs, ys), dim=-1) * mask_stride
            refine_labels[cat_matched_inds] += is_correct_sampled

        return refine_points, refine_labels, do_refine

    def _get_refine_prompts(
        self,
        tar_feats,
        lr_masks_all,
        pred_labels_all,
        pool_size=7,
        mask_stride=4,
        thr=0.6
    ):
        device = lr_masks_all.device
        n_masks = lr_masks_all.shape[0]
        mask_h, mask_w = lr_masks_all.shape[-2:]

        lr_masks_all = lr_masks_all > 0
        bboxes = batched_mask_to_box(lr_masks_all)

        start = 0.5 / pool_size
        end = 1.0 - start
        intervals = torch.linspace(start, end, pool_size).to(dtype=torch.float32, device=device)
        grid_x, grid_y = torch.meshgrid(intervals, intervals)
        grid_x = grid_x.reshape(-1)
        grid_y = grid_y.reshape(-1)

        x1, y1, x2, y2 = bboxes[:, 0:1], bboxes[:, 1:2], bboxes[:, 2:3], bboxes[:, 3:4]
        xs = x1 + (x2 - x1) * grid_x  # [n_masks, pool_size * pool_size]
        ys = y1 + (y2 - y1) * grid_y  # [n_masks, pool_size * pool_size]
        sampled_points_lr = torch.stack((xs, ys), dim=-1)

        xs = xs.clamp(min=0, max=mask_w) / mask_w
        ys = ys.clamp(min=0, max=mask_h) / mask_h
        sampled_points_normed = torch.stack((xs, ys), dim=-1)

        grid = sampled_points_normed.reshape(1, 1, -1, 2)
        grid = (grid - 0.5) * 2.0  # normalise to [-1, 1]

        tar_feats = tar_feats.reshape(1, self.encoder_h, self.encoder_w, self.encoder_dim).permute(0, 3, 1, 2)
        sampled_feats = F.grid_sample(tar_feats, grid).reshape(n_masks, pool_size**2, -1)
        sampled_feats = F.normalize(sampled_feats, p=2, dim=-1)

        temp_feats = self.mem_feats_avg[pred_labels_all].unsqueeze(dim=-1)  # [n_masks, c]
        temp_feats = F.normalize(temp_feats, p=2, dim=1)

        sim = sampled_feats @ temp_feats
        sim = (sim.reshape(n_masks, pool_size**2) + 1) * 0.5  # [n_masks, pool_size**2]

        sampled_points = sampled_points_lr * mask_stride
        sampled_labels = sim > thr
        return sampled_points, sampled_labels

    def _forward_encoder(self, imgs):
        assert len(imgs.shape) == 4
        # Use unified encoder path; attention is optional
        feats, last_attn = self.encoder.tokens_with_attn(imgs, normalize=True)
        return feats, last_attn

    def _forward_encoder_attn_roll(self, imgs):
        # Use unified encoder path; for DINOv3 attention is None
        feats, last_attn = self.encoder.tokens_with_attn(imgs, normalize=True)
        return feats, last_attn

    def _forward_sam_decoder(
        self,
        backbone_features,
        sparse_embeddings,
        dense_embeddings,
        backbone_hr_features,
        multimask_output=True
    ):
        B = backbone_features.shape[0]
        device = backbone_features.device
        (
            low_res_multimasks,
            ious,
            _,
            _,
        ) = self.predictor.sam_mask_decoder(
            image_embeddings=backbone_features,
            image_pe=self.predictor.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            repeat_image=False,  # the image is already batched
            high_res_features=backbone_hr_features,
            return_iou_token_out=False,
            disable_custom_iou_embed=True,
            disable_mlp_obj_scores=True,
            output_all_masks=True,
        )

        n_pred = ious.shape[-1]
        assert n_pred == low_res_multimasks.shape[1]

        # We skip the SAM2's multimask_output but use the custom IoU to determine the output mask
        # TODO: add advanced mask postprocessing tricks in the sam2 auto mask generator
        if multimask_output:
            best_iou_inds = torch.argmax(ious[:, 1:], dim=-1) + 1
            batch_inds = torch.arange(B, device=device)
            low_res_masks = low_res_multimasks[batch_inds, best_iou_inds]
            scores = ious[batch_inds, best_iou_inds]
        else:
            low_res_masks = low_res_multimasks[:, 0]
            scores = ious[:, 0]
        return low_res_masks, scores

    def _compute_masks(
        self,
        backbone_features,
        backbone_hr_features,
        point_inputs
    ):
        '''
        Similar to SAM2Base._forward_sam_heads. Putting it here for easy customization
        '''
        B = backbone_features.size(0)
        device = self.predictor.device
        assert backbone_features.size(1) == self.predictor.sam_prompt_embed_dim
        assert backbone_features.size(2) == self.predictor.sam_image_embedding_size
        assert backbone_features.size(3) == self.predictor.sam_image_embedding_size

        sam_point_coords = point_inputs["point_coords"]
        sam_point_labels = point_inputs["point_labels"]
        assert sam_point_coords.size(0) == B and sam_point_labels.size(0) == B

        sparse_embeddings, dense_embeddings = self.predictor.sam_prompt_encoder(
            points=(sam_point_coords, sam_point_labels),
            boxes=None,
            masks=None,
        )
        low_res_masks, scores = self._forward_sam_decoder(
            backbone_features,
            sparse_embeddings,
            dense_embeddings,
            backbone_hr_features,
            multimask_output=True
        )
        return low_res_masks, scores

    def _compute_masks_refine(
        self,
        point_inputs,
        boxes_inputs,
        mask_inputs
    ):
        assert self.backbone_features is not None
        assert self.backbone_hr_features is not None
        # assert mask_inputs is not None

        backbone_features = self.backbone_features  # [1, c, h, w]
        backbone_hr_features = self.backbone_hr_features  # each [1, c, h, w]

        B = point_inputs["point_coords"].size(0)
        device = self.predictor.device

        backbone_features = backbone_features.expand(B, -1, -1, -1)
        backbone_hr_features = [x.expand(B, -1, -1, -1) for x in backbone_hr_features]

        if point_inputs is not None:
            points = (point_inputs["point_coords"], point_inputs["point_labels"])
        else:
            points = None
        if mask_inputs is not None:
            mask_inputs = mask_inputs.reshape(B, 1, *mask_inputs.shape[-2:]).to(dtype=backbone_features.dtype)
        else:
            mask_inputs = None

        sparse_embeddings, dense_embeddings = self.predictor.sam_prompt_encoder(
            points=points,
            boxes=None,
            masks=mask_inputs,
        )

        low_res_masks, scores = self._forward_sam_decoder(
            backbone_features,
            sparse_embeddings,
            dense_embeddings,
            backbone_hr_features,
            multimask_output=True
        )
        return low_res_masks, scores

    def _forward_sam_multiscale(self, imgs, scales=(0.5, 0.75, 1.0)):
        assert len(imgs.shape) == 4
        assert imgs.shape[-2] == imgs.shape[-1]
        assert self.backbone_features is None
        assert self.backbone_hr_features is None

        device = imgs.device

        sam_input_size = imgs.shape[-2]
        points_per_side = self.points_per_side

        lr_masks_all, scores_all = [], []
        for scale in scales:
            self.backbone_features = None
            self.backbone_hr_features = None

            if scale == 1.0:
                hw = sam_input_size
                padded_imgs = imgs
            else:
                hw = int(scale * sam_input_size)
                resized_imgs = F.interpolate(imgs, size=(hw, hw), mode='bicubic')
                padded_imgs = torch.zeros_like(imgs)  # use SAM's original input size to avoid potential bugs
                padded_imgs[:, :, :hw, :hw] += resized_imgs

            x, y = torch.meshgrid(
                torch.linspace(0, hw - 1, points_per_side),
                torch.linspace(0, hw - 1, points_per_side)
            )
            query_points = torch.stack((x.reshape(-1), y.reshape(-1)), dim=-1)
            query_points += 0.5
            query_points = query_points.to(device=device)

            lr_masks, scores, _ = self._forward_sam(self.sam_transform(padded_imgs), query_points, point_normed=False)
            mask_hw = lr_masks.shape[-2]
            valid_mask_hw = int(mask_hw * scale)
            lr_masks = lr_masks[:, :valid_mask_hw, :valid_mask_hw]
            if valid_mask_hw != mask_hw:
                lr_masks = F.interpolate(
                    lr_masks.unsqueeze(dim=1),
                    size=(mask_hw, mask_hw),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True
                ).squeeze(dim=1)
            lr_masks_all.append(lr_masks)
            scores_all.append(scores)

        return torch.cat(lr_masks_all, dim=0), torch.cat(scores_all, dim=0), None

    def _forward_sam(self, imgs, precomputed_points=None, point_normed=True):
        assert len(imgs.shape) == 4
        assert imgs.shape[-2] == imgs.shape[-1]
        assert self.backbone_features is None
        assert self.backbone_hr_features is None

        device = imgs.device

        sam_input_size = imgs.shape[-2]
        points_per_side = self.points_per_side
        testing_point_bs = self.testing_point_bs
        iou_thr = self.iou_thr

        # Prepare input
        if precomputed_points is None:
            x, y = torch.meshgrid(
                torch.linspace(0, sam_input_size-1, points_per_side),
                torch.linspace(0, sam_input_size-1, points_per_side)
            )
            query_points = torch.stack((x.reshape(-1), y.reshape(-1)), dim=-1)
            query_points += 0.5
            query_points = query_points.to(device=device)
        else:
            if point_normed:
                query_points = precomputed_points * sam_input_size
            else:
                query_points = precomputed_points


        # forward model
        backbone_out = self.predictor.forward_image(imgs)
        _, img_vision_features, img_vision_pos_embeds, img_feat_sizes = (
            self.predictor._prepare_backbone_features(backbone_out)
        )

        img_feats = img_vision_features[-1].permute(1, 2, 0).reshape(1, -1, *img_feat_sizes[-1])
        self.backbone_features = img_feats
        img_feats = img_feats.expand(testing_point_bs, -1, -1, -1)

        hr_feats = [
            x.permute(1, 2, 0).reshape(1, -1, *s)
            for x, s in zip(img_vision_features[:-1], img_feat_sizes[:-1])
        ]
        self.backbone_hr_features = hr_feats
        hr_feats = [
            x.expand(testing_point_bs, -1, -1, -1) for x in hr_feats
        ]

        points = query_points.reshape(-1, 2)
        point_labels = torch.ones_like(points[:, 0:1]).to(dtype=torch.int32)
        n_points = points.shape[0]

        mask_scores = []
        lr_masks = []
        for i in range(0, n_points // testing_point_bs):
            i_start = i * testing_point_bs
            i_end = i_start + testing_point_bs
            points_i = points[i_start:i_end, :]
            p_labels_i = point_labels[i_start:i_end, :]
            point_inputs_i = dict(
                point_coords=points_i.reshape(testing_point_bs, 1, 2),
                point_labels=p_labels_i.reshape(testing_point_bs, 1)
            )
            lr_masks_i, scores_i = self._compute_masks(
                img_feats, hr_feats, point_inputs_i
            )
            mask_scores.append(scores_i.reshape(-1))
            lr_masks.append(lr_masks_i.reshape(-1, *lr_masks_i.shape[-2:]))
        scores_all = torch.cat(mask_scores, dim=0).reshape(-1)
        lr_masks_all = torch.cat(lr_masks, dim=0)
        lr_masks_all = lr_masks_all.reshape(-1, *lr_masks_all.shape[-2:])

        inds = scores_all > iou_thr
        points_all = points[inds]
        lr_masks_all = lr_masks_all[inds]
        scores_all = scores_all[inds]

        return lr_masks_all, scores_all, points_all

    def forward_fill_memory(self, input_dicts, is_positive):
        with torch.inference_mode():
            assert len(input_dicts) == 1

            device = self.predictor.device

            ref_cat_ind = list(input_dicts[0]["refs_by_cat"].keys())[0]

            ref_imgs = input_dicts[0]["refs_by_cat"][ref_cat_ind]["imgs"].to(device=device)
            ref_masks = input_dicts[0]["refs_by_cat"][ref_cat_ind]["masks"].to(dtype=ref_imgs.dtype)

            ref_imgs = F.interpolate(
                ref_imgs,
                size=(self.encoder_img_size, self.encoder_img_size),
                mode="bicubic"
            )
            ref_feats, _ = self._forward_encoder(ref_imgs)
            ref_feats = ref_feats.reshape(1, -1, self.encoder_dim)

            ref_masks = F.interpolate(
                ref_masks.unsqueeze(dim=0),
                size=(self.encoder_h, self.encoder_w),
                mode="nearest"
            ).reshape(1, -1)

            cat_ind_tensor = torch.tensor([ref_cat_ind], dtype=torch.long, device=device).reshape(1, 1)
            cat_ind_all = concat_all_gather(cat_ind_tensor).reshape(-1).to(dtype=torch.long).detach()
            feats_all = concat_all_gather(ref_feats.contiguous())
            masks_all = concat_all_gather(ref_masks.contiguous())

            for i in range(cat_ind_all.shape[0]):
                if is_positive:
                    if dist.is_initialized():
                        assert (self.mem_n_classes * self.mem_length) % dist.get_world_size() == 0
                    fill_ind = self.mem_fill_counts[cat_ind_all[i]]
                    self.mem_feats[cat_ind_all[i], fill_ind] += feats_all[i]
                    self.mem_masks[cat_ind_all[i], fill_ind] += masks_all[i]
                    self.mem_fill_counts[cat_ind_all[i]] += 1
                else:
                    if dist.is_initialized():
                        assert (self.mem_n_classes * self.mem_length_negative) % dist.get_world_size() == 0
                    fill_ind = self.mem_fill_counts_neg[cat_ind_all[i]]
                    self.mem_feats_neg[cat_ind_all[i], fill_ind] += feats_all[i]
                    self.mem_masks_neg[cat_ind_all[i], fill_ind] += masks_all[i]
                    self.mem_fill_counts_neg[cat_ind_all[i]] += 1

            return {}

    def forward_vis_memory(self, input_dicts):
        assert len(input_dicts) == 1
        assert self.mem_fill_counts[0].item() > 0
        assert self.n_pca_components == 3  # RGB

        device = self.predictor.device
        output_dir = "./results_analysis/memory_vis"

        ref_cat_ind = list(input_dicts[0]["refs_by_cat"].keys())[0]

        ref_imgs = input_dicts[0]["refs_by_cat"][ref_cat_ind]["imgs"].to(device=device)
        ref_imgs = F.interpolate(
            ref_imgs,
            size=(self.encoder_img_size, self.encoder_img_size),
            mode="bicubic"
        )
        ref_feats, _ = self._forward_encoder(ref_imgs)
        ref_feats = ref_feats.reshape(-1, self.encoder_dim)

        ref_masks_ori = input_dicts[0]["refs_by_cat"][ref_cat_ind]["masks"].to(dtype=ref_feats.dtype, device=device)
        ref_masks = F.interpolate(
            ref_masks_ori.unsqueeze(dim=0),
            size=(self.encoder_h, self.encoder_w),
            mode="nearest"
        ).reshape(-1)

        encoder_shape_info = dict(
            height=self.encoder_h,
            width=self.encoder_w,
            patch_size=self.encoder_patch_size
        )

        pca_vis_result = vis_pca(
            ref_imgs,
            ref_masks_ori,
            ref_cat_ind,
            ref_feats,
            ref_masks,
            self.mem_pca_mean,
            self.mem_pca_components,
            encoder_shape_info,
            device,
            transparency=1.0
        )
        kmeans_vis_result = vis_kmeans(
            ref_imgs,
            ref_masks_ori,
            ref_cat_ind,
            ref_feats,
            ref_masks,
            self.mem_feats_centers,
            encoder_shape_info,
            device,
            transparency=1.0
        )
        ori_img = ref_imgs[0].permute(1, 2, 0) * 255.0
        margin = torch.zeros((ori_img.shape[0], 5, 3), dtype=ori_img.dtype, device=device) + 255
        output_final = torch.cat((
            ori_img, margin, kmeans_vis_result, margin, pca_vis_result
        ), dim=1)

        import os
        from PIL import Image

        out_vis_img = Image.fromarray(output_final.cpu().numpy().astype(np.uint8))
        img_id = int(input_dicts[0]["refs_by_cat"][ref_cat_ind]["img_info"][0]['id'])
        out_vis_img.save(os.path.join(output_dir, "%d_%d.png" % (ref_cat_ind, img_id)))
        return {}

    def _compute_sim_attn_guided_global_avg(self, tar_feat, attn_weights, masks_feat_size_bool):
        n_masks = masks_feat_size_bool.shape[0]

        # bboxes = batched_mask_to_box(masks_feat_size_bool.reshape(-1, self.encoder_h, self.encoder_h))
        # box_size = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        # box_scale = box_size / (self.encoder_h * self.encoder_w)

        attn_weights = attn_weights.reshape(tar_feat.shape[0], tar_feat.shape[0])
        tar_avg_feats_all = []
        tar_sizes = []
        hit_ratios = []
        for i in range(n_masks):
            feats_i = tar_feat[masks_feat_size_bool[i]]
            fore_attn = attn_weights[masks_feat_size_bool[i]]

            hit_attn = fore_attn[:, masks_feat_size_bool[i]]
            hit_ratios.append(hit_attn.sum() / fore_attn.sum())

            tar_avg_feats_all.append(feats_i.mean(dim=0, keepdim=True))
            tar_sizes.append(torch.ones_like(feats_i[:, 0]).sum())
            # avg_feat = (feats_i * hit_ratio.unsqueeze(dim=1)).sum(dim=0) / hit_ratio.sum()
            # tar_avg_feats_all.append(avg_feat.unsqueeze(dim=0))

        tar_avg_feats = torch.cat(tar_avg_feats_all, dim=0)  # [n_mask, c]
        tar_avg_feats = F.normalize(tar_avg_feats, p=2, dim=-1)

        tar_sizes = torch.stack(tar_sizes).reshape(-1, 1)
        tar_scale = tar_sizes / (self.encoder_w * self.encoder_h)

        mem_feats_avg = self.mem_feats_avg  # [n_class, c]
        mem_feats_avg = F.normalize(mem_feats_avg, p=2, dim=-1)
        sim_avg = tar_avg_feats @ mem_feats_avg.t()  # [n_masks, n_class]
        sim_avg = sim_avg.clamp(min=0.0)

        hit_ratios = torch.stack(hit_ratios).reshape(n_masks, 1)
        sigma = (1 - tar_scale) * 3.0
        # sigma = 2.0
        sim_avg = sim_avg * torch.exp(-(1 - hit_ratios)**2 / sigma**2)
        return sim_avg

    def _compute_sim_attn_guided_global_weighted(self, tar_feat, attn_weights, masks_feat_size_bool):
        n_masks = masks_feat_size_bool.shape[0]

        attn_weights = attn_weights.reshape(tar_feat.shape[0], tar_feat.shape[0])
        tar_avg_feats_all = []
        for i in range(n_masks):
            feats_i = tar_feat[masks_feat_size_bool[i]]
            fore_attn = attn_weights[masks_feat_size_bool[i]][:, masks_feat_size_bool[i]]

            importance = fore_attn.mean(dim=1).unsqueeze(dim=1)

            feats_i_avg = (feats_i * importance).sum(dim=0) / importance.sum()
            tar_avg_feats_all.append(feats_i_avg.unsqueeze(dim=0))

        tar_avg_feats = torch.cat(tar_avg_feats_all, dim=0)  # [n_mask, c]
        tar_avg_feats = F.normalize(tar_avg_feats, p=2, dim=-1)

        mem_feats_avg = self.mem_feats_avg  # [n_class, c]
        mem_feats_avg = F.normalize(mem_feats_avg, p=2, dim=-1)

        sim_avg = tar_avg_feats @ mem_feats_avg.t()  # [n_masks, n_class]
        sim_avg = sim_avg.clamp(min=0.0)
        return sim_avg

    def _compute_sim_gaussian(self, tar_feat, masks_feat_size_bool):
        n_classes = self.mem_n_classes
        mem_feats = self.mem_feats_ins_avg  # [n_class, n_ins, c]

        mu = mem_feats.mean(dim=1)  # [n_class, c]

        feats_centered = mem_feats - mu.unsqueeze(dim=1)  # [n_class, n_ins, c]
        sigma = feats_centered.transpose(-1, -2) @ feats_centered / float(self.mem_length)  # [n_class, c, c]


        inv_sigma = torch.linalg.inv(sigma)  # [n_class, c, c]
        #
        # inverse_det_sigma = 1. / torch.sqrt(torch.det(sigma))  # [n_class]
        # print(torch.det(sigma))
        # exit()
        # inverse_det_sigma = inverse_det_sigma.unsqueeze(dim=0)

        n_masks = masks_feat_size_bool.shape[0]
        masks = masks_feat_size_bool.to(dtype=tar_feat.dtype)
        x = (masks @ tar_feat) / masks.sum(dim=-1, keepdim=True)  # [n_masks, c]

        scores_all = []
        for i in range(n_classes):
            x_centered = x - mu[i:i+1, :]  # [n_masks, c]
            x_centered = x_centered.unsqueeze(dim=1)  # [n_masks, 1, c]
            score = torch.exp(-0.5 * (x_centered @ inv_sigma[i:i+1] @ x_centered.transpose(-1, -2)))
            scores_all.append(score.reshape(n_masks, 1))
        scores_all = torch.cat(scores_all, dim=1) # * inverse_det_sigma

        return scores_all.reshape(n_masks, n_classes)

    def _compute_sim_global_avg(self, tar_feat, masks_feat_size_bool, softmax=False, temp=1.0, ret_feats=False):
        n_masks = masks_feat_size_bool.shape[0]

        masks = masks_feat_size_bool.to(dtype=tar_feat.dtype)
        tar_avg_feats = (masks @ tar_feat) / masks.sum(dim=-1, keepdim=True)
        tar_avg_feats = F.normalize(tar_avg_feats, p=2, dim=-1)

        # mem_feats_avg = self.mem_feats_avg  # [n_class, c]
        # mem_feats_avg = F.normalize(mem_feats_avg, p=2, dim=-1)
        #
        mem_feats_avg = self.mem_feats_ins_avg.mean(dim=1)  # [n_class, n_ins, c]
        mem_feats_avg = F.normalize(mem_feats_avg, p=2, dim=-1)

        sim_avg = tar_avg_feats @ mem_feats_avg.t()  # [n_masks, n_class]
        if softmax:
            sim_avg = torch.softmax(sim_avg / temp, dim=-1)
        else:
            sim_avg = sim_avg.clamp(min=0.0)
        if not ret_feats:
            return sim_avg
        else:
            return sim_avg, tar_avg_feats

    def _compute_sim_global_l2(self, tar_feat, masks_feat_size_bool, sigma=1.0):
        n_masks = masks_feat_size_bool.shape[0]

        masks = masks_feat_size_bool.to(dtype=tar_feat.dtype)
        tar_avg_feats = (masks @ tar_feat) / masks.sum(dim=-1, keepdim=True)

        mem_feats_ins = self.mem_feats_ins_avg  # [n_class, n_ins, c]
        ins_dist = fast_l2(mem_feats_ins, mem_feats_ins, sqrt=True)
        ins_variance = ins_dist.reshape(self.mem_n_classes, -1).mean(dim=1)

        mem_feats_avg = self.mem_feats_avg   # [n_class, c]

        dist = fast_l2(tar_avg_feats, mem_feats_avg, sqrt=True)
        # norm = torch.pow(ins_variance, 0.15).unsqueeze(dim=0)
        # dist = dist / norm
        scores = torch.exp(-dist / sigma)
        return scores

    def _compute_completeness_decay(self, tar_feat, masks_feat_size_bool, global_sim, labels, decay=0.6):
        n_masks = masks_feat_size_bool.shape[0]

        tar_feat_normed = F.normalize(tar_feat, p=2, dim=-1)

        mem_feats_avg = self.mem_feats_avg  # [n_class, c]
        mem_feats_avg = F.normalize(mem_feats_avg, p=2, dim=-1)

        completenesses = []
        for i in range(n_masks):
            feats_i = tar_feat_normed[masks_feat_size_bool[i]]
            template_i = mem_feats_avg[labels[i]].unsqueeze(dim=1)

            sim_i = feats_i @ template_i
            sim_i = sim_i.clamp(min=0.0).flatten()

            thr_i = sim_i.max() * decay
            completeness = (sim_i > thr_i).sum() / torch.ones_like(sim_i).sum()
            completenesses.append(completeness)

        completenesses = torch.stack(completenesses).flatten()
        return completenesses

    def _compute_unification_decay(self, tar_feat, masks_feat_size_bool):
        device = tar_feat.device
        n_masks = masks_feat_size_bool.shape[0]

        # tar_feat_normed = F.normalize(tar_feat, p=2, dim=-1)

        unificationesses = []
        for i in range(n_masks):
            feats_i = tar_feat[masks_feat_size_bool[i]]
            if feats_i.shape[0] <= 1:
                unificationesses.append(torch.ones((1,), device=device))
                continue

            centers_i = kmeans(feats_i, k=2, n_iter=50)
            centers_i_normed = F.normalize(centers_i, p=2, dim=-1)
            center_sim = centers_i_normed @ centers_i_normed.t()
            unificationess = center_sim[0, 1]
            unificationesses.append(unificationess.reshape(1))

        unificationesses = torch.stack(unificationesses).flatten()
        return unificationesses

    def _compute_negative_decay(self, tar_feat, masks_feat_size_bool, sim_pos, labels):
        n_masks = masks_feat_size_bool.shape[0]
        c = tar_feat.shape[-1]

        tar_avg_feats_all = []
        for i in range(n_masks):
            feats_i = tar_feat[masks_feat_size_bool[i]]
            tar_avg_feats_all.append(feats_i.mean(dim=0, keepdim=True))

        tar_avg_feats = torch.cat(tar_avg_feats_all, dim=0)  # [n_mask, c]
        tar_avg_feats = F.normalize(tar_avg_feats, p=2, dim=-1)

        mem_feats_ins_avg_neg = self.mem_feats_ins_avg_neg[labels]  # [n_masks, n_ins, c]
        mem_feats_ins_avg_neg = F.normalize(mem_feats_ins_avg_neg, p=2, dim=-1)

        sim_neg = tar_avg_feats.unsqueeze(dim=1) @ mem_feats_ins_avg_neg.transpose(-1, -2)  # [n_masks, n_ins]
        sim_neg = sim_neg.clamp(min=0.0).squeeze(dim=1).max(dim=-1)[0]
        return sim_neg

    def _compute_kmeans_decay(self, tar_feat, masks_feat_size_bool):
        pass

    def _compute_sim_global_avg_with_neg(self, tar_feat, masks_feat_size_bool, sigma=1.0):
        n_masks = masks_feat_size_bool.shape[0]
        c = tar_feat.shape[-1]

        masks = masks_feat_size_bool.to(dtype=tar_feat.dtype)
        tar_avg_feats = (masks @ tar_feat) / masks.sum(dim=-1, keepdim=True)
        tar_avg_feats = F.normalize(tar_avg_feats, p=2, dim=-1)

        mem_feats_avg = self.mem_feats_avg  # [n_class, c]
        mem_feats_avg = F.normalize(mem_feats_avg, p=2, dim=-1)

        # mem_feats_avg = self.mem_feats_ins_avg.mean(dim=1)  # [n_class, n_ins, c]
        # mem_feats_avg = F.normalize(mem_feats_avg, p=2, dim=-1)

        mem_feats_ins_avg_neg = self.mem_feats_ins_avg_neg  # [n_class, n_ins, c]
        mem_feats_ins_avg_neg = F.normalize(mem_feats_ins_avg_neg, p=2, dim=-1).reshape(-1, c)
        
        pos_neg_sim = mem_feats_avg.unsqueeze(dim=1) @ mem_feats_ins_avg_neg.reshape(self.mem_n_classes, -1, c).transpose(-1,- 2)
        pos_neg_sim = pos_neg_sim.reshape(self.mem_n_classes, self.mem_length_negative)  # [n_class, n_ins]

        sim_pos = tar_avg_feats @ mem_feats_avg.t()  # [n_masks, n_class]
        sim_pos = sim_pos.clamp(min=0.0)

        sim_neg = tar_avg_feats @ mem_feats_ins_avg_neg.t()
        sim_neg = sim_neg.clamp(min=0.0)
        sim_neg = sim_neg.reshape(n_masks, self.mem_n_classes, -1)
        sim_neg, max_inds = sim_neg.max(dim=-1)

        # pos_neg_sim_selected = []
        # _arr_inds = torch.arange(self.mem_n_classes, device=tar_feat.device)
        # for i in range(n_masks):
        #     pos_neg_sim_selected.append(
        #         pos_neg_sim[_arr_inds, max_inds[i]].unsqueeze(dim=0)
        #     )  # [n_class]
        # pos_neg_sim_selected = torch.cat(pos_neg_sim_selected, dim=0)  # [n_masks, n_class]

        # sim_final = sim_pos * torch.exp(-1.0 * (sim_neg / (sim_pos+1e-10)).clamp(min=0.0) / sigma)
        sim_final = sim_pos * torch.exp(-1.0 * (sim_neg - sim_pos).clamp(min=0.0) / sigma)
        # sim_final = sim_pos * torch.exp(-1.0 * sim_neg.clamp(min=0.0) / sigma)
        # decay_term = sim_neg.clamp(min=0.0) / pos_neg_sim_selected.clamp(min=1e-10)
        # sim_final = sim_pos * torch.exp(-1.0 * decay_term / sigma)
        return sim_final

    def _compute_sim_instance_softmax(self, tar_feat, masks_feat_size_bool, temp=1.0):
        n_masks = masks_feat_size_bool.shape[0]
        c = tar_feat.shape[-1]

        tar_avg_feats_all = []
        for i in range(n_masks):
            feats_i = tar_feat[masks_feat_size_bool[i]]
            tar_avg_feats_all.append(feats_i.mean(dim=0, keepdim=True))

        tar_avg_feats = torch.cat(tar_avg_feats_all, dim=0)  # [n_mask, c]
        tar_avg_feats = F.normalize(tar_avg_feats, p=2, dim=-1)

        mem_ins_feats_avg = F.normalize(self.mem_feats_ins_avg.flatten(0, 1), p=2, dim=-1)

        sim_avg = tar_avg_feats @ mem_ins_feats_avg.t()
        sim_avg = sim_avg.reshape(n_masks, self.mem_n_classes, self.mem_length)

        scores = torch.softmax((sim_avg / temp).sum(dim=-1), dim=-1)
        # scores = torch.softmax(sim_avg / temp, dim=-1).reshape(n_masks, self.mem_n_classes, self.mem_length)
        # scores = scores.sum(dim=-1)
        return scores

    def _compute_sim_center(self, tar_feat, masks_feat_size_bool):
        n_masks = masks_feat_size_bool.shape[0]

        bboxes = batched_mask_to_box(masks_feat_size_bool.reshape(n_masks, self.encoder_h, self.encoder_w))
        x_centers = (bboxes[:, 0] + bboxes[:, 2]) * 0.5
        y_centers = (bboxes[:, 1] + bboxes[:, 3]) * 0.5
        x_centers = (x_centers / self.encoder_w - 0.5) * 2.0
        y_centers = (y_centers / self.encoder_h - 0.5) * 2.0
        xy_centers = torch.stack((x_centers, y_centers), dim=-1).reshape(1, 1, n_masks, 2)

        tar_feat_2d = tar_feat.reshape(1, self.encoder_h, self.encoder_w, self.encoder_dim).permute(0, 3, 1, 2)
        sampled_feats = F.grid_sample(tar_feat_2d, xy_centers, mode='bilinear').reshape(self.encoder_dim, n_masks).t()
        sampled_feats = F.normalize(sampled_feats, p=2, dim=-1)

        mem_feats_avg = self.mem_feats_avg  # [n_class, c]
        mem_feats_avg = F.normalize(mem_feats_avg, p=2, dim=-1)
        sim_avg = sampled_feats @ mem_feats_avg.t()  # [n_masks, n_class]
        sim_avg = (sim_avg + 1.0) * 0.5
        return sim_avg

    def _compute_sim_matching(self, tar_feat, masks_feat_size_bool):
        n_masks = masks_feat_size_bool.shape[0]

        tar_avg_feats_all = []
        tar_foreground_feats = []
        for i in range(n_masks):
            feats_i = tar_feat[masks_feat_size_bool[i]]
            tar_avg_feats_all.append(feats_i.mean(dim=0, keepdim=True))
            tar_foreground_feats.append(feats_i)

        tar_avg_feats = torch.cat(tar_avg_feats_all, dim=0)
        tar_avg_feats = F.normalize(tar_avg_feats, p=2, dim=-1)

        mem_feats_avg = self.mem_feats_avg
        mem_feats_avg = F.normalize(mem_feats_avg, p=2, dim=-1)
        mem_feats_centers = self.mem_feats_centers  # already normed
        mem_feats_centers = mem_feats_centers.reshape(-1, self.encoder_dim)  # [n_class * n_centers, c]

        sim_global = tar_avg_feats @ mem_feats_avg.t()
        sim_global = (sim_global + 1.0) * 0.5

        sims_matching = []
        for i in range(n_masks):
            tar_feats_i = tar_foreground_feats[i]  # [n_fore, c]
            _norm = float(tar_feats_i.shape[0]) * 10.0  # to avoid over float

            tar_mag_i = torch.sqrt(torch.pow((tar_feats_i / _norm).sum(dim=0), 2).sum())  # [1,]

            sim = tar_feats_i @ mem_feats_centers.t()  # [n_fore, n_classes * n_centers]
            sim = sim.reshape(-1, self.mem_n_classes, self.kmeans_k)
            sim = sim.max(dim=-1)[0]  # [n_fore, n_classes]
            sim = (sim / _norm).sum(dim=0, keepdim=True)  # [1, n_classes]
            sim = sim / tar_mag_i
            sim = (sim + 1.0) * 0.5

            # sim = tar_feats_i @ mem_feats_avg.t()   # [n_fore, n_classes]
            # sim = (sim / _norm).sum(dim=0, keepdim=True)  # [1, n_classes]
            # sim = sim / tar_mag_i
            # sim = (sim + 1.0) * 0.5

            sims_matching.append(sim)
        sim_matching = torch.cat(sims_matching, dim=0)
        r = 0.0
        similarity = sim_global * r + sim_matching * (1.0 - r)
        return similarity

    def _compute_sim_knn(self, tar_feat, masks_feat_size_bool, k=5, sigma=0.01):
        device = tar_feat.device
        n_masks = masks_feat_size_bool.shape[0]

        tar_avg_feats_all = []

        for i in range(n_masks):
            feats_i = tar_feat[masks_feat_size_bool[i]]
            tar_avg_feats_all.append(feats_i.mean(dim=0, keepdim=True))

        tar_avg_feats = torch.cat(tar_avg_feats_all, dim=0)
        tar_avg_feats = F.normalize(tar_avg_feats, p=2, dim=-1)
        #
        # mem_ins_feats_avg = (
        #     torch.sum(self.mem_feats * self.mem_masks.unsqueeze(dim=-1), dim=2)
        #     / self.mem_masks.sum(dim=2).unsqueeze(dim=2)
        # )
        mem_ins_feats_avg = F.normalize(self.mem_feats_ins_avg.flatten(0, 1), p=2, dim=-1)

        sim = tar_avg_feats @ mem_ins_feats_avg.t()  # [n_mask, n_class * n_ins]
        dis = 1.0 - sim
        sim = (sim + 1.0) * 0.5

        # sim = sim.reshape(n_masks, self.mem_n_classes, self.mem_length).mean(dim=-1)
        # scores, labels = torch.topk(sim, 1)
        # scores = scores.flatten()
        # labels = labels.flatten()

        top_sim, top_inds = torch.topk(sim, k=k, dim=-1)   # [n_mask, k]
        top_class_inds = top_inds // self.mem_length  # [n_mask, k]
        if k == 1:
            return top_sim.flatten(), top_class_inds.flatten()

        labels = torch.zeros((n_masks), dtype=torch.long, device=device)
        scores = torch.zeros((n_masks), dtype=torch.float32, device=device)
        for i in range(n_masks):
            counts = torch.bincount(top_class_inds[i], minlength=self.mem_n_classes)
            _label = torch.argmax(counts)
            _hit_inds = top_class_inds[i] == _label
            # _score = top_sim[i][top_class_inds[i] == _label].mean()
            ws = torch.exp(-1 * torch.pow(dis[i][top_inds[i]][_hit_inds], 2) / sigma**2)
            _score = (top_sim[i][_hit_inds] * ws).sum() / ws.sum()
            labels[i] += _label
            scores[i] += _score

        return scores, labels

    def _compute_sim_covariance_cosine(self, tar_feat, masks_feat_size_bool):
        device = tar_feat.device

        n_masks = masks_feat_size_bool.shape[0]
        n_classes = self.mem_n_classes
        c = tar_feat.shape[-1]

        masks = masks_feat_size_bool.to(dtype=tar_feat.dtype)
        x = (masks @ tar_feat) / masks.sum(dim=-1, keepdim=True)  # [n_masks, c]
        x = x.unsqueeze(dim=1)  # [n_masks, 1, c]

        t = self.mem_feats_ins_avg.mean(dim=1)  # [n_class, c]

        sigma = self.mem_feats_covariances
        lamda_s = 0.001
        sigma = (1.0 - lamda_s) * sigma + lamda_s * torch.eye(c, device=device).unsqueeze(dim=0)

        invert_sigma = torch.linalg.inv(sigma)  # [n_class, c, c]

        sim_classes = []
        for i in range(n_classes):
            x_norm = torch.sqrt(x @ invert_sigma[i:i+1] @ x.transpose(1, 2)).reshape(n_masks, 1)
            t_norm = torch.sqrt(t[i:i+1] @ invert_sigma[i] @ t[i:i+1].t()).reshape(1)
            sim = (x @ invert_sigma[i:i+1] @ t[i].reshape(1, c, 1)).reshape(n_masks, 1)
            sim = sim / (x_norm * t_norm)
            sim_classes.append(sim)
        scores = torch.cat(sim_classes, dim=1)
        scores = scores.clamp(min=0.0)
        return scores

    def _compute_sim_intra_class_norm(self, tar_feat, masks_feat_size_bool, margin=0.1, sigma=1.0):
        masks = masks_feat_size_bool.to(dtype=tar_feat.dtype)
        x = (masks @ tar_feat) / masks.sum(dim=-1, keepdim=True)  # [n_masks, c]
        x = x.unsqueeze(dim=1)  # [n_masks, c]
        x_norm = F.normalize(x, p=2, dim=-1)

        t_ins = self.mem_feats_ins_avg
        t_ins_norm = F.normalize(t_ins, p=2, dim=-1)
        intra_class_sim = t_ins_norm @ t_ins_norm.transpose(-1, -2)
        intra_class_sim = intra_class_sim.mean(dim=(1, 2))
        intra_class_sim = intra_class_sim.unsqueeze(dim=0)

        t = t_ins.mean(dim=1)  # [n_class, c]
        t_norm = F.normalize(t, p=2, dim=-1)

        sim = x_norm @ t_norm.t()
        sim = sim.clamp(min=0.0)
        sim = torch.where(
            sim > intra_class_sim - margin,
            sim,
            sim / torch.pow(intra_class_sim, sigma)
        )
        return sim

    def _compute_sim_covariance_diag_cosine(self, tar_feat, masks_feat_size_bool):
        device = tar_feat.device

        n_masks = masks_feat_size_bool.shape[0]
        n_classes = self.mem_n_classes
        c = tar_feat.shape[-1]

        masks = masks_feat_size_bool.to(dtype=tar_feat.dtype)
        x = (masks @ tar_feat) / masks.sum(dim=-1, keepdim=True)  # [n_masks, c]

        # t = self.mem_feats_ins_avg.mean(dim=1)  # [n_class, c]
        t = self.mem_feats_avg

        _inds = torch.arange(c, device=device)
        diag_cov = self.mem_feats_covariances[:, _inds, _inds]  # [n_class, c]
        t_scaled = t / torch.sqrt(diag_cov)
        t_scaled = F.normalize(t_scaled, p=2, dim=-1)

        sim_classes = []
        for i in range(n_classes):
            x_scaled = x / diag_cov[i:i+1]
            x_scaled = F.normalize(x_scaled, p=2, dim=-1)
            sim = x_scaled @ t_scaled[i:i+1].t()
            sim_classes.append(sim)

        scores = torch.cat(sim_classes, dim=1)
        scores = scores.clamp(min=0.0)
        return scores

    def _compute_sim_vMF(self, tar_feat, masks_feat_size_bool):
        n_masks = masks_feat_size_bool.shape[0]
        n_classes = self.mem_n_classes
        c = tar_feat.shape[-1]

        masks = masks_feat_size_bool.to(dtype=tar_feat.dtype)
        x = (masks @ tar_feat) / masks.sum(dim=-1, keepdim=True)  # [n_masks, c]
        mu = self.mem_feats_ins_avg.mean(dim=1)  # [n_class, c]
        r = self.mem_feats_ins_avg.sum(dim=1).norm(p=2, dim=-1) / self.mem_feats_ins_avg.shape[1]
        kappa = (r * (c - r ** 2)) / (1 - r ** 2)

    def _compute_sim_matching_soft(self, tar_feat, masks_feat_size_bool):
        n_masks = masks_feat_size_bool.shape[0]

        tar_avg_feats_all = []
        tar_foreground_feats = []
        for i in range(n_masks):
            feats_i = tar_feat[masks_feat_size_bool[i]]
            tar_avg_feats_all.append(feats_i.mean(dim=0, keepdim=True))
            tar_foreground_feats.append(feats_i)

        mem_feats_centers = self.mem_feats_centers  # already normed
        mem_feats_centers = mem_feats_centers.reshape(-1, self.encoder_dim)  # [n_class * n_centers, c]

        sims_matching = []
        for i in range(n_masks):
            tar_feats_i = tar_foreground_feats[i]  # [n_fore, c]
            _norm = float(tar_feats_i.shape[0]) * 10.0  # to avoid over float

            tar_mag_i = torch.sqrt(torch.pow((tar_feats_i / _norm).sum(dim=0), 2).sum())  # [1,]

            sim = tar_feats_i @ mem_feats_centers.t()  # [n_fore, n_classes * n_centers]
            sim = sim.reshape(-1, self.mem_n_classes, self.kmeans_k)

            theta = 1.0
            w = torch.exp(sim/theta)
            sim = (sim * w).sum(dim=-1) / w.sum(dim=-1)

            # sim = sim.max(dim=-1)[0]  # [n_fore, n_classes]
            sim = (sim / _norm).sum(dim=0, keepdim=True)  # [1, n_classes]
            sim = sim / tar_mag_i
            sim = (sim + 1.0) * 0.5

            # sim = tar_feats_i @ mem_feats_avg.t()   # [n_fore, n_classes]
            # sim = (sim / _norm).sum(dim=0, keepdim=True)  # [1, n_classes]
            # sim = sim / tar_mag_i
            # sim = (sim + 1.0) * 0.5

            sims_matching.append(sim)
        sim_matching = torch.cat(sims_matching, dim=0)
        return sim_matching

    def forward_test(self, input_dicts, with_negative):

        if PRINT_TIMING:
            start_time = time.time()

        assert len(input_dicts) == 1

        device = self.predictor.device

        tar_img = input_dicts[0]["target_img"].to(device=device)
        sam_input_size = tar_img.shape[-2]
        tar_img_encoder = F.interpolate(
            tar_img.unsqueeze(dim=0),
            size=(self.encoder_img_size, self.encoder_img_size),
            mode="bicubic"
        )
        tar_feat, last_attn = self._forward_encoder_attn_roll(tar_img_encoder)
        tar_feat = tar_feat.reshape(-1, self.encoder_dim)  # [N, C]

        # ----------------------------------------------------------------------------------------
        # SAM inference
        tar_img = tar_img.unsqueeze(dim=0)

        # Method 1: first matching then SAM
        # match_size = (37, 37)
        # num_points = min(match_size[0] * match_size[1], 1024)
        # precomputed_points = self._compute_query_points(tar_feat, match_size, num_points)
        # lr_masks, pred_ious, query_points = self._forward_sam(
        #     self.sam_transform(tar_img), precomputed_points, point_normed=True
        # )

        # Method 2: Normal inference
        lr_masks, pred_ious, query_points = self._forward_sam(self.sam_transform(tar_img))

        # Method 3: Multi-scale inference
        # lr_masks, pred_ious, query_points = self._forward_sam_multiscale(tar_img, scales=(0.7, 0.8, 0.9, 1.0))
        # ----------------------------------------------------------------------------------------

        n_masks = lr_masks.shape[0]
        masks_feat_size_bool = lr_masks > 0
        masks_feat_size_bool = masks_feat_size_bool.reshape(n_masks, -1)
        tar_feat = tar_feat.reshape(1, self.encoder_h, self.encoder_w, -1).permute(0, 3, 1, 2)
        tar_feat = F.interpolate(
            tar_feat,
            size=tuple(lr_masks.shape[-2:]),
            mode="bilinear",
            align_corners=False,
            antialias=True
        ).reshape(-1, lr_masks.shape[-2] * lr_masks.shape[-1]).t()

        if not with_negative:
            # sim_local = self._compute_sim_matching(tar_feat, masks_feat_size_bool)
            # pca_scores = self._compute_pca_scores(tar_feat, masks_feat_size_bool)
            # sim_global = self._compute_sim_center(tar_feat, masks_feat_size_bool)
            if PRINT_TIMING:
                start_time_sim_global = time.time()
            sim_global, obj_feats = self._compute_sim_global_avg(tar_feat, masks_feat_size_bool, ret_feats=True)
            if PRINT_TIMING:
                end_time_sim_global = time.time()
                print("--------------------------------")
                print("TIMING SIM GLOBAL: ", end_time_sim_global - start_time_sim_global)
                print("--------------------------------")
            # sim_global = self._compute_sim_covariance_diag_cosine(tar_feat, masks_feat_size_bool)
            # sim_global = self._compute_sim_vMF(tar_feat, masks_feat_size_bool)
            # sim_global = self._compute_sim_intra_class_norm(tar_feat, masks_feat_size_bool, margin=0.2, sigma=0.2)
            # sim_global = self._compute_sim_covariance_cosine(tar_feat, masks_feat_size_bool)
            # sim_global = self._compute_sim_global_avg(tar_feat, masks_feat_size_bool, softmax=True, temp=0.5)
            # sim_global = self._compute_sim_instance_softmax(tar_feat, masks_feat_size_bool, temp=0.75)
            # sim_global = self._compute_sim_attn_guided_global_avg(tar_feat, last_attn, masks_feat_size_bool)
        else:
            assert self.with_negative_refs
            assert self.memory_neg_ready
            # sim_global = self._compute_sim_attn_guided_global_weighted(tar_feat, last_attn, masks_feat_size_bool)
            # sim_global = self._compute_sim_attn_guided_global_avg(tar_feat, last_attn, masks_feat_size_bool)
            # sim_global = self._compute_sim_global_avg(tar_feat, masks_feat_size_bool)
            sim_global = self._compute_sim_global_avg_with_neg(tar_feat, masks_feat_size_bool, sigma=0.8)
            # sim_global = self._compute_sim_global_l2(tar_feat, masks_feat_size_bool, sigma=30.0)
            # sim_global = self._compute_sim_covariance_cosine(tar_feat, masks_feat_size_bool)
            # sim_global = self._compute_sim_gaussian(tar_feat, masks_feat_size_bool)

        merged_scores = sim_global  #  * torch.pow(pca_scores, 0.25)
        
        # SEM enhancement (multi-scale semantic matching)
        sem_cache = {}
        if self.sem_enabled and self.sem_matcher:
            print("=" * 60)
            print("🎯 SEM ENABLED - Multi-scale semantic matching")
            print(f"   Scales: {self.sem_cfg.get('scales')}")
            print(f"   Alpha: {self.sem_cfg.get('alpha')}")
            print(f"   Method: {self.sem_cfg.get('method')}")
            print(f"   Cache pyramid: {self.sem_cfg.get('cache_pyr')}")
            print(f"   Score dtype: {self.sem_cfg.get('score_dtype')}")
            print("=" * 60)
            
            # 重塑特征用于多尺度匹配
            tar_feat_2d = tar_feat.t().reshape(1, self.encoder_dim, lr_masks.shape[-2], lr_masks.shape[-1])
            
            # 为每个类别构建原型特征
            proto_feats = []
            for cat_idx in range(self.mem_n_classes):
                mem_feat = self.mem_feats_avg[cat_idx].view(1, self.encoder_dim, 1, 1)
                proto_feats.append(mem_feat.squeeze(0))
            
            # 执行 SEM 匹配
            if proto_feats:
                match_results, aux_info = self.sem_matcher.match(proto_feats, tar_feat_2d.squeeze(0))
                
                # 提取相似度增强因子
                sem_boost = torch.ones_like(merged_scores)
                for cat_idx, result in enumerate(match_results):
                    if result.similarity_scores.numel() > 0:
                        # 检查并处理 NaN
                        sim_scores = result.similarity_scores
                        if torch.isnan(sim_scores).any():
                            print(f"   ⚠️  Warning: NaN detected in similarity_scores for category {cat_idx}, skipping boost")
                            continue
                        boost_factor = sim_scores.mean().item()
                        if not torch.isfinite(torch.tensor(boost_factor)):
                            print(f"   ⚠️  Warning: Non-finite boost_factor for category {cat_idx}, skipping boost")
                            continue
                        sem_boost[:, cat_idx] = sem_boost[:, cat_idx] * (1.0 + boost_factor)
                
                # 应用 SEM 增强（检查 NaN）
                if torch.isnan(merged_scores).any():
                    print(f"   ⚠️  Warning: NaN detected in merged_scores before SEM boost, resetting to 1.0")
                    merged_scores = torch.where(torch.isnan(merged_scores), torch.ones_like(merged_scores), merged_scores)
                
                merged_scores = merged_scores * sem_boost
                
                # 检查增强后是否有 NaN
                if torch.isnan(merged_scores).any():
                    print(f"   ⚠️  Warning: NaN detected after SEM boost, replacing with original scores")
                    merged_scores = torch.where(torch.isnan(merged_scores), torch.ones_like(merged_scores), merged_scores)
                
                # 缓存调试信息
                if match_results:
                    sem_cache["match_results"] = match_results
                    sem_cache["aux_info"] = aux_info
                    if match_results[0].scale_weights is not None:
                        sem_cache["scale_weights"] = match_results[0].scale_weights
                    # 构建聚合的相似度图
                    all_sim_scores = torch.stack([r.similarity_scores for r in match_results if r.similarity_scores.numel() > 0])
                    sem_cache["similarity_map"] = all_sim_scores.mean(dim=0) if all_sim_scores.numel() > 0 else torch.tensor([])
                
                # 打印调试信息
                if aux_info.get("alpha_weights") is not None:
                    print(f"   α weights (scales): {aux_info['alpha_weights'].tolist()}")
                if aux_info.get("beta_weights") is not None:
                    print(f"   β weights (layers): {aux_info['beta_weights'].tolist()}")
                if aux_info.get("gamma_auto") is not None:
                    print(f"   γ (auto-adjusted): {aux_info['gamma_auto']:.3f}")
                
                score_min = merged_scores.min().item()
                score_max = merged_scores.max().item()
                if torch.isnan(merged_scores).any():
                    print(f"   SEM boost applied: score range [NaN detected, replacing with 1.0]")
                else:
                    print(f"   SEM boost applied: score range [{score_min:.3f}, {score_max:.3f}]")

        if self.cls_num_per_mask == -1:
            self.cls_num_per_mask = self.mem_n_classes
        top_scores, labels = torch.topk(merged_scores, k=self.cls_num_per_mask)

        if self.cls_num_per_mask == self.mem_n_classes:
            max_scores = top_scores[:, 0:1]
            top_scores = top_scores * (top_scores > (max_scores * 0.6))

        labels = labels.flatten()
        scores_all_class = top_scores.flatten()

        # scores_thr_by_classes = self.mem_ins_sim_avg[labels] * 0.8
        # scores_all_class[scores_all_class < scores_thr_by_classes] = 0.0

        # scores_all_class, labels = self._compute_sim_knn(tar_feat, masks_feat_size_bool, k=7, sigma=0.1)

        # assert self.cls_num_per_mask == 1
        # ambiguous_decay = self._compute_ambiguous_decay(sim_global, labels)
        # scores_all_class = scores_all_class * ambiguous_decay

        # ----------------------------------------------------------------------------------------
        # Local-global similarity analysis
        # n_masks = masks_feat_size_bool.shape[0]
        # local_global_mean = []
        # local_global_std = []
        # for i in range(n_masks):
        #     _feats_fore = tar_feat[masks_feat_size_bool[i]]
        #     _feats_avg = _feats_fore.mean(dim=0, keepdim=True)
        #     _sim = F.normalize(_feats_fore, p=2, dim=-1) @ F.normalize(_feats_avg, p=2, dim=-1).t()
        #     _sim = (_sim + 1.0) * 0.5
        #     local_global_std.append(torch.std(_sim))
        #     local_global_mean.append(torch.mean(_sim))
        # local_global_mean = torch.stack(local_global_mean).flatten()
        # local_global_std = torch.stack(local_global_std).flatten()
        # ----------------------------------------------------------------------------------------

        # ----------------------------------------------------------------------------------------
        # Oracle Analysis
        # scores_oracle = self._get_oracle_iou(lr_masks, input_dicts[0]["tar_anns_by_cat"]).t()
        # assert self.cls_num_per_mask == 1
        #
        # top_scores, labels = torch.topk(scores_oracle, k=self.cls_num_per_mask)
        # labels = labels.flatten()
        # scores_all_class = top_scores.flatten()

        #
        # select_sim_global = sim_global[torch.arange(scores_oracle.shape[0], device=device), labels].flatten()
        # select_scores_oracle = scores_oracle[torch.arange(scores_oracle.shape[0], device=device), labels].flatten()
        # select_labels = labels.flatten()
        # selected_ins_sims = self.mem_ins_sim_avg[labels].flatten()
        # select_pca_scores = pca_scores[torch.arange(scores_oracle.shape[0], device=device), labels].flatten()

        # top_scores_oracle, oracle_labels = torch.topk(scores_oracle, k=self.cls_num_per_mask)
        # oracle_labels = oracle_labels.flatten()
        # top_scores_oracle = top_scores_oracle.flatten()
        # ----------------------------------------------------------------------------------------

        lr_bboxes = batched_mask_to_box(lr_masks > 0)
        lr_bboxes_expand = (
            lr_bboxes.unsqueeze(dim=1)
            .expand(-1, self.cls_num_per_mask, -1)
            .reshape(n_masks * self.cls_num_per_mask, 4)
        )

        expand_ratio = 8
        out_num = int(min(self.num_out_instance * expand_ratio, labels.shape[0]))

        nms_keep_inds = batched_nms(
            lr_bboxes_expand.float(),
            # scores_all_class,
            pred_ious.flatten(),
            labels,
            iou_threshold=self.nms_thr
        )[:out_num]
        scores_out = scores_all_class[nms_keep_inds]
        pred_ious_out = pred_ious[nms_keep_inds]
        lr_masks_out = lr_masks[nms_keep_inds // self.cls_num_per_mask]
        obj_feats_out = obj_feats[nms_keep_inds // self.cls_num_per_mask]
        masks_feat_size_bool = masks_feat_size_bool[nms_keep_inds // self.cls_num_per_mask]
        labels_out = labels[nms_keep_inds]

        pos_inds = scores_out > 0.0
        scores_out = scores_out[pos_inds]
        lr_masks_out = lr_masks_out[pos_inds]
        obj_feats_out = obj_feats_out[pos_inds]
        masks_feat_size_bool = masks_feat_size_bool[pos_inds]
        labels_out = labels_out[pos_inds]

        # UAM preparation (before merging)
        uam_cache = None
        if self.uam_enabled:
            print("=" * 60)
            print("🔥 UAM ENABLED - Starting uncertainty-aware merging")
            print(f"   Input masks: {lr_masks_out.shape[0]}")
            print(f"   Threshold: {self.uam_cfg.get('threshold')}")
            print(f"   Temperature: {self.uam_cfg.get('temperature')}")
            print(f"   CRF enabled: {self.uam_cfg.get('crf', {}).get('enable')}")
            print(f"   Debug save_probs: {self._uam_debug.get('save_probs')}")
            print(f"   Debug save_masks: {self._uam_debug.get('save_masks')}")
            print(f"   Debug out_dir: {self._uam_debug.get('out_dir')}")
            print("=" * 60)
            uam_cache = self._uam_prepare_scores(
                lr_masks_out,
                scores_out,
                labels_out,
                input_dicts[0],
            )

        # query_points = query_points[keep_inds // self.cls_num_per_mask]

        # ----------------------------------------------------------------------------------------
        # Iteratively Refine
        # refine_points = query_points.reshape(self.num_out_instance, 1, 2)
        # refine_point_labels = torch.ones_like(refine_points[:, :, 0], dtype=torch.int32).reshape(-1, 1)

        # refine_points, refine_labels, do_refine = self._get_oracle_refine_prompts(
        #     lr_masks_out, labels_out, input_dicts[0]["tar_anns_by_cat"], pool_size=5
        # )

        # refine_points, refine_labels = self._get_refine_prompts(
        #     tar_feat, lr_masks_out, labels_out, pool_size=5, thr=0.6
        # )
        # do_refine = torch.ones((lr_masks_out.shape[0],), device=device)
        #
        # do_refine = do_refine > 0
        # if do_refine.sum() > 0:
        #     refine_point_inputs = dict(
        #         point_coords=refine_points[do_refine],
        #         point_labels=refine_labels[do_refine].to(dtype=torch.int32)
        #     )
        #     lr_masks_refine, _ = self._compute_masks_refine(
        #         point_inputs=refine_point_inputs, boxes_inputs=None, mask_inputs=lr_masks_out[do_refine]
        #     )
        #     lr_masks_out[do_refine] = lr_masks_refine
        # lr_masks_out = lr_masks_out.reshape(self.num_out_instance, *lr_masks_out.shape[-2:])
        # ----------------------------------------------------------------------------------------

        # ----------------------------------------------------------------------------------------
        # Other decay

        # score_decay = 1.0 - self._compute_ios(lr_masks_out>0, labels_out, rank_score=True)
        # scores_out = scores_out * torch.pow(score_decay, 0.1)

        # compelteness = self._compute_completeness_decay(tar_feat, masks_feat_size_bool, scores_out, labels_out, decay=0.4)
        # scores_out = scores_out * torch.pow(compelteness, 0.2)

        # unificationess = self._compute_unification_decay(tar_feat, masks_feat_size_bool)
        # scores_out = scores_out * torch.pow(unificationess, 0.1)

        # sim_neg = self._compute_negative_decay(tar_feat, masks_feat_size_bool, scores_out, labels_out)
        # s_square = 1.0
        # scores_out = scores_out * torch.exp(-1.0 * (sim_neg - scores_out).clamp(min=0.0) / s_square)
        # ----------------------------------------------------------------------------------------

        # resizing and converting to output format
        ori_h = input_dicts[0]["target_img_info"]["ori_height"]
        ori_w = input_dicts[0]["target_img_info"]["ori_width"]

        
        if lr_masks_out.shape[0] == 0:
            raise ValueError("No masks found")
            # self._reset()
            # return [{
            #     "binary_masks": torch.zeros((0, ori_h, ori_w), device=device).bool(),
            #     "bboxes": torch.zeros((0, 4), device=device),
            #     "scores": torch.zeros((0,), device=device),
            #     "labels": torch.zeros((0,), dtype=torch.long, device=device),
            #     "image_info": input_dicts[0]["target_img_info"],
            # }]

        masks_out_binary = F.interpolate(
            lr_masks_out.unsqueeze(dim=1),
            size=(ori_h, ori_w),
            mode="bilinear",
            align_corners=False,
            antialias=True
        ).squeeze(dim=1) > 0

        bboxes = batched_mask_to_box(masks_out_binary)

        # ----------------------------------------------------------------------------------------
        # Merging Masks

        # use_batches = False # This should be False for better results
        # if use_batches:
        #     batch_size = 10
        #     n_masks = masks_out_binary.shape[0]
        #     all_score_decays = []
        #     for i in range(0, n_masks, batch_size):
        #         batch_end = min(i + batch_size, n_masks)
        #         batch_masks = masks_out_binary[i:batch_end]
        #         batch_labels = labels_out[i:batch_end]
        #         batch_score_decay = 1.0 - self._compute_ios_batched(
        #             batch_masks, 
        #             batch_labels,
        #             rank_score=True,
        #             batch_size=batch_size
        #         )
        #         all_score_decays.append(batch_score_decay)
        #     score_decay = torch.cat(all_score_decays)
        # else:
        #     obj_sim = obj_feats_out @ obj_feats_out.t()
        #     obj_sim = obj_sim.clamp(min=0.0)
        #     ios = self._compute_semantic_ios(masks_out_binary, labels_out, obj_sim, use_semantic=True, rank_score=True)
        #     score_decay = 1 - ios
        #     # # # # # Old version
        #     # # # # score_decay = 1.0 - self._compute_ios(masks_out_binary, labels_out, rank_score=True)


        # Method 1: Soft merging
        if PRINT_TIMING:
            start_time_merging = time.time()
        obj_sim = obj_feats_out @ obj_feats_out.t()
        obj_sim = obj_sim.clamp(min=0.0)
        ios = self._compute_semantic_ios(masks_out_binary, labels_out, obj_sim, use_semantic=True, rank_score=True)
        score_decay = 1 - ios
        scores_out = scores_out * torch.pow(score_decay, 0.5)
        
        # Apply UAM merging if enabled
        if self.uam_enabled and uam_cache is not None:
            print("🔥 Applying UAM merge...")
            scores_out, lr_masks_out, labels_out, pred_ious_out = self._uam_merge(
                scores_out, lr_masks_out, labels_out, pred_ious_out, uam_cache
            )
            prior_maps = uam_cache.get("prior_maps", {})
            if prior_maps:
                prior_desc = ", ".join(prior_maps.keys())
                print(f"   UAM priors used: {prior_desc}")
            print(f"   After UAM: {lr_masks_out.shape[0]} masks retained")
        
        if PRINT_TIMING:
            end_time_merging = time.time()
            print("--------------------------------")
            print("TIMING MERGING: ", end_time_merging - start_time_merging)
            print("--------------------------------")

        # Method 2: Hard merging
        # obj_sim = obj_feats_out @ obj_feats_out.t()
        # obj_sim = obj_sim.clamp(min=0.0)
        # ios = self._compute_semantic_ios(masks_out_binary, labels_out, obj_sim, use_semantic=False, rank_score=True)
        # keep_inds = ios < 1.0

        # scores_out = scores_out[keep_inds]
        # masks_out_binary = masks_out_binary[keep_inds]
        # bboxes = bboxes[keep_inds]
        # labels_out = labels_out[keep_inds]
        # ----------------------------------------------------------------------------------------

        final_out_num = min(self.num_out_instance, scores_out.shape[0])
        final_out_inds = torch.argsort(scores_out, descending=True)[:final_out_num]

        pred_ious_out = pred_ious_out[final_out_inds]
        masks_out_binary = masks_out_binary[final_out_inds]
        bboxes = bboxes[final_out_inds]
        scores_out = scores_out[final_out_inds]
        labels_out = labels_out[final_out_inds]

        # score_to_analysis = torch.stack(
        #     (
        #         select_sim_global[nms_keep_inds][pos_inds][final_out_inds],
        #         select_labels[nms_keep_inds][pos_inds][final_out_inds],
        #         select_scores_oracle[nms_keep_inds][pos_inds][final_out_inds]
        #     ),
        #     dim=-1
        # )

        output_dict = dict(
            binary_masks=masks_out_binary,
            bboxes=bboxes,
            scores=scores_out,
            labels=labels_out,
            image_info=input_dicts[0]["target_img_info"],
        )

        image_info_full = input_dicts[0]["target_img_info"]
        stem = Path(image_info_full.get("file_name", f"img_{image_info_full.get('id', 'unknown')}")).stem
        export_root = Path("./results_analysis") / (self.dataset_name or "custom")

        output_export_path_raw = image_info_full.get("export_path")
        per_instance_dir_raw = image_info_full.get("export_instances_dir")
        json_path_raw = image_info_full.get("export_json_path")

        output_export_path_path = Path(output_export_path_raw) if output_export_path_raw else export_root / "predictions" / f"{stem}_prediction.png"
        per_instance_dir_path = Path(per_instance_dir_raw) if per_instance_dir_raw else export_root / "instances" / stem
        json_path_path = Path(json_path_raw) if json_path_raw else export_root / "json" / f"{stem}_prediction.json"
        binary_masks_dir_path = export_root / "binary_masks" / stem  # 新增：纯二值掩码目录

        export_paths_path = {
            "image": output_export_path_path,
            "json": json_path_path,
            "instances": per_instance_dir_path,
            "binary_masks": binary_masks_dir_path,  # 新增
        }

        export_payload = self._build_export_payload(output_dict)
        
        # 获取原始图像路径用于正确比例的可视化
        img_path = self._get_image_path(input_dicts[0]["target_img_info"])
        
        # 打印导出开始信息
        n_total = len(export_payload['instances'])
        n_filtered = sum(1 for ins in export_payload['instances'] if ins['score'] >= self.vis_thr)
        print("=" * 60)
        print(f"📦 开始导出预测结果 (vis_thr={self.vis_thr})")
        print(f"   检测到 {n_total} 个实例，{n_filtered} 个满足阈值")
        
        saved_paths = self._save_export_outputs(
            export_payload,
            export_paths_path,
            img_path,
            input_dicts[0]["tar_anns_by_cat"],
            stem,
        )
        output_dict["export_paths"] = saved_paths
        
        # 打印导出完成信息
        print(f"   ✓ 最终可视化: {saved_paths['image']}")
        print(f"   ✓ JSON 结果: {saved_paths['json']} (完整数据: {n_total} 实例)")
        print(f"   ✓ 实例图像: {saved_paths['instances']} (已保存: {n_filtered} 实例)")
        print(f"   ✓ 二值掩码: {saved_paths['binary_masks']} (已保存: {n_filtered} 个掩码)")
        print("=" * 60)

        # UAM debug dump
        if self.uam_enabled and uam_cache is not None:
            print("🔥 Dumping UAM debug artifacts...")
            self._uam_debug_dump(uam_cache, input_dicts[0]["target_img_info"], masks_out_binary)
            print(f"   Debug files saved to: {self._uam_debug.get('out_dir')}")

        # SEM debug dump
        if self.sem_enabled and sem_cache:
            print("🎯 Dumping SEM debug artifacts...")
            self._sem_debug_dump(sem_cache, input_dicts[0]["target_img_info"])
            print(f"   Debug files saved to: {self._sem_debug.get('out_dir')}")

        if self.eval_out_format in {"obb", "polygon"}:
            output_dict["export_format"] = self.eval_out_format

        if self.online_vis:
            self._vis_results_online(
                output_dict,
                input_dicts[0]["tar_anns_by_cat"],
                score_thr=self.vis_thr,
                show_scores=True,
                dataset_name=self.dataset_name,
                dataset_imgs_path=self.dataset_imgs_path,
                class_names=self.class_names,
            )
        self._reset()

        if PRINT_TIMING:
            end_time = time.time()
            total_time = end_time - start_time
            print(f"\n===== TIMING FORWARD TEST RESULTS =====")
            print(f"Total processing time: {total_time:.2f} seconds")
            print(f"===========================\n")

        return [output_dict]

    def _build_export_payload(self, output_dict: Dict[str, Any]) -> Dict[str, Any]:
        """构建导出所需的中间数据结构。

        该方法负责：
        - 将张量转换为 CPU numpy 形式，便于后续序列化；
        - 计算 COCO JSON 所需的 bbox (xywh)、mask RLE；
        - 保存基础元数据（分数、标签、类别 id）。

        参数:
            output_dict: `forward_test` 构建的输出结果。

        返回:
            包含 `instances` 列表的字典，每个元素拥有 `mask`、`score`、`label`、
            `bbox_xyxy`、`bbox_xywh`、`segmentation`、`category_id` 等字段。
        """

        binary_masks = output_dict["binary_masks"].detach().cpu()
        bboxes = output_dict["bboxes"].detach().cpu()
        scores = output_dict["scores"].detach().cpu()
        labels = output_dict["labels"].detach().cpu()
        image_info = output_dict["image_info"]
        
        # 归一化 scores 到 [0, 1] 范围（处理 SEM 增强后可能超过 1 的情况）
        if scores.numel() > 0:
            # 首先处理 NaN：将 NaN 替换为 0.0
            nan_mask = torch.isnan(scores)
            nan_count = nan_mask.sum().item()
            if nan_count > 0:
                print(f"   ⚠️  Warning: {nan_count} NaN scores detected, replacing with 0.0")
                scores = torch.where(nan_mask, torch.zeros_like(scores), scores)
            
            # 处理 Inf：将 Inf 替换为 1.0
            inf_mask = torch.isinf(scores)
            inf_count = inf_mask.sum().item()
            if inf_count > 0:
                print(f"   ⚠️  Warning: {inf_count} Inf scores detected, replacing with 1.0")
                scores = torch.where(inf_mask, torch.ones_like(scores), scores)
            
            score_min = scores.min().item()
            score_max = scores.max().item()
            if score_max > 1.0 or score_min < 0.0:
                print(f"   [Score Normalization] Before: min={score_min:.3f}, max={score_max:.3f}")
                # Min-Max 归一化到 [0, 1]
                if score_max > score_min:
                    scores = (scores - score_min) / (score_max - score_min)
                else:
                    scores = scores.clamp(0.0, 1.0)
                print(f"   [Score Normalization] After: min={scores.min().item():.3f}, max={scores.max().item():.3f}")
            else:
                # 确保在 [0, 1] 范围内
                scores = scores.clamp(0.0, 1.0)
            
            # 最终检查：确保没有 NaN 或 Inf
            if torch.isnan(scores).any() or torch.isinf(scores).any():
                print(f"   ⚠️  Warning: NaN/Inf still present after normalization, replacing with 0.0")
                scores = torch.where(torch.isnan(scores) | torch.isinf(scores), torch.zeros_like(scores), scores)

        instances: List[Dict[str, Any]] = []
        for idx in range(scores.shape[0]):
            mask_bool = binary_masks[idx] > 0
            mask_np = mask_bool.numpy().astype(np.uint8)
            bbox_xyxy = bboxes[idx].numpy().astype(float)
            score_val = float(scores[idx].item())
            label_val = int(labels[idx].item())

            bbox_xywh = self._mask_to_xywh(bbox_xyxy)
            segmentation = mask_utils.encode(np.asfortranarray(mask_np))
            segmentation["counts"] = segmentation["counts"].decode("utf-8")
            
            # 计算 OBB（旋转边界框）
            obb = self._compute_obb_from_mask(mask_np)

            instance = {
                "mask": mask_np,
                "score": score_val,
                "label": label_val,
                "bbox_xyxy": bbox_xyxy,
                "bbox_xywh": bbox_xywh,
                "segmentation": segmentation,
                "obb": obb,  # 新增 OBB 字段
                "category_id": self._label_to_category_id(label_val),
                "image_id": image_info.get("id"),
                "file_name": image_info.get("file_name"),
            }
            instances.append(instance)

        return {"instances": instances, "image_info": image_info}

    def _save_export_outputs(
        self,
        payload: Dict[str, Any],
        paths: Dict[str, Path],
        img_path: str,
        tar_anns_by_cat: OrderedDict,
        stem: str,
    ) -> Dict[str, str]:
        """保存最终可视化图、COCO JSON、逐实例 PNG，并返回路径。

        参数:
            payload: `_build_export_payload` 输出结构。
            paths: 包含 `image`、`json`、`instances` 的路径字典。
            img_path: 原始图像文件路径（保持正确比例）。
            tar_anns_by_cat: 字典，包含按类别组织的 GT annotations。
            stem: 当前图片的名字（不含扩展名）。

        返回:
            三个实际写入路径字符串组成的字典。
        """

        image_path = self._save_prediction_image(
            payload=payload,
            img_path=img_path,
            tar_anns_by_cat=tar_anns_by_cat,
            save_path=paths["image"],
        )
        json_path = self._save_prediction_json(payload, paths["json"])
        instances_dir = self._save_instance_images(payload, img_path, paths["instances"], stem)
        binary_masks_dir = self._save_binary_masks(payload, paths["binary_masks"], stem)  # 新增：保存二值掩码

        return {
            "image": str(image_path),
            "json": str(json_path),
            "instances": str(instances_dir),
            "binary_masks": str(binary_masks_dir),  # 新增
        }

    def _save_prediction_json(self, payload: Dict[str, Any], save_path: Path) -> Path:
        """保存 COCO 风格 JSON 结果（包含 OBB），并返回写入路径。"""

        _ensure_dir(save_path)
        
        # 检查是否覆盖旧文件
        if save_path.exists():
            print(f"   🔄 更新已有 JSON 文件")
        
        import math
        
        export_list: List[Dict[str, Any]] = []
        skipped_count = 0
        for ins in payload["instances"]:
            # 检查 score 是否为 NaN 或 Inf
            score_val = float(ins["score"])
            if math.isnan(score_val) or math.isinf(score_val) or not (0.0 <= score_val <= 1.0):
                skipped_count += 1
                print(f"   ⚠️  Warning: Skipping instance with invalid score: {score_val}")
                continue
            
            # 获取类别名称和置信度百分比
            category_name = self._label_to_name(ins["label"])
            score_percent = int(round(score_val * 100))
            category_label = f"{category_name}={score_percent}"
            
            result = {
                "image_id": ins["image_id"],
                "category_id": ins["category_id"],
                "category_name": category_label,  # 新增：格式为 "ship=51"
                "bbox": ins["bbox_xywh"].tolist(),
                "score": score_val,
                "segmentation": ins["segmentation"],
                "obb": ins["obb"],  # 添加 OBB 字段
            }
            export_list.append(result)
        
        if skipped_count > 0:
            print(f"   ⚠️  Warning: Skipped {skipped_count} instances with invalid scores")

        with save_path.open("w", encoding="utf-8") as f:
            json.dump(export_list, f, ensure_ascii=False, indent=2)
        return save_path

    def _save_prediction_image(
        self,
        payload: Dict[str, Any],
        img_path: str,
        tar_anns_by_cat: OrderedDict,
        save_path: Path,
    ) -> Path:
        """保存叠加了预测实例的最终可视化结果（使用 vis_coco 生成左右对比图）。"""

        _ensure_dir(save_path)
        
        # 检查是否覆盖旧文件
        if save_path.exists():
            print(f"   🔄 更新已有可视化图像")
        
        # 使用原有的 vis_coco 函数生成左右对比图
        from zods_rs.dataset.visualization import vis_coco
        
        import math
        import numpy as np
        
        # 准备数据（过滤 NaN/Inf）
        valid_instances = []
        for ins in payload["instances"]:
            score_val = float(ins["score"])
            if not (math.isnan(score_val) or math.isinf(score_val)):
                valid_instances.append(ins)
        
        if len(valid_instances) == 0:
            print(f"   ⚠️  Warning: No valid instances after filtering NaN/Inf scores")
            # 创建空的可视化
            scores_np = np.array([])
            labels_np = np.array([])
            bboxes_np = np.array([])
            masks_np = np.array([])
        else:
            scores_np = np.array([ins["score"] for ins in valid_instances])
            labels_np = np.array([ins["label"] for ins in valid_instances])
            bboxes_np = np.array([ins["bbox_xyxy"] for ins in valid_instances])
            masks_np = np.array([ins["mask"] for ins in valid_instances])
        
        # 准备 GT 数据
        gt_bboxes_list = []
        gt_labels_list = []
        gt_masks_list = []
        for cat_ind, ann in tar_anns_by_cat.items():
            gt_masks = ann["masks"].detach().cpu().numpy()
            gt_bboxes = ann.get("bboxes", batched_mask_to_box(ann["masks"])).detach().cpu().numpy()
            n_gt = gt_masks.shape[0]
            gt_masks_list.append(gt_masks)
            gt_bboxes_list.append(gt_bboxes)
            gt_labels_list.extend([cat_ind] * n_gt)
        
        gt_bboxes_np = np.concatenate(gt_bboxes_list) if gt_bboxes_list else np.array([])
        gt_masks_np = np.concatenate(gt_masks_list) if gt_masks_list else np.array([])
        gt_labels_np = np.array(gt_labels_list)
        
        # 调用 vis_coco 生成左右对比图（使用配置的 vis_thr）
        vis_coco(
            gt_bboxes_np,
            gt_labels_np,
            gt_masks_np,
            scores_np,
            labels_np,
            bboxes_np,
            masks_np,
            score_thr=self.vis_thr,  # 使用配置的阈值
            img_path=img_path,
            out_path=str(save_path),
            show_scores=True,
            class_names=self.class_names,
            dataset_name=self.dataset_name,
        )
        
        return save_path
    
    def _save_binary_masks(
        self,
        payload: Dict[str, Any],
        save_dir: Path,
        stem: str,
    ) -> Path:
        """保存每个实例的纯二值掩码（黑白图，不含边界框）。
        
        参数:
            payload: 导出数据字典
            save_dir: 保存目录
            stem: 图像名称（不含扩展名）
        
        返回:
            保存目录路径
        """
        import shutil
        
        # 先清空旧文件
        if save_dir.exists():
            old_files = list(save_dir.glob("*.png"))
            if old_files:
                print(f"   🗑️  清理旧二值掩码: {len(old_files)} 个文件")
            shutil.rmtree(save_dir)
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        import math
        
        # 只保存分数高于阈值的实例
        saved_count = 0
        for idx, ins in enumerate(payload["instances"]):
            score_val = float(ins["score"])
            # 检查 NaN/Inf 和阈值
            if math.isnan(score_val) or math.isinf(score_val) or score_val < self.vis_thr:
                continue  # 跳过无效分数或低分实例
            
            # 提取二值掩码（0 或 255）
            mask = ins["mask"].astype(np.uint8) * 255
            
            # 保存为灰度图（黑白二值）
            mask_img = Image.fromarray(mask, mode='L')
            
            label_str = self._label_to_name(ins["label"])
            score_int = int(round(score_val * 100))
            mask_path = save_dir / f"{stem}_{label_str}_{score_int:03d}_mask_{saved_count:03d}.png"
            
            mask_img.save(mask_path)
            saved_count += 1
        
        return save_dir

    def _save_instance_images(
        self,
        payload: Dict[str, Any],
        img_path: str,
        save_dir: Path,
        stem: str,
    ) -> Path:
        """保存每个实例的 PNG，可视化遮罩覆盖效果（单实例高亮）。"""

        # 先清空旧的实例图像
        import shutil
        if save_dir.exists():
            old_files = list(save_dir.glob("*.png"))
            if old_files:
                print(f"   🗑️  清理旧实例图像: {len(old_files)} 个文件")
            shutil.rmtree(save_dir)
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 从原始文件加载图像（保持正确比例）
        base_img = Image.open(img_path).convert("RGB")
        
        try:
            import cv2
        except ImportError:
            cv2 = None
            warnings.warn("cv2 not available, instance images will be basic overlays")
        
        try:
            from zods_rs.dataset.visualization import draw_box_on_image
        except ImportError:
            draw_box_on_image = None

        import math
        
        # 只保存分数高于阈值的实例
        saved_count = 0
        for idx, ins in enumerate(payload["instances"]):
            score_val = float(ins["score"])
            # 检查 NaN/Inf 和阈值
            if math.isnan(score_val) or math.isinf(score_val) or score_val < self.vis_thr:
                continue  # 跳过无效分数或低分实例
            
            # 为每个实例创建独立的可视化
            img_copy = base_img.copy()
            img_np = np.array(img_copy)
            
            color = INSTANCE_COLORS[idx % len(INSTANCE_COLORS)].tolist()
            mask = ins["mask"].astype(np.uint8)
            
            # 使用 cv2 绘制轮廓（更清晰）
            if cv2 is not None:
                seg_thickness = max(2, int(img_np.shape[1] * 0.003))
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img_np, contours, -1, color, seg_thickness)
            
            # 转回 PIL 并绘制边界框和标签
            overlay_img = Image.fromarray(img_np)
            
            if draw_box_on_image is not None:
                bbox = ins["bbox_xyxy"]
                label_str = self._label_to_name(ins["label"])
                score_int = int(round(score_val * 100))
                text = f"{label_str}={score_int}"
                
                try:
                    draw_box_on_image(
                        overlay_img,
                        [bbox.tolist()],
                        [text],
                        show_label=True,
                        colors=[color],
                    )
                except Exception as e:
                    # 如果绘制失败，至少保存掩码图
                    pass
            
            overlay_path = save_dir / f"{stem}_mask_{saved_count:03d}.png"
            overlay_img.save(overlay_path)
            saved_count += 1

        return save_dir

    def _tensor_to_image(self, tensor_img: torch.Tensor) -> np.ndarray:
        """反标准化张量到 uint8 图像。"""

        if tensor_img.dim() == 3:
            pass
        elif tensor_img.dim() == 4:
            tensor_img = tensor_img[0]
        else:
            raise ValueError("Unsupported image tensor shape")

        img = tensor_img.detach().cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        img = (img * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        return img

    def _label_to_category_id(self, label: int) -> int:
        """将内部 label 转换为原始 COCO category id。
        
        注意：对于自定义数据集，category_id 通常与内部 label 索引一致。
        如需真实 COCO id 映射，需在 dataset 中提供 cat_inds_to_ids。
        """
        return int(label)

    @staticmethod
    def _mask_to_xywh(bbox_xyxy: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = bbox_xyxy
        return np.array([x1, y1, x2 - x1, y2 - y1], dtype=float)
    
    @staticmethod
    def _compute_obb_from_mask(mask: np.ndarray) -> List[float]:
        """从掩码计算 OBB（旋转边界框）。
        
        参数:
            mask: 二值掩码 (H, W)，0 或 1
        
        返回:
            8 个浮点数列表 [x1, y1, x2, y2, x3, y3, x4, y4]，
            表示旋转矩形的四个角点坐标。
        """
        try:
            import cv2
        except ImportError:
            # cv2 不可用时，返回空 OBB
            warnings.warn("cv2 not available, OBB will be empty")
            return [0.0] * 8
        
        ys, xs = np.where(mask > 0)
        if xs.size > 0:
            coords = np.column_stack([xs, ys]).astype(np.float32)
            rect = cv2.minAreaRect(coords)
            obb = cv2.boxPoints(rect).flatten().tolist()
            return obb
        else:
            return [0.0] * 8

    def _label_to_name(self, label: int) -> str:
        if self.class_names and 0 <= label < len(self.class_names):
            return str(self.class_names[label])
        return f"class_{label}"
    
    def _get_image_path(self, image_info: Dict[str, Any]) -> str:
        """从 image_info 中获取原始图像的完整路径。"""
        file_name = image_info.get("file_name", "")
        
        # 根据 dataset_name 确定基础路径
        if self.dataset_name == "coco":
            base_path = "./data/coco/val2017"
        elif self.dataset_name == "lvis":
            base_path = "./data/coco/allimages"
        elif self.dataset_imgs_path:
            base_path = self.dataset_imgs_path
        else:
            # 尝试从 image_info 中获取
            base_path = image_info.get("dataset_dir", ".")
        
        return os.path.join(base_path, file_name)

    def testing_classifier(self, input_dicts, with_negative):
        assert len(input_dicts) == 1

        device = self.predictor.device

        ori_h = input_dicts[0]["target_img_info"]["ori_height"]
        ori_w = input_dicts[0]["target_img_info"]["ori_width"]

        tar_img = input_dicts[0]["target_img"].to(device=device)
        sam_input_size = tar_img.shape[-2]
        tar_img_encoder = F.interpolate(
            tar_img.unsqueeze(dim=0),
            size=(self.encoder_img_size, self.encoder_img_size),
            mode="bicubic"
        )
        tar_feat, last_attn = self._forward_encoder_attn_roll(tar_img_encoder)
        tar_feat = tar_feat.reshape(-1, self.encoder_dim)  # [N, C]

        tar_anns_by_cat = input_dicts[0]["tar_anns_by_cat"]

        masks = []
        for cat_ind in tar_anns_by_cat.keys():
            masks.append(tar_anns_by_cat[cat_ind]["masks"].to(device=device))

        if len(masks) > 0:
            masks = torch.cat(masks, dim=0)
        else:
            masks = torch.ones((1, sam_input_size, sam_input_size)).to(device=device, dtype=torch.float32)

        n_masks = masks.shape[0]
        masks_feat_size = F.interpolate(
            masks.unsqueeze(dim=1),
            size=(self.encoder_h, self.encoder_w),
            mode="nearest"
        ).reshape(n_masks, -1)

        masks_feat_size_bool = masks_feat_size > 0
        non_empty_inds = masks_feat_size_bool.sum(dim=1) > 0

        masks_feat_size = masks_feat_size[non_empty_inds]
        masks_feat_size_bool = masks_feat_size_bool[non_empty_inds]
        masks_out = masks[non_empty_inds]

        n_masks = masks_feat_size_bool.shape[0]
        if n_masks == 0:
            output_dict = dict(
                binary_masks=torch.zeros((0, ori_h, ori_w)) > 0,
                bboxes=torch.zeros((0, 4)),
                scores=torch.zeros((0,)),
                labels=torch.zeros((0,)),
                # score_to_analysis=score_to_analysis,
                image_info=input_dicts[0]["target_img_info"],
            )
            return [output_dict]

        if not with_negative:
            # sim_local = self._compute_sim_matching(tar_feat, masks_feat_size_bool)
            # pca_scores = self._compute_pca_scores(tar_feat, masks_feat_size_bool)
            # sim_global = self._compute_sim_center(tar_feat, masks_feat_size_bool)
            # sim_global = self._compute_sim_global_avg(tar_feat, masks_feat_size_bool)
            # sim_global = self._compute_sim_global_avg(tar_feat, masks_feat_size_bool, softmax=True, temp=0.5)
            # sim_global = self._compute_sim_instance_softmax(tar_feat, masks_feat_size_bool, temp=0.75)
            sim_global = self._compute_sim_attn_guided_global_avg(tar_feat, last_attn, masks_feat_size_bool)
        else:
            assert self.with_negative_refs
            assert self.memory_neg_ready
            sim_global = self._compute_sim_global_avg(tar_feat, masks_feat_size_bool)
            # sim_global = self._compute_sim_global_avg_with_neg(tar_feat, masks_feat_size_bool, margin=0.6, sigma=1.0)

        self.cls_num_per_mask = 1
        merged_scores = sim_global  # * torch.pow(pca_scores, 0.25)
        top_scores, labels = torch.topk(merged_scores, k=self.cls_num_per_mask)

        # top_scores, labels = self._compute_sim_knn(tar_feat, masks_feat_size_bool, k=1, sigma=0.1)

        labels_out = labels.flatten()
        scores_out = top_scores.flatten()

        masks_out_binary = F.interpolate(
            masks_out.unsqueeze(dim=1),
            size=(ori_h, ori_w),
            mode="bilinear",
            align_corners=False,
            antialias=True
        ).squeeze(dim=1) > 0

        bboxes = batched_mask_to_box(masks_out_binary)

        final_out_num = min(self.num_out_instance, scores_out.shape[0])
        final_out_inds = torch.argsort(scores_out, descending=True)[:final_out_num]
        masks_out_binary = masks_out_binary[final_out_inds]
        bboxes = bboxes[final_out_inds]
        scores_out = scores_out[final_out_inds]
        labels_out = labels_out[final_out_inds]

        # score_to_analysis = torch.stack((local_global_mean, local_global_std, select_scores_oracle), dim=-1)

        output_dict = dict(
            binary_masks=masks_out_binary,
            bboxes=bboxes,
            scores=scores_out,
            labels=labels_out,
            # score_to_analysis=score_to_analysis,
            image_info=input_dicts[0]["target_img_info"],
        )
        # self._vis_results_online(output_dict, input_dicts[0]["tar_anns_by_cat"], score_thr=0.0, show_scores=True)
        return [output_dict]

    def postprocess_memory(self):
        if PRINT_TIMING:
            start_time = time.time()
        # Compute class-wise average features
        device = self.mem_feats_avg.device
        c = self.mem_feats.shape[-1]

        # Reset PP caches before recomputation
        if hasattr(self, "mem_pp_prototypes"):
            self.mem_pp_prototypes.zero_()
        self.mem_pp_subprototypes = {}
        self.mem_pp_debug = {}

        self.mem_feats_avg *= 0.0
        mem_feats_avg = (
            torch.sum(self.mem_feats * self.mem_masks.unsqueeze(dim=-1), dim=(1, 2))
            / self.mem_masks.sum(dim=(1, 2)).unsqueeze(dim=1)
        )
        self.mem_feats_avg += mem_feats_avg

        mem_feats_ins_avg = (
            torch.sum(self.mem_feats * self.mem_masks.unsqueeze(dim=-1), dim=2)
            / self.mem_masks.sum(dim=2).unsqueeze(dim=2)
        )
        self.mem_feats_ins_avg += mem_feats_ins_avg

        sigmas = []
        for i in range(self.mem_n_classes):
            # feats_i = self.mem_feats[i].reshape(-1, c)
            # masks_i = self.mem_masks[i].reshape(-1)
            # feats_i_fore = feats_i[masks_i > 0]
            # mu_i = feats_i_fore.mean(dim=0, keepdim=True)
            # feats_i_centered = feats_i_fore - mu_i

            feats_i = self.mem_feats_ins_avg[i].reshape(-1, c)
            mu_i = feats_i.mean(dim=0, keepdim=True)
            feats_i_centered = feats_i - mu_i

            sigma_i = feats_i_centered.t() @ feats_i_centered / float(feats_i_centered.shape[0])
            sigmas.append(sigma_i.unsqueeze(dim=0))
        sigmas = torch.cat(sigmas, dim=0)
        self.mem_feats_covariances += sigmas




        # compute mean sim, method 1
        # ins_sims = []
        # for i in range(self.mem_n_classes):
        #     feats_i = []
        #     for j in range(self.mem_length):
        #         feat_ij = self.mem_feats[i, j]
        #         mask_ij = self.mem_masks[i, j]
        #         feats_i.append(feat_ij[mask_ij > 0].mean(dim=0, keepdim=True))
        #     feats_i = F.normalize(torch.cat(feats_i, dim=0), p=2, dim=-1)
        #     ins_sims_i = feats_i @ feats_i.t()
        #     ins_sims_i = (ins_sims_i + 1.0) * 0.5
        #     ins_sims_i = ins_sims_i[ins_sims_i < 1.0].mean()
        #     ins_sims.append(ins_sims_i)
        # ins_sims = torch.stack(ins_sims).reshape(self.mem_n_classes)
        # self.mem_ins_sim_avg += ins_sims

        # compute mean sim, method 2
        ins_sims = []
        for i in range(self.mem_n_classes):
            sims_i = []
            for j in range(self.mem_length):
                feat_ij = self.mem_feats[i, j]
                feats_i_rest = torch.cat((self.mem_feats[i, :j], self.mem_feats[i, j+1:]), dim=0)
                mask_ij = self.mem_masks[i, j]
                mask_i_rest = torch.cat((self.mem_masks[i, :j], self.mem_masks[i, j+1:]), dim=0)

                feats_ij = F.normalize(feat_ij[mask_ij > 0].mean(dim=0, keepdim=True), p=2, dim=1)
                feats_i_rest = F.normalize(feats_i_rest[mask_i_rest > 0].mean(dim=0, keepdim=True), p=2, dim=-1)
                sim_ij = feats_ij @ feats_i_rest.t()
                sim_ij = (sim_ij + 1.0) * 0.5
                sims_i.append(sim_ij)
            sim_i = torch.stack(sims_i).mean()
            ins_sims.append(sim_i)
        ins_sims = torch.stack(ins_sims).reshape(self.mem_n_classes)
        self.mem_ins_sim_avg += ins_sims

        # K-means
        kmeans_iters = 100
        for i in range(self.mem_n_classes):
            feats = self.mem_feats[i].reshape(-1, self.encoder_dim)[self.mem_masks[i].reshape(-1) > 0]
            assert feats.shape[0] > 0
            centers_i = kmeans(feats, self.kmeans_k, kmeans_iters)
            # centers_i = kmeans_decouple(feats, feats_fore, self.kmeans_k, kmeans_iters)
            self.mem_feats_centers[i] += centers_i

        # PCA
        for i in range(self.mem_n_classes):
            feats = self.mem_feats[i].reshape(-1, self.encoder_dim)[self.mem_masks[i].reshape(-1) > 0]
            assert feats.shape[0] > 0
            feats = feats.cpu().numpy()
            pca = PCA(n_components=self.n_pca_components)
            pca.fit(feats)
            pca_mean = torch.from_numpy(pca.mean_).to(device=device)
            pca_components = torch.from_numpy(pca.components_).to(device=device)
            self.mem_pca_mean[i] += pca_mean
            self.mem_pca_components[i] += pca_components

        # TODO: FoundPose's Method

        if self.pp_enabled:
            try:
                ref_feats_by_class = self._gather_pp_reference_features()
                if ref_feats_by_class:
                    clip_hooks = self._build_pp_clip_hooks()
                    debug_store: Dict[int, Dict[str, Any]] = {}
                    
                    # 根据配置选择PP变体
                    use_ot = self.pp_cfg.get("ot", {}).get("enable", False)
                    use_robust = self.pp_cfg.get("robust", {}).get("enable", False)
                    
                    if use_ot:
                        print("[PP] Using OT-PP (Sinkhorn-OT Alignment)")
                        proto_dict = class_prototypes_ot(
                            ref_feats_by_class,
                            cfg=self.pp_cfg,
                            clip_hooks=clip_hooks,
                            debug_store=debug_store,
                            store_raw_feats=self.pp_debug_cfg.get("enable", False),
                        )
                    elif use_robust:
                        print("[PP] Using Robust-PP (Tyler's M-estimator)")
                        proto_dict = class_prototypes_robust(
                            ref_feats_by_class,
                            cfg=self.pp_cfg,
                            clip_hooks=clip_hooks,
                            debug_store=debug_store,
                            store_raw_feats=self.pp_debug_cfg.get("enable", False),
                        )
                    else:
                        proto_dict = class_prototypes(
                            ref_feats_by_class,
                            cfg=self.pp_cfg,
                            clip_hooks=clip_hooks,
                            debug_store=debug_store,
                            store_raw_feats=self.pp_debug_cfg.get("enable", False),
                        )

                    if hasattr(self, "mem_pp_prototypes"):
                        self.mem_pp_prototypes.zero_()

                    for cls_id, prototypes in proto_dict.items():
                        if not prototypes:
                            continue

                        stored = [p.to(device=device, dtype=self.mem_feats_avg.dtype) for p in prototypes]
                        primary = stored[0]

                        if hasattr(self, "mem_pp_prototypes"):
                            self.mem_pp_prototypes[cls_id] = primary

                        self.mem_pp_subprototypes[cls_id] = stored
                        self.mem_pp_debug[cls_id] = debug_store.get(cls_id, {})

                        self._log_pp_summary(cls_id, self.mem_pp_debug[cls_id])

                    if PRINT_TIMING:
                        summary_msgs = []
                        for cls_id, info in self.mem_pp_debug.items():
                            summary = info.get("summary", {})
                            if not summary:
                                continue
                            if self.class_names and cls_id < len(self.class_names):
                                cls_name = self.class_names[cls_id]
                            else:
                                cls_name = str(cls_id)
                            scores = summary.get("top_scores")
                            if isinstance(scores, torch.Tensor):
                                score_vals = [f"{float(s):.4f}" for s in scores.tolist()]
                            else:
                                score_vals = []
                            summary_msgs.append(f"{cls_name}:r={summary.get('top_r', 0)}:{score_vals}")
                        if summary_msgs:
                            print("[PP] purified prototypes:", ", ".join(summary_msgs))
                        elif self.mem_pp_subprototypes:
                            classes = [self.class_names[idx] if self.class_names and idx < len(self.class_names) else str(idx) for idx in self.mem_pp_subprototypes.keys()]
                            print("[PP] purified prototypes computed for classes:", ", ".join(classes))

                    if self.pp_debug_cfg.get("enable", False):
                        self._pp_save_debug_artifacts(ref_feats_by_class, debug_store)
                else:
                    if hasattr(self, "mem_pp_prototypes"):
                        self.mem_pp_prototypes.zero_()
                    self.mem_pp_subprototypes = {}
                    self.mem_pp_debug = {}
            except Exception as exc:  # pragma: no cover - fallback guard
                warnings.warn(
                    f"Prototype purification failed and will be skipped: {exc}",
                    RuntimeWarning,
                )

        self.mem_postprocessed[0] = True

        if PRINT_TIMING:
            end_time = time.time()
            print("--------------------------------")
            print("TIMING POSTPROCESS MEMORY: ", end_time - start_time)
            print("--------------------------------")

    def postprocess_memory_negative(self):
        device = self.mem_feats_avg.device

        mem_feats_avg_neg = (
                torch.sum(self.mem_feats_neg * self.mem_masks_neg.unsqueeze(dim=-1), dim=(1, 2))
                / self.mem_masks_neg.sum(dim=(1, 2)).unsqueeze(dim=1)
        )
        self.mem_feats_avg_neg += mem_feats_avg_neg

        mem_feats_ins_avg_neg = (
                torch.sum(self.mem_feats_neg * self.mem_masks_neg.unsqueeze(dim=-1), dim=2)
                / self.mem_masks_neg.sum(dim=2).unsqueeze(dim=2)
        )
        self.mem_feats_ins_avg_neg += mem_feats_ins_avg_neg
        self.mem_postprocessed_neg[0] = True

    def forward(self, input_dicts):
        data_mode = input_dicts[0].pop("data_mode", None)

        assert data_mode is not None
        assert not self.training

        if data_mode == "fill_memory":
            if PRINT_TIMING:
                start_time = time.time()
            results = self.forward_fill_memory(input_dicts, is_positive=True)
            if PRINT_TIMING:
                end_time = time.time()
                print("--------------------------------")
                print("TIMING FILL MEMORY: ", end_time - start_time)
                print("--------------------------------")
            return results
        elif data_mode == "fill_memory_neg":
            assert self.with_negative_refs
            assert not self.memory_neg_ready
            assert not self.mem_postprocessed_neg[0].item()
            return self.forward_fill_memory(input_dicts, is_positive=False)
        elif data_mode == "vis_memory":
            return self.forward_vis_memory(input_dicts)
        elif data_mode == "test":
            if self.with_negative_refs:
                if not self.memory_ready:
                    if self.mem_postprocessed[0].item():
                        self.memory_ready = True
                    else:
                        raise RuntimeError("Memory is not ready!")
                if not self.memory_neg_ready:
                    if self.mem_postprocessed_neg[0].item():
                        self.memory_neg_ready = True
                    else:
                        raise RuntimeError("Negative memory is not ready!")

                return self.forward_test(input_dicts, with_negative=True)
                # return self.testing_classifier(input_dicts, with_negative=True)
            else:
                if not self.memory_ready:
                    if self.mem_postprocessed[0].item():
                        self.memory_ready = True
                    else:
                        raise RuntimeError("Memory is not ready!")
                return self.forward_test(input_dicts, with_negative=False)
                # return self.testing_classifier(input_dicts, with_negative=False)
        elif data_mode == "test_support":
            assert self.with_negative_refs
            if not self.memory_ready:
                if self.mem_postprocessed[0].item():
                    self.memory_ready = True
                else:
                    raise RuntimeError("Memory is not ready!")
            assert not self.memory_neg_ready
            assert not self.mem_postprocessed_neg[0].item()
            return self.forward_test(input_dicts, with_negative=False)
            # return self.testing_classifier(input_dicts, with_negative=False)
        else:
            raise NotImplementedError(f"Unrecognized data mode during inference: {data_mode}")

    def _vis_results_online(self, output_dict, tar_anns_by_cat, score_thr=0.65, show_scores=False, dataset_name=None, dataset_imgs_path=None, class_names=None):
        import os
        from zods_rs.dataset.visualization import vis_coco

        scores = output_dict["scores"].cpu().numpy()
        masks_pred = output_dict["binary_masks"].cpu().numpy()
        bboxes = output_dict["bboxes"].cpu().numpy()
        labels = output_dict["labels"].cpu().numpy()

        image_info = output_dict["image_info"]
        if dataset_name == "coco" or dataset_name == "few_shot_classes":
            img_path = os.path.join(f"./data/coco/val2017", image_info["file_name"])
        elif dataset_name == "lvis":
            img_path = os.path.join(f"./data/coco/allimages", image_info["file_name"])
        else:
            img_path = os.path.join(dataset_imgs_path, image_info["file_name"])
        out_path = os.path.join(f"./results_analysis/{dataset_name}", image_info["file_name"])

        gt_masks = []
        gt_bboxes = []
        gt_labels = []

        for cat_ind in tar_anns_by_cat.keys():
            gt_masks.append(tar_anns_by_cat[cat_ind]["masks"].cpu().numpy())
            gt_bboxes.append(tar_anns_by_cat[cat_ind]["bboxes"].cpu().numpy())
            gt_labels.extend([cat_ind for _ in range(len(tar_anns_by_cat[cat_ind]["masks"]))])
        if len(gt_bboxes) > 0:
            gt_bboxes = np.concatenate(gt_bboxes)
            gt_masks = np.concatenate(gt_masks)

            gt_bboxes[:, 0] = gt_bboxes[:, 0] * image_info["ori_width"] / self.sam_img_size
            gt_bboxes[:, 1] = gt_bboxes[:, 1] * image_info["ori_height"] / self.sam_img_size
            gt_bboxes[:, 2] = gt_bboxes[:, 2] * image_info["ori_width"] / self.sam_img_size
            gt_bboxes[:, 3] = gt_bboxes[:, 3] * image_info["ori_height"] / self.sam_img_size

        # Resize gt masks
        if len(gt_masks) > 0:
            gt_masks = F.interpolate(
                torch.from_numpy(gt_masks).unsqueeze(dim=1),
                size=(image_info["ori_height"], image_info["ori_width"]),
                mode="nearest"
            ).squeeze(dim=1).numpy()

        vis_coco(
            gt_bboxes,
            gt_labels,
            gt_masks,
            scores,
            labels,
            bboxes,
            masks_pred,
            score_thr=score_thr,
            img_path=img_path,
            out_path=out_path,
            show_scores=show_scores,
            dataset_name=dataset_name,
            class_names=class_names
        )

    def _build_uam_cfg(self, cfg: dict) -> dict:
        default_debug = {
            "save_probs": False,
            "save_entropy": False,
            "save_energy": False,
            "save_masks": False,
            "out_dir": "./results_analysis/uam_debug",
        }
        default_cfg = {
            "enable": False,
            "temperature": 1.0,
            "threshold": 0.5,
            "calibrate": True,
            "auto_tau": {
                "enable": False,
                "method": "entropy_target",
                "target_entropy": 0.5,
                "percentile": 90.0,
                "tau_range": [0.1, 3.0],
                "verbose": False,
            },
            "prior": {
                "enable": False,
                "combine": "multiply",
                "margin": {
                    "enable": False,
                    "width": 0.1,
                    "sharpen": 2.0,
                },
                "norm": {
                    "enable": False,
                    "sigma": 0.3,
                },
            },
            "crf": {
                "enable": False,
                "n_iters": 5,
                "sxy": [3, 3],
                "srgb": [5, 5, 5],
            },
            "memory": {
                "sparsify": False,
                "chunk": None,
            },
            "debug": default_debug,
        }
        merged = copy.deepcopy(default_cfg)
        merged.update(cfg)
        merged["debug"] = {**default_debug, **cfg.get("debug", {})}
        merged["crf"] = {**default_cfg["crf"], **cfg.get("crf", {})}
        merged["memory"] = {**default_cfg["memory"], **cfg.get("memory", {})}
        merged["auto_tau"] = {**default_cfg["auto_tau"], **cfg.get("auto_tau", {})}
        merged["prior"] = {**default_cfg["prior"], **cfg.get("prior", {})}
        merged["prior"]["margin"] = {
            **default_cfg["prior"]["margin"],
            **cfg.get("prior", {}).get("margin", {}),
        }
        merged["prior"]["norm"] = {
            **default_cfg["prior"]["norm"],
            **cfg.get("prior", {}).get("norm", {}),
        }
        return merged
    
    def _build_sem_cfg(self, cfg: dict) -> dict:
        """构建 SEM 配置，提供默认值。"""
        default_debug = {
            "save_weights": False,
            "save_heatmap": False,
            "save_matches": False,
            "save_alpha_beta": False,
            "save_attn": False,
            "out_dir": "./results_analysis/sem_debug",
        }
        default_dino = {
            "use_multilayer": False,
            "layers": [6, 10, -1],
            "use_attn_prior": False,
            "gamma": 0.5,
            "gamma_auto": False,
            "cheap_scales": False,
        }
        default_cfg = {
            "enable": False,
            "scales": [1.0, 0.5, 0.25],
            "alpha": 0.5,
            "method": "greedy",
            "cache_pyr": False,
            "score_dtype": "fp32",
            "dino": default_dino,
            "debug": default_debug,
        }
        merged = copy.deepcopy(default_cfg)
        merged.update(cfg)
        merged["debug"] = {**default_debug, **cfg.get("debug", {})}
        merged["dino"] = {**default_dino, **cfg.get("dino", {})}
        return merged

    def _build_pp_cfg(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """构建 PP 配置，提供默认值并确保层级存在。"""

        default_robust = {
            "enable": False,  # 默认关闭，使用标准 PP
            "iters": 20,
            "tol": 1e-4,
            "init": "identity",
            "eps": 1e-6,
            "trace_norm": True,
            "verbose": False,
        }

        default_cluster = {
            "method": "hdbscan",
            "min_size": 10,
            "max_k": 4,
        }

        default_clip = {
            "enable": False,
            "model": "ViT-B/16",
            "texts": {},
        }

        default_cfg = {
            "enable": True,
            "robust": default_robust,
            "top_r": 32,
            "alpha": 0.7,
            "beta": 0.3,
            "use_whitening": False,
            "use_subprototypes": True,
            "cluster": default_cluster,
            "clip": default_clip,
        }

        merged = copy.deepcopy(default_cfg)
        merged.update(cfg or {})
        merged["robust"] = {**default_robust, **(cfg or {}).get("robust", {})}
        merged["cluster"] = {**default_cluster, **(cfg or {}).get("cluster", {})}
        merged["clip"] = {**default_clip, **(cfg or {}).get("clip", {})}
        return merged

    def _gather_pp_reference_features(self) -> Dict[int, torch.Tensor]:
        """从 memory bank 中收集每个类别的参考特征集合。"""

        ref_feats: Dict[int, torch.Tensor] = {}
        device = self.mem_feats.device

        min_samples = max(3, int(self.pp_cfg.get("min_samples", 3)))

        for cls_id in range(self.mem_n_classes):
            masks_cls = self.mem_masks[cls_id]
            if masks_cls.sum() <= 0:
                continue

            mask_flat = masks_cls.reshape(-1) > 0
            if mask_flat.sum() < min_samples:
                continue

            feats_cls = self.mem_feats[cls_id].reshape(-1, self.encoder_dim)
            selected = feats_cls[mask_flat].clone()
            if selected.numel() == 0:
                continue

            ref_feats[cls_id] = selected.to(device=device, dtype=self.mem_feats.dtype)

        if not ref_feats:
            warnings.warn("Prototype purification收到空的参考特征，将跳过 PP 阶段。", RuntimeWarning)

        return ref_feats

    def _build_pp_clip_hooks(self) -> Dict[str, Callable[[int], Optional[torch.Tensor]]]:
        """构建可选的 CLIP 先验获取钩子。"""

        clip_cfg = self.pp_cfg.get("clip", {}) if isinstance(self.pp_cfg, dict) else {}
        if not clip_cfg.get("enable", False):
            return {}

        try:
            from utils import clip_adapters  # type: ignore
        except Exception as exc:  # pragma: no cover - graceful degradation
            warnings.warn(
                f"CLIP 先验不可用（{exc}），PP 将退化为单源谱净化。",
                RuntimeWarning,
            )
            return {}

        hooks = clip_adapters.build_clip_hooks(
            texts=clip_cfg.get("texts", {}),
            class_names=self.class_names,
            image_feats=clip_cfg.get("image_feats"),
            device=self.mem_feats.device,
            dtype=self.mem_feats.dtype,
            model_name=clip_cfg.get("model_name", "ViT-B/16"),
        )

        if not isinstance(hooks, dict):
            warnings.warn("CLIP 先验钩子返回结果异常，将忽略。", RuntimeWarning)
            return {}

        return hooks

    def _log_pp_summary(self, cls_id: int, info: Dict[str, Any]) -> None:
        """记录谱方向得分的摘要信息，便于调试。"""

        if not info:
            return

        purify_res = info.get("purify")
        if not isinstance(purify_res, dict):
            return

        scores = purify_res.get("scores")
        if isinstance(scores, torch.Tensor) and scores.numel() > 0:
            top_k = min(3, scores.numel())
            top_vals, _ = torch.topk(scores, k=top_k)
            summary = {
                "top_r": int(scores.numel()),
                "top_scores": top_vals.detach().cpu(),
            }
            info["summary"] = summary

            if self.pp_cfg.get("log_summary", False):
                if self.class_names and cls_id < len(self.class_names):
                    cls_name = self.class_names[cls_id]
                else:
                    cls_name = f"class_{cls_id}"
                score_list = [round(v.item(), 4) for v in top_vals]
                print(f"[PP] {cls_name}: top_r={summary['top_r']} top_scores={score_list}")

    def get_pp_prototype(self, cls_id: int) -> Optional[torch.Tensor]:
        """获取指定类别的主原型，若 PP 未启用或不存在则返回 None。"""

        if not self.pp_enabled:
            return None

        if not hasattr(self, "mem_pp_prototypes"):
            return None

        if cls_id < 0 or cls_id >= self.mem_pp_prototypes.shape[0]:
            return None

        proto = self.mem_pp_prototypes[cls_id]
        if proto.abs().sum() == 0:
            return None
        return F.normalize(proto, dim=0)

    def get_pp_subprototypes(self, cls_id: int) -> Optional[List[torch.Tensor]]:
        """获取指定类别的子原型列表（若存在）。"""

        if not self.pp_enabled:
            return None

        return self.mem_pp_subprototypes.get(cls_id)

    def _pp_save_debug_artifacts(
        self,
        ref_feats_by_class: Dict[int, torch.Tensor],
        debug_store: Dict[int, Dict[str, Any]],
    ) -> None:
        max_classes = self.pp_debug_cfg.get("max_classes", 6)
        out_dir = self.pp_debug_cfg.get("out_dir", Path("pp_debug"))
        out_dir.mkdir(parents=True, exist_ok=True)

        top_r = self.pp_cfg.get("top_r", 0)
        alpha = self.pp_cfg.get("alpha", 0.0)
        beta = self.pp_cfg.get("beta", 0.0)
        clip_enabled = self.pp_cfg.get("clip", {}).get("enable", False)
        print(f"[PP] debug: top_r={top_r} alpha={alpha} beta={beta} clip={clip_enabled}")

        classes_to_process = list(debug_store.keys())[:max_classes]

        if self.pp_debug_cfg.get("save_spectrum", False):
            for cls_id in classes_to_process:
                info = debug_store.get(cls_id, {})
                purify = info.get("purify", {})
                S_r = purify.get("S_r")
                scores = purify.get("scores")
                if S_r is None or scores is None:
                    continue

                indices = torch.arange(scores.shape[0])
                top_indices = [int(i) for i in indices.tolist()]
                score_list = scores.detach().cpu().tolist()
                eigvals = S_r.detach().cpu().tolist()

                if self.class_names and cls_id < len(self.class_names):
                    cls_name = self.class_names[cls_id]
                else:
                    cls_name = f"class_{cls_id}"

                plt.figure(figsize=(6, 4))
                plt.plot(eigvals, label="Eigenvalues")
                ax2 = plt.twinx()
                ax2.bar(top_indices, score_list, alpha=0.3, label="Scores")
                plt.title(f"Spectrum & scores - {cls_name}")
                plt.xlabel("Component")
                plt.tight_layout()
                out_path = out_dir / f"{cls_name}_spectrum.png"
                plt.savefig(out_path)
                plt.close()

        if self.pp_debug_cfg.get("save_heatmap", False):
            for cls_id in classes_to_process:
                info = debug_store.get(cls_id, {})
                purify = info.get("purify", {})
                p_bar = purify.get("p_bar")
                p_hat = purify.get("p_hat")
                if p_bar is None or p_hat is None:
                    continue

                sample_feats = ref_feats_by_class.get(cls_id)
                if sample_feats is None:
                    continue

                sims_bar = sample_feats @ p_bar
                sims_hat = sample_feats @ p_hat

                if self.class_names and cls_id < len(self.class_names):
                    cls_name = self.class_names[cls_id]
                else:
                    cls_name = f"class_{cls_id}"

                plt.figure(figsize=(6, 4))
                heat_data = torch.vstack([sims_bar, sims_hat]).cpu().numpy()
                if sns is not None:
                    sns.heatmap(heat_data, cmap="viridis")
                else:
                    plt.imshow(heat_data, aspect="auto", cmap="viridis")
                    plt.colorbar()
                plt.yticks([0.5, 1.5], ["p_bar", "p_hat"])
                plt.title(f"Similarity heatmap - {cls_name}")
                plt.tight_layout()
                out_path = out_dir / f"{cls_name}_heatmap.png"
                plt.savefig(out_path)
                plt.close()

        if self.pp_debug_cfg.get("save_clusters", False):
            for cls_id in classes_to_process:
                info = debug_store.get(cls_id, {})
                purify = info.get("purify", {})
                U_r = purify.get("U_r")
                raw_feats = info.get("raw_feats")
                subs = self.mem_pp_subprototypes.get(cls_id)
                if U_r is None or raw_feats is None or subs is None or not subs:
                    continue

                components = U_r[:, : min(U_r.shape[1], 32)].cpu()
                coords = (raw_feats @ components).numpy()

                if umap is not None and coords.shape[1] > 2:
                    reducer = umap.UMAP(n_components=2)
                    coords_2d = reducer.fit_transform(coords)
                else:
                    pca = SKPCA(n_components=2)
                    coords_2d = pca.fit_transform(coords)

                if self.class_names and cls_id < len(self.class_names):
                    cls_name = self.class_names[cls_id]
                else:
                    cls_name = f"class_{cls_id}"

                plt.figure(figsize=(5, 5))
                plt.scatter(coords_2d[:, 0], coords_2d[:, 1], s=5, alpha=0.4, label="samples")
                centers = []
                for proto in subs:
                    proto_np = proto.cpu().numpy()
                    center_high = proto_np @ components.numpy()
                    center = center_high[:2]
                    centers.append(center)
                centers = np.stack(centers)
                plt.scatter(centers[:, 0], centers[:, 1], c="red", s=40, label="prototypes")
                plt.legend()
                plt.title(f"Subprototype clusters - {cls_name}")
                out_path = out_dir / f"{cls_name}_clusters.png"
                plt.tight_layout()
                plt.savefig(out_path)
                plt.close()

    def _uam_crf_available(self) -> bool:
        return self.uam_cfg.get("crf", {}).get("enable", False) and "pydensecrf" in globals() and globals().get("dcrf") is not None

    def _uam_debug_dir(self) -> Path:
        out_dir = self._uam_debug.get("out_dir", "./results_analysis/uam_debug")
        path = Path(out_dir)
        if any(self._uam_debug.get(flag, False) for flag in ["save_probs", "save_entropy", "save_energy", "save_masks"]):
            path.mkdir(parents=True, exist_ok=True)
        return path
    
    def _sem_debug_dir(self) -> Path:
        """获取 SEM 调试输出目录。"""
        out_dir = self._sem_debug.get("out_dir", "./results_analysis/sem_debug")
        path = Path(out_dir)
        if any(self._sem_debug.get(flag, False) for flag in ["save_weights", "save_heatmap", "save_matches"]):
            path.mkdir(parents=True, exist_ok=True)
        return path
    
    def _sem_debug_dump(self, sem_cache: Dict, img_info: Dict):
        """保存 SEM 调试产物。"""
        debug_flags = ["save_weights", "save_heatmap", "save_matches", "save_alpha_beta", "save_attn"]
        if not any(self._sem_debug.get(flag, False) for flag in debug_flags):
            return
        
        out_dir = self._sem_debug_dir()
        stem = Path(img_info.get("file_name", f"img_{img_info.get('id', 'unknown')}")).stem
        
        if self._sem_debug.get("save_weights") and "scale_weights" in sem_cache:
            torch.save(sem_cache["scale_weights"].cpu(), out_dir / f"{stem}_scale_weights.pt")
        
        if self._sem_debug.get("save_heatmap") and "similarity_map" in sem_cache:
            torch.save(sem_cache["similarity_map"].cpu(), out_dir / f"{stem}_sim_heatmap.pt")
        
        if self._sem_debug.get("save_matches") and "match_results" in sem_cache:
            torch.save(sem_cache["match_results"], out_dir / f"{stem}_matches.pt")
        
        if self._sem_debug.get("save_alpha_beta") and "aux_info" in sem_cache:
            aux = sem_cache["aux_info"]
            alpha_beta_data = {
                "alpha_weights": aux.get("alpha_weights"),
                "beta_weights": aux.get("beta_weights"),
                "gamma": aux.get("gamma_auto", aux.get("attn_contribution", 0.0)),
            }
            torch.save(alpha_beta_data, out_dir / f"{stem}_alpha_beta.pt")
        
        if self._sem_debug.get("save_attn") and "attn_map" in sem_cache:
            attn_data = {
                "map": sem_cache["attn_map"].cpu() if isinstance(sem_cache["attn_map"], torch.Tensor) else None,
                "contribution": sem_cache.get("aux_info", {}).get("attn_contribution", 0.0),
            }
            torch.save(attn_data, out_dir / f"{stem}_attn.pt")

    def _uam_prepare_scores(
        self,
        lr_masks_out,
        scores_out,
        labels_out,
        input_dict,
    ):
        temp = self.uam_cfg.get("temperature", 1.0)
        
        # 构建像素级 logits：分数作为前景类，背景补零
        device = lr_masks_out.device
        n_masks, h, w = lr_masks_out.shape
        
        # 每个掩码建模为二分类：[背景, 前景]
        pixel_logits = torch.zeros(n_masks, 2, h, w, device=device)
        for i in range(n_masks):
            mask = lr_masks_out[i].float()
            score = scores_out[i].item()
            
            # 前景像素的 logit = score，背景像素 = 0
            pixel_logits[i, 0] = (1 - mask) * 0.1  # 背景略高于0
            pixel_logits[i, 1] = mask * score       # 前景根据分数设定
        
        # 为每个掩码单独计算像素级分布
        pixel_probs_list = []
        pixel_entropy_list = []
        
        for i in range(n_masks):
            # 单个掩码的二分类 logits: (1, 2, h, w)
            single_logits = pixel_logits[i:i+1]  # (1, 2, h, w)
            
            stats = pixelwise_distribution(
                single_logits,
                temperature=temp,
                chunk_size=self.uam_cfg.get("memory", {}).get("chunk"),
                mixed_precision=self.uam_cfg.get("memory", {}).get("mixed_precision", False),
                memory_sparsify=self.uam_cfg.get("memory", {}).get("sparsify", False),
                auto_tau_cfg=self.uam_cfg.get("auto_tau"),
                verbose=self.uam_cfg.get("auto_tau", {}).get("verbose", False),
            )
            
            pixel_probs_list.append(stats.probs.squeeze(0))  # (2, h, w)
            pixel_entropy_list.append(stats.entropy.squeeze(0))  # (1, h, w)
        
        pixel_probs = torch.stack(pixel_probs_list, dim=0)  # (n_masks, 2, h, w)
        pixel_entropy = torch.stack(pixel_entropy_list, dim=0)  # (n_masks, 1, h, w)
        
        prior_cfg = self.uam_cfg.get("prior", {})
        prior_maps = {}

        if prior_cfg.get("enable", False):
            masks_binary = lr_masks_out.float()
            if prior_cfg.get("margin", {}).get("enable", False):
                prior_maps["margin"] = margin_prior(
                    masks_binary,
                    margin=prior_cfg["margin"].get("width", 0.1),
                    sharpen=prior_cfg["margin"].get("sharpen", 2.0),
                )
            if prior_cfg.get("norm", {}).get("enable", False):
                prior_maps["norm"] = norm_prior(
                    masks_binary,
                    sigma=prior_cfg["norm"].get("sigma", 0.3),
                )
            if prior_maps:
                combine_mode = prior_cfg.get("combine", "multiply")
                # "logit"/"prob"/"weight" 是置信先验的融合模式，不是空间先验的
                # 这里只处理空间先验的组合
                if combine_mode in {"logit", "prob", "weight"}:
                    # 置信先验模式，空间先验用默认方式组合
                    spatial_combine = "multiply"
                else:
                    spatial_combine = combine_mode
                
                if spatial_combine == "augment":
                    # Probabilistic union to highlight any mask evidence
                    combined = combine_priors(list(prior_maps.values()), mode="augment")
                    prior_maps["combined"] = combined
                    prior_maps["complement"] = 1.0 - combined
                else:
                    prior_maps["combined"] = combine_priors(list(prior_maps.values()), mode=spatial_combine)

        cache = {
            "pixel_logits": pixel_logits,
            "pixel_probs": pixel_probs,
            "pixel_entropy": pixel_entropy,
            "scores": scores_out,
            "labels": labels_out,
            "lr_masks": lr_masks_out,
            "image": input_dict.get("target_img", None),
            "prior_maps": prior_maps,
        }
        return cache

    def _uam_merge(self, scores, masks, labels, pred_ious, cache):
        threshold = self.uam_cfg.get("threshold", 0.5)
        pixel_probs = cache["pixel_probs"]  # (n_masks, 2, h, w)
        pixel_entropy = cache["pixel_entropy"]  # (n_masks, 1, h, w)
        prior_maps = cache.get("prior_maps", {})
        
        # 确保输入长度一致
        n_masks = len(scores)
        assert len(masks) == n_masks, f"Mismatch: {len(masks)} masks vs {n_masks} scores"
        assert len(labels) == n_masks, f"Mismatch: {len(labels)} labels vs {n_masks} scores"
        assert len(pred_ious) == n_masks, f"Mismatch: {len(pred_ious)} pred_ious vs {n_masks} scores"
        assert pixel_probs.shape[0] == n_masks, f"Mismatch: {pixel_probs.shape[0]} pixel_probs vs {n_masks} scores"
        assert pixel_entropy.shape[0] == n_masks, f"Mismatch: {pixel_entropy.shape[0]} pixel_entropy vs {n_masks} scores"
        
        print(f"   UAM merge: input {n_masks} masks")
        
        # 在像素级别应用阈值，但保持掩码结构
        refined_masks = []
        mask_qualities = []
        
        for i in range(len(masks)):
            # 获取前景概率 (第1通道)
            fg_prob = pixel_probs[i, 1]  # (h, w)
            entropy = pixel_entropy[i, 0]  # (h, w)
            prior = prior_maps.get("combined")
            if prior is not None:
                fg_prior = prior[i] if prior.dim() == 3 else prior.squeeze(0)
                fg_prob = fg_prob * fg_prior.to(fg_prob.dtype)

            # 像素级阈值过滤：只保留高置信度的前景像素
            confident_pixels = fg_prob > threshold

            # 构建细化后的掩码
            original_mask = masks[i].bool()
            refined_mask = original_mask & confident_pixels

            # 计算掩码质量（保留像素的平均置信度）
            if refined_mask.any():
                quality = fg_prob[refined_mask].mean()
                mask_qualities.append(quality)
                refined_masks.append(refined_mask)
            else:
                # 如果阈值过高导致掩码完全消失，保留原始掩码但降低质量
                if original_mask.any():
                    quality = fg_prob[original_mask].mean() * 0.1  # 惩罚因子
                else:
                    # 如果原始掩码也为空，使用非常低的质量值
                    quality = torch.tensor(0.01, device=fg_prob.device, dtype=fg_prob.dtype)
                mask_qualities.append(quality)
                refined_masks.append(original_mask)
        
        # 确保所有掩码都被处理
        assert len(mask_qualities) == len(scores), f"Mismatch: {len(mask_qualities)} qualities vs {len(scores)} scores"
        assert len(refined_masks) == len(scores), f"Mismatch: {len(refined_masks)} masks vs {len(scores)} scores"
        
        mask_qualities = torch.stack(mask_qualities)
        refined_masks = torch.stack(refined_masks)
        
        # 检查 NaN
        if torch.isnan(mask_qualities).any():
            print(f"   ⚠️  Warning: NaN detected in mask_qualities, replacing with threshold")
            mask_qualities = torch.where(torch.isnan(mask_qualities), torch.full_like(mask_qualities, threshold), mask_qualities)
        
        print(f"   Pixel threshold ({threshold}): quality range [{mask_qualities.min().item():.3f}, {mask_qualities.max().item():.3f}]")
        
        # 应用温标校准
        calibrate = self.uam_cfg.get("calibrate", True)
        if calibrate:
            scale = 5.0  # 降低sigmoid斜率，避免过于激进
            calibrated_conf = torch.sigmoid((mask_qualities - threshold) * scale)
        else:
            calibrated_conf = mask_qualities
        
        # 确保张量大小匹配
        assert calibrated_conf.shape[0] == scores.shape[0], f"Mismatch: calibrated_conf {calibrated_conf.shape[0]} vs scores {scores.shape[0]}"
        assert calibrated_conf.shape[0] == pred_ious.shape[0], f"Mismatch: calibrated_conf {calibrated_conf.shape[0]} vs pred_ious {pred_ious.shape[0]}"
        
        # 重新加权分数，但保留所有掩码
        refined_scores = scores * calibrated_conf
        refined_pred = pred_ious * calibrated_conf
        
        print(f"   UAM merge: kept all {len(refined_scores)} masks with pixel-level refinement")
        
        cache["refined_masks"] = refined_masks
        cache["mask_qualities"] = mask_qualities
        cache["calibrated_conf"] = calibrated_conf

        return refined_scores, refined_masks.float(), labels, refined_pred

    def _uam_apply_crf(self, image, probs):
        if image is None:
            return probs
        refined = crf_refine(image, probs, n_iters=self.uam_cfg["crf"]["n_iters"], sxy=tuple(self.uam_cfg["crf"]["sxy"]), srgb=tuple(self.uam_cfg["crf"]["srgb"]))
        return refined

    def _uam_debug_dump(self, cache, img_info, final_masks):
        if not any(self._uam_debug.get(flag, False) for flag in ["save_probs", "save_entropy", "save_energy", "save_masks"]):
            return
        out_dir = self._uam_debug_dir()
        stem = Path(img_info.get("file_name", f"img_{img_info.get('id', 'unknown')}")).stem
        
        # 保存像素级 UAM 数据
        if self._uam_debug.get("save_probs"):
            pixel_probs = cache.get("pixel_probs", torch.tensor([]))
            if pixel_probs.numel() > 0:
                # 保存前景概率 (第1通道)
                fg_probs = pixel_probs[:, 1]  # (n_masks, h, w)
                torch.save(fg_probs.cpu(), out_dir / f"{stem}_fg_probs.pt")
        
        if self._uam_debug.get("save_entropy"):
            pixel_entropy = cache.get("pixel_entropy", torch.tensor([]))
            if pixel_entropy.numel() > 0:
                torch.save(pixel_entropy.squeeze(1).cpu(), out_dir / f"{stem}_entropy.pt")  # (n_masks, h, w)
        
        if self._uam_debug.get("save_energy"):
            mask_qualities = cache.get("mask_qualities", torch.tensor([]))
            calibrated_conf = cache.get("calibrated_conf", torch.tensor([]))
            torch.save({
                "mask_qualities": mask_qualities.cpu(),
                "calibrated_conf": calibrated_conf.cpu(),
            }, out_dir / f"{stem}_qualities.pt")
        
        if self._uam_debug.get("save_masks"):
            refined_masks = cache.get("refined_masks", final_masks)
            torch.save(refined_masks.cpu(), out_dir / f"{stem}_refined_masks.pt")
            torch.save(final_masks.cpu(), out_dir / f"{stem}_final_masks.pt")

