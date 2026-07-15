"""Merging utilities for uncertainty-aware mask refinement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch

from modules.uam_uncert_merge import bayes_merge, pixelwise_distribution, crf_refine
from utils.prior_maps import margin_prior, norm_prior, combine_priors


@dataclass
class MergerOutput:
    indices: torch.Tensor
    masks: torch.Tensor
    confidence: torch.Tensor
    aux: Dict[str, torch.Tensor]


class UAMMerger:
    def __init__(self, cfg: Dict):
        self.cfg = cfg

    def merge(
        self,
        mask_logits: torch.Tensor,
        image: Optional[torch.Tensor] = None,
        class_weights: Optional[torch.Tensor] = None,
        prior_maps: Optional[Dict[str, torch.Tensor]] = None,
    ) -> MergerOutput:
        stats = pixelwise_distribution(mask_logits, temperature=self.cfg.get("temperature", 1.0))
        probs = stats.probs.squeeze(0)
        entropy = stats.entropy.squeeze(0)
        energy = stats.energy.squeeze(0)

        prior_cfg = self.cfg.get("prior", {})
        prior_map = None
        if prior_cfg.get("enable", False):
            prior_candidates = []
            if prior_maps is not None:
                if prior_cfg.get("margin", {}).get("enable", False) and "margin" in prior_maps:
                    prior_candidates.append(prior_maps["margin"])
                if prior_cfg.get("norm", {}).get("enable", False) and "norm" in prior_maps:
                    prior_candidates.append(prior_maps["norm"])
                if "combined" in prior_maps:
                    prior_map = prior_maps["combined"]
            combine_mode = prior_cfg.get("combine", "multiply")
            if combine_mode == "augment":
                if prior_maps is not None and "combined" in prior_maps and "complement" in prior_maps:
                    # Use complement map to highlight not-yet-covered regions
                    prior_map = prior_maps.get("complement")
                elif prior_candidates:
                    prior_map = combine_priors(prior_candidates, mode="augment")
            else:
                if prior_map is None and prior_candidates:
                    prior_map = combine_priors(prior_candidates, mode=combine_mode)

        if image is not None and self.cfg.get("crf", {}).get("enable", False):
            probs = crf_refine(image, probs)

        threshold = self.cfg.get("threshold", 0.5)
        calibrate = self.cfg.get("calibrate", True)
        scale = self.cfg.get("calibrate_scale", 10.0)

        # Ensure prior_map matches logits batch dimension if provided
        if prior_map is not None:
            if prior_map.dim() == 3:  # (N, H, W)
                # Take mean across candidates for single inference
                prior_map = prior_map.mean(dim=0, keepdim=True)  # (1, H, W)
            elif prior_map.dim() == 2:  # (H, W)
                prior_map = prior_map.unsqueeze(0)  # (1, H, W)

        merge_res = bayes_merge(
            probs.unsqueeze(0),
            entropy.unsqueeze(0),
            threshold=threshold,
            calibrate=calibrate,
            class_weights=class_weights,
            prior_map=prior_map,
        )
        mask_valid = merge_res["mask"].squeeze(0)
        confidence_map = merge_res["confidence"].squeeze(0)
        labels_map = merge_res["label"].squeeze(0)

        if calibrate:
            max_prob = probs.max(dim=0)[0]
            confidence_map = torch.sigmoid((max_prob - threshold) * scale)

        kept_indices = []
        kept_masks = []
        kept_conf = []
        C = probs.shape[0]
        for idx in range(C):
            mask_idx = (labels_map == idx) & mask_valid
            if not mask_idx.any():
                continue
            kept_indices.append(idx)
            kept_masks.append(mask_idx)
            kept_conf.append(confidence_map[mask_idx].mean())

        if len(kept_indices) == 0:
            device = mask_logits.device
            empty = torch.zeros((0,), device=device)
            return MergerOutput(
                indices=empty.long(),
                masks=torch.zeros((0,) + mask_logits.shape[-2:], dtype=torch.bool, device=device),
                confidence=empty,
                aux={
                    "entropy": entropy,
                    "energy": energy,
                    "prob": probs,
                    "confidence_map": confidence_map,
                    "mask_valid": mask_valid,
                },
            )

        device = mask_logits.device
        indices = torch.tensor(kept_indices, dtype=torch.long, device=device)
        masks = torch.stack(kept_masks, dim=0).to(dtype=torch.bool)
        confidence = torch.stack(kept_conf)

        return MergerOutput(
            indices=indices,
            masks=masks,
            confidence=confidence,
            aux={
                "entropy": entropy,
                "energy": energy,
                "prob": probs,
                "confidence_map": confidence_map,
                "mask_valid": mask_valid,
            },
        )


def build_merger(cfg: Dict) -> UAMMerger:
    return UAMMerger(cfg)


