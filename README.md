# ZODS-RS: Zero-training Oriented Detection and Segmentation for Remote Sensing

Official implementation of **ZODS-RS** (CVPR 2026 Findings) — training-free few-shot instance segmentation for remote sensing imagery, built on SAM2 and DINOv3.

Zuan Gu, Tianhan Gao, Langxu Zhao — Northeastern University, Shenyang, China

[Project Page](https://gzaicebreak.github.io/zods-rs/) | Paper (arXiv coming soon)

Given as few as one annotated reference image per category, ZODS-RS segments all instances of that category in unseen target images — without any training or fine-tuning.

## Highlights

- **Training-free**: no gradient updates; a memory bank of reference features drives the matching.
- **DINOv3 encoder**: multi-layer, scale-aware semantic matching (SEM / R-SEM) with consistency-weighted layer aggregation (CWLA).
- **Prototype purification (PP)**: robust prototype estimation (Tyler's M-estimator / Sinkhorn OT variants) for clean reference memories.
- **Uncertainty-aware merging (UAM)**: Bayesian mask merging with confidence priors and optional CRF refinement.
- **Flexible export**: binary masks, COCO JSON, oriented bounding boxes (OBB), and per-instance visualizations.

## Installation

```bash
conda env create -f environment.windows.yml   # Windows
# or
conda env create -f environment.yml           # Linux
conda activate zods-rs-win

# Install PyTorch matching your CUDA version, e.g.:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install the package (SAM2 CUDA extension is optional)
pip install -e .
cd dinov2 && pip install -e . && cd ..
```

## Model Weights

Weights are **not** included in this repository. Download them separately:

| Model | File | Location |
|---|---|---|
| SAM2 (Hiera-L) | `sam2_hiera_large.pt` | `./checkpoints/` |
| DINOv3 ViT-L/16 | `model.safetensors` (HF format) | `./dinov3-vitl16-local/` |

- SAM2: https://github.com/facebookresearch/segment-anything-2
- DINOv3: https://github.com/facebookresearch/dinov3 (or the Hugging Face hub)

## Data Layout

```
data/<dataset>/
├── images/                                   # reference + target images
└── annotations/
    ├── custom_references_with_segm_sam21.json  # reference annotations (COCO format)
    ├── custom_references_with_segm.pkl         # memory sampling file
    └── custom_targets.json                     # target image list (COCO format)
```

## Usage

Two example instance configurations are provided under `zods_rs/pl_configs/`:

- `build_dinov3.yaml` — building extraction
- `ship_dinov3.yaml` — ship detection (FAIR1M-style)

The pipeline has three stages, driven by the same config:

```bash
CONFIG=zods_rs/pl_configs/ship_dinov3.yaml

# 1) Fill the memory bank with reference features
python run_lightening.py test --config $CONFIG \
    --model.test_mode fill_memory \
    --out_path ./tmp_ckpts/ship/ship_refs_memory.pth

# 2) Post-process the memory bank (prototype purification)
python run_lightening.py test --config $CONFIG \
    --model.test_mode postprocess_memory \
    --ckpt_path ./tmp_ckpts/ship/ship_refs_memory.pth \
    --out_path ./tmp_ckpts/ship/ship_refs_memory_postprocessed.pth

# 3) Run inference on target images
python run_lightening.py test --config $CONFIG \
    --model.test_mode test \
    --ckpt_path ./tmp_ckpts/ship/ship_refs_memory_postprocessed.pth
```

Or use the helper script for a one-shot end-to-end run on a new image:

```bash
python scripts/process_and_test.py path/to/image.jpg
```

Results (visualizations, COCO JSON, binary masks, per-instance crops) are written to `./results_analysis/<dataset>/`.

## Key Configuration Blocks

| Block | Purpose |
|---|---|
| `sem` | Scale-aware semantic matching (multi-scale / multi-layer DINOv3, optional rotation equivariance) |
| `sem.cwla` | Consistency-weighted layer aggregation |
| `memory_bank_cfg.pp` | Prototype purification (robust / OT estimators, clustering) |
| `uam` | Uncertainty-aware mask merging (priors, negative prototypes, CRF) |
| `eval.out_format` | Output format: `mask`, `obb`, or `polygon` |

## Repository Structure

```
zods_rs/            # main package (models, datasets, Lightning wrappers, configs)
modules/            # core algorithm modules (SEM, PP, UAM)
utils/              # priors, calibration, CLIP adapters
sam2/, sam2_configs/  # SAM2 (Meta AI, Apache 2.0)
dinov2/             # DINOv2/DINOv3 support code (Meta AI)
scripts/            # data preparation, batch processing, visualization tools
tests/              # unit tests for SEM / PP / UAM
```

## Citation

```bibtex
@inproceedings{gu2026zodsrs,
  title     = {{ZODS-RS}: Zero-training Oriented Detection and Segmentation for Remote Sensing},
  author    = {Gu, Zuan and Gao, Tianhan and Zhao, Langxu},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision
               and Pattern Recognition (CVPR) Findings},
  year      = {2026},
  address   = {Nashville, TN, USA},
  note      = {To appear}
}
```

## License

This project builds on SAM2 (Apache 2.0) and DINOv2/DINOv3 (Meta AI licenses). See the respective subdirectories for details.
