import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
import numpy as np

from zods_rs.models.encoder_factory import build_encoder


def load_image(image_path: Path, target_size: int) -> torch.Tensor:
    tfm = transforms.Compose(
        [
            transforms.Resize((target_size, target_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ]
    )
    img = Image.open(image_path).convert("RGB")
    return tfm(img).unsqueeze(0)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description="Minimal DINOv3 smoke test (Windows friendly)")
    parser.add_argument(
        "--image",
        type=str,
        default=str(Path("notebooks") / "images" / "truck.jpg"),
        help="Path to an RGB image",
    )
    parser.add_argument(
        "--hf_model_name",
        type=str,
        default="facebook/dinov3-large",
        help="Hugging Face model id for DINOv3",
    )
    parser.add_argument(
        "--local_safetensors",
        type=str,
        default="",
        help="Optional path to local .safetensors weights (overrides HF weights if provided)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="cuda|cpu|auto",
    )
    args = parser.parse_args()

    if args.device == "cuda":
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder_cfg = {
        "name": "dinov3_large",
        "hf_model_name": args.hf_model_name,
        "local_safetensors": args.local_safetensors or None,
        # img_size/patch_size will be inferred from HF config if omitted
    }

    print("[SmokeTest] Building DINOv3 encoder ...")
    encoder = build_encoder(encoder_cfg, encoder_ckpt_path=None)
    encoder.model.to(device)

    img_path = Path(args.image)
    assert img_path.exists(), f"Image not found: {img_path}"
    print(f"[SmokeTest] Loading image: {img_path}")
    inp = load_image(img_path, target_size=encoder.img_size).to(device)

    print("[SmokeTest] Running encoder forward ...")
    tokens = encoder.tokens_from_images(inp, normalize=True)  # [1, N, C]
    tokens = torch.nn.functional.normalize(tokens, p=2, dim=-1)

    print("[SmokeTest] Done.")
    print(f" - device: {device}")
    print(f" - img_size: {encoder.img_size}, patch_size: {encoder.patch_size}")
    print(f" - tokens shape: {tuple(tokens.shape)} (B, N, C)")
    print(f" - feat_dim (C): {encoder.feat_dim}")
    # quick sanity: cosine self-sim of first token vector
    sim = (tokens[0, 0] @ tokens[0, 0]).clamp(min=0.0, max=1.0).item()
    print(f" - self cosine(sim) of first token: {sim:.4f}")

    # Visualize token L2-norm as a heatmap and save alongside original image
    B, N, C = tokens.shape
    H = W = int(np.sqrt(N))
    if H * W != N:
        print("[SmokeTest] Skip heatmap save: tokens cannot be reshaped to square grid.")
        return
    norm_map = tokens.norm(p=2, dim=-1).reshape(H, W)
    norm_map = (norm_map - norm_map.min()) / (norm_map.max() - norm_map.min() + 1e-8)
    norm_map = torch.nn.functional.interpolate(norm_map.unsqueeze(0).unsqueeze(0), size=(encoder.img_size, encoder.img_size), mode="bilinear", align_corners=False)[0,0]

    # load original image again (uint8) and overlay heatmap
    img = Image.open(args.image).convert("RGB").resize((encoder.img_size, encoder.img_size))
    img_np = np.array(img).astype(np.float32)
    heat = (norm_map.cpu().numpy() * 255.0).astype(np.uint8)
    heat_color = np.stack([heat * 0, heat, heat], axis=-1)  # blue-ish
    overlay = (0.6 * img_np + 0.4 * heat_color).clip(0, 255).astype(np.uint8)

    out_path = Path(args.image).with_suffix("")
    out_path = Path(str(out_path) + "_dinov3_heatmap.png")
    Image.fromarray(overlay).save(out_path)
    print(f"[SmokeTest] Saved annotated image to: {out_path}")


if __name__ == "__main__":
    main()


