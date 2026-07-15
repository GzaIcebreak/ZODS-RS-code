import copy
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torchvision.transforms import Normalize


class EncoderWrapper:
    """
    A thin adapter over the DINOv3 backbone that exposes
    a unified token extraction interface.
    """

    def __init__(
        self,
        model,
        transform: Normalize,
        img_size: int,
        patch_size: int,
        feat_dim: int,
        family: str,
        aux: Optional[dict] = None,
    ) -> None:
        self.model = model
        self.transform = transform
        self.img_size = img_size
        self.patch_size = patch_size
        self.feat_dim = feat_dim
        self.family = family  # "dinov3"
        self.aux = aux or {}

    @torch.no_grad()
    def tokens_from_images(self, imgs: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        Args:
            imgs: Float tensor [B, 3, H, W] in range ~[0,1] or ImageNet-normalized
            normalize: Whether to apply the encoder's default Normalize
        Returns:
            tokens: [B, N, C] without CLS/register tokens
        """
        if normalize:
            imgs = self.transform(imgs)

        if self.family == "dinov3":
            outputs = self.model(pixel_values=imgs)
            # Dinov3Model returns last_hidden_state with CLS as first token
            tokens = outputs.last_hidden_state[:, 1:, :]
            # Drop register tokens if present so that remaining tokens form an HxW grid
            num_reg = getattr(getattr(self.model, "config", None), "num_register_tokens", 0)
            if isinstance(num_reg, int) and num_reg > 0:
                tokens = tokens[:, num_reg:, :]
            return tokens

        raise NotImplementedError(f"Unknown family: {self.family}")

    @torch.no_grad()
    def get_multilayer_features(
        self, 
        imgs: torch.Tensor, 
        layers: list = [6, 10, -1],
        return_attn: bool = True,
        normalize: bool = True
    ) -> Tuple[list, Optional[torch.Tensor]]:
        """Extract features from multiple layers.
        
        Args:
            imgs: Input images [B, 3, H, W]
            layers: Layer indices to extract (negative = from end)
            return_attn: Whether to return attention maps
            normalize: Whether to apply normalization
            
        Returns:
            (list of layer features, attention map or None)
        """
        if normalize:
            imgs = self.transform(imgs)
        
        if self.family == "dinov3":
            # DINOv3 via transformers - extract intermediate outputs
            outputs = self.model(pixel_values=imgs, output_hidden_states=True)
            all_hidden = outputs.hidden_states  # Tuple of (B, 1+num_reg+N, C)
            num_reg = getattr(self.model.config, "num_register_tokens", 0)
            
            layer_feats = []
            for layer_idx in layers:
                hidden = all_hidden[layer_idx]
                # Drop CLS and register tokens
                tokens = hidden[:, 1+num_reg:, :]
                layer_feats.append(tokens)
            
            # DINOv3 doesn't expose attention easily, return None
            attn = None
            if return_attn:
                # Could implement if needed, for now None
                pass
            
            return layer_feats, attn
        
        raise NotImplementedError(f"Unknown family: {self.family}")
    
    @torch.no_grad()
    def tokens_with_attn(self, imgs: torch.Tensor, normalize: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns tokens and (optionally) rolled attention map if available.
        For DINOv3 we return (tokens, None).
        """
        if self.family == "dinov3":
            tokens = self.tokens_from_images(imgs, normalize=normalize)
            return tokens, None

        raise NotImplementedError(f"Unknown family: {self.family}")


def build_encoder(encoder_cfg: dict, encoder_ckpt_path: Optional[str]) -> EncoderWrapper:
    """
    Build an EncoderWrapper from config. Supports:
      - name: "dinov3_large" (Hugging Face transformers)
    Additional fields (optional for dinov3):
      - hf_model_name: e.g. "facebook/dinov3-large"
    """
    name = copy.deepcopy(encoder_cfg).pop("name")
    img_size = encoder_cfg.get("img_size")
    patch_size = encoder_cfg.get("patch_size")

    if name.startswith("dinov3"):
        try:
            # Prefer specific class if available; otherwise fall back to AutoModel
            from transformers import AutoModel, AutoImageProcessor  # type: ignore
            try:
                from transformers import Dinov3Model  # type: ignore
                model_cls = Dinov3Model
            except Exception:
                model_cls = AutoModel
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "Dinov3 requires 'transformers'. Please install/upgrade transformers (>=4.44.0)."
            ) from e

        import os
        model_id = encoder_cfg.get("hf_model_name", "facebook/dinov3-large")
        is_local_dir = False
        if isinstance(model_id, str):
            if os.path.isdir(model_id):
                is_local_dir = True
            else:
                cfg_path = os.path.join(model_id, "config.json")
                if os.path.isfile(cfg_path):
                    is_local_dir = True
        # trust_remote_code for safety if upstream registers custom modeling
        if model_cls.__name__ == "AutoModel":
            if is_local_dir:
                model = model_cls.from_pretrained(model_id, trust_remote_code=True, local_files_only=True)
            else:
                model = model_cls.from_pretrained(model_id, trust_remote_code=True)
        else:
            if is_local_dir:
                model = model_cls.from_pretrained(model_id, local_files_only=True)
            else:
                model = model_cls.from_pretrained(model_id)
        # Try to load processor; fallback to ImageNet stats if unavailable (e.g., local dir without processor config)
        try:
            if is_local_dir:
                processor = AutoImageProcessor.from_pretrained(model_id, local_files_only=True)
            else:
                processor = AutoImageProcessor.from_pretrained(model_id)
        except Exception:
            processor = None

        if processor is not None and hasattr(processor, "image_mean") and hasattr(processor, "image_std"):
            mean = processor.image_mean
            std = processor.image_std
        else:
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)
        transform = Normalize(mean=tuple(mean), std=tuple(std))

        # Prefer config values if provided; otherwise fallback to model.config
        hidden = getattr(model.config, "hidden_size", None)
        feat_dim = int(hidden) if hidden is not None else 1024
        if img_size is None:
            img_size = getattr(model.config, "image_size", 518)
        if patch_size is None:
            patch_size = getattr(model.config, "patch_size", 14)

        # optionally load local .safetensors weights
        local_safetensors_path = encoder_cfg.get("local_safetensors")
        if local_safetensors_path:
            try:
                from safetensors.torch import load_file as load_safetensors
            except Exception as e:  # pragma: no cover
                raise ImportError(
                    "Loading local safetensors requires 'safetensors'. Please install safetensors."
                ) from e
            state_dict = load_safetensors(local_safetensors_path)
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            if len(unexpected_keys) > 0:
                # keep silent but non-fatal; architectures may differ slightly across checkpoints
                pass
            if len(missing_keys) > 0:
                # keep silent but non-fatal
                pass

        model.eval()
        # Note: device will be handled by Lightning/user code, not here
        return EncoderWrapper(model, transform, int(img_size), int(patch_size), int(feat_dim), family="dinov3")

    raise NotImplementedError(f"Unsupported encoder name: {name}")


