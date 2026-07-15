"""CLIP prototype helpers for Prototype Purification.

提供图像/文本原型获取接口，优先使用 OpenAI 官方 `clip`，否则回退到
`transformers.CLIPModel`。若运行环境缺少 CLIP 依赖，将抛出 `ImportWarning`
让上层逻辑降级到无 CLIP 模式。
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable, Dict, Iterable, List, Optional

import torch
import torch.nn.functional as F

# 尝试导入 OpenAI CLIP
try:  # pragma: no cover - optional dependency
    import clip as openai_clip  # type: ignore
except Exception:  # pragma: no cover - import guard
    openai_clip = None  # type: ignore

# 尝试导入 HuggingFace CLIP
try:  # pragma: no cover - optional dependency
    from transformers import CLIPModel, CLIPTokenizer  # type: ignore
except Exception:  # pragma: no cover - import guard
    CLIPModel = None  # type: ignore
    CLIPTokenizer = None  # type: ignore


_MODEL_CACHE: Dict[str, torch.nn.Module] = {}
_TOKENIZER_CACHE: Dict[str, Callable] = {}


def _default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_hf_model_name(model_name: str) -> str:
    if "/" in model_name:
        return model_name
    mapping = {
        "vit-b/16": "openai/clip-vit-base-patch16",
        "vit-b/32": "openai/clip-vit-base-patch32",
        "vit-l/14": "openai/clip-vit-large-patch14",
        "vit-l/14@336px": "openai/clip-vit-large-patch14-336",
    }
    key = model_name.lower()
    return mapping.get(key, "openai/clip-vit-base-patch16")


def _ensure_backend_available() -> str:
    if openai_clip is not None:
        return "openai"
    if CLIPModel is not None and CLIPTokenizer is not None:
        return "hf"
    raise ImportWarning("CLIP backend is not available. Please install 'clip' or 'transformers'.")


@lru_cache(maxsize=None)
def _get_openai_clip_model(model_name: str) -> torch.nn.Module:
    if openai_clip is None:  # pragma: no cover - runtime guard
        raise ImportWarning("OpenAI CLIP is not installed.")
    device = _default_device()
    model, _ = openai_clip.load(model_name, device=device, jit=False)
    model.eval()
    return model


@lru_cache(maxsize=None)
def _get_hf_clip_model(model_name: str) -> torch.nn.Module:
    if CLIPModel is None or CLIPTokenizer is None:  # pragma: no cover - runtime guard
        raise ImportWarning("Transformers CLIP is not installed.")
    resolved = _resolve_hf_model_name(model_name)
    model = CLIPModel.from_pretrained(resolved)
    model.to(_default_device()).eval()
    return model


@lru_cache(maxsize=None)
def _get_hf_tokenizer(model_name: str):
    if CLIPTokenizer is None:  # pragma: no cover - runtime guard
        raise ImportWarning("Transformers CLIP tokenizer is not installed.")
    resolved = _resolve_hf_model_name(model_name)
    return CLIPTokenizer.from_pretrained(resolved)


def get_clip_image_proto(image_feats: List[torch.Tensor]) -> torch.Tensor:
    """根据一组 CLIP 图像特征生成单位范数原型。"""

    if not image_feats:
        raise ValueError("image_feats 不能为空。")

    feats: List[torch.Tensor] = [f.detach().to(dtype=torch.float32) for f in image_feats]
    stacked = torch.stack(feats, dim=0)
    proto = stacked.mean(dim=0)
    return F.normalize(proto, dim=0)


def get_clip_text_proto(text: str, model_name: str = "ViT-B/16") -> torch.Tensor:
    """编码文本提示并返回单位范数原型。"""

    backend = _ensure_backend_available()
    device = _default_device()

    if backend == "openai":
        model = _get_openai_clip_model(model_name)
        assert openai_clip is not None  # mypy guard
        with torch.inference_mode():
            tokens = openai_clip.tokenize([text]).to(device)
            text_features = model.encode_text(tokens)
        proto = text_features.squeeze(0).to(dtype=torch.float32)
    else:
        model = _get_hf_clip_model(model_name)
        tokenizer = _get_hf_tokenizer(model_name)
        inputs = tokenizer([text], return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            text_features = model.get_text_features(**inputs)
        proto = text_features.squeeze(0).to(dtype=torch.float32)

    return F.normalize(proto, dim=0)


def build_clip_hooks(
    texts: Optional[Dict[int, str]] = None,
    class_names: Optional[Iterable[str]] = None,
    image_feats: Optional[Dict[int, List[torch.Tensor]]] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    model_name: str = "ViT-B/16",
) -> Dict[str, Callable[[int], Optional[torch.Tensor]]]:
    """构建 CLIP 图像/文本原型获取钩子。"""

    class_names_list = list(class_names) if class_names is not None else None
    needs_backend = bool(
        (texts and len(texts) > 0)
        or (class_names_list and len(class_names_list) > 0)
        or (image_feats and len(image_feats) > 0)
    )

    if needs_backend:
        _ensure_backend_available()

    target_device = device if device is not None else _default_device()
    target_dtype = dtype if dtype is not None else torch.float32

    hooks: Dict[str, Callable[[int], Optional[torch.Tensor]]] = {}
    text_cache: Dict[int, torch.Tensor] = {}
    image_cache: Dict[int, torch.Tensor] = {}

    if texts or class_names_list:

        def _text_hook(cls_id: int) -> Optional[torch.Tensor]:
            if cls_id in text_cache:
                return text_cache[cls_id]

            phrase = None
            if texts and cls_id in texts:
                phrase = texts[cls_id]
            elif class_names_list and 0 <= cls_id < len(class_names_list):
                phrase = class_names_list[cls_id]

            if not phrase:
                return None

            proto = get_clip_text_proto(phrase, model_name=model_name)
            proto = proto.to(device=target_device, dtype=target_dtype)
            text_cache[cls_id] = proto
            return proto

        hooks["txt"] = _text_hook

    if image_feats:

        def _img_hook(cls_id: int) -> Optional[torch.Tensor]:
            if cls_id in image_cache:
                return image_cache[cls_id]

            feats = image_feats.get(cls_id)
            if not feats:
                return None

            proto = get_clip_image_proto(feats)
            proto = proto.to(device=target_device, dtype=target_dtype)
            image_cache[cls_id] = proto
            return proto

        hooks["img"] = _img_hook

    return hooks


__all__ = [
    "get_clip_image_proto",
    "get_clip_text_proto",
    "build_clip_hooks",
]


