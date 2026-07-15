import pytest
import torch

from modules.uam_uncert_merge import pixelwise_distribution, bayes_merge, crf_refine


def _make_logits(num_masks=3, height=16, width=16):
    logits = torch.randn(num_masks, height, width, dtype=torch.float32)
    return logits


def test_pixelwise_distribution_basic():
    logits = _make_logits(4)
    stats = pixelwise_distribution(logits, temperature=0.5)
    assert stats.logits.shape == logits.shape
    assert stats.probs.shape == logits.shape
    probs_sum = stats.probs.sum(dim=0)
    assert torch.allclose(probs_sum, torch.ones_like(probs_sum), atol=1e-5)


def test_bayes_merge_confidence_threshold():
    logits = torch.randn(1, 3, 10, 10)
    entropy = torch.zeros_like(logits[:, :1])
    result = bayes_merge(logits, entropy, threshold=0.3)
    mask = result["mask"]
    assert mask.dtype == torch.bool
    assert mask.shape[-2:] == (10, 10)
    assert torch.all(result["confidence"] >= 0.0)


@pytest.mark.skipif("pydensecrf" not in globals() or globals().get("dcrf") is None, reason="pydensecrf not available")
def test_crf_refine_available():
    image = torch.rand(3, 8, 8)
    probs = torch.rand(2, 8, 8)
    probs = probs / probs.sum(dim=0, keepdim=True)
    refined = crf_refine(image, probs)
    assert refined.shape == probs.shape
    assert torch.allclose(refined.sum(dim=0), torch.ones_like(refined[0]), atol=1e-3)


def test_crf_refine_fallback():
    image = torch.rand(3, 8, 8)
    probs = torch.rand(2, 8, 8)
    probs = probs / probs.sum(dim=0, keepdim=True)
    refined = crf_refine(image, probs)
    assert refined.shape == probs.shape


@pytest.mark.parametrize("threshold", [0.2, 0.5, 0.8])
def test_merge_overlapping_mask(threshold):
    height, width = 32, 32
    logits = torch.zeros(1, 2, height, width)
    y = torch.linspace(0, 1, steps=height).unsqueeze(1).repeat(1, width)
    logits[0, 0] = y
    logits[0, 1] = 1 - y
    entropy = torch.zeros(1, 1, height, width)
    result = bayes_merge(logits, entropy, threshold=threshold)
    mask = result["mask"].squeeze()
    assert mask.shape == (height, width)
    assert mask.any() or threshold >= 0.8


