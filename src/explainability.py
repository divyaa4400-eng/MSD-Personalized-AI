from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch

from .synthetic_data import MODALITY_NAMES


def gradcam_for_sample(model, batch: Dict[str, torch.Tensor], modality: str, device: torch.device) -> np.ndarray:
    if modality not in {"mri", "ct", "xray"}:
        raise ValueError("Grad-CAM is implemented for mri, ct, or xray")
    encoder = getattr(model, f"{modality}_encoder")
    target_layer = encoder.final_conv
    activations = []
    gradients = []

    def fwd_hook(_module, _inp, output):
        activations.append(output)

    def bwd_hook(_module, _grad_in, grad_out):
        gradients.append(grad_out[0])

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)
    try:
        model.eval()
        model.zero_grad(set_to_none=True)
        inputs = {
            "mri": batch["mri"].to(device),
            "ct": batch["ct"].to(device),
            "xray": batch["xray"].to(device),
            "clinical": batch["clinical"].to(device),
            "biomechanical": batch["biomechanical"].to(device),
            "modality_mask": batch["modality_mask"].to(device),
        }
        out = model(**inputs)
        out["risk"].sum().backward()
        act = activations[-1].detach()
        grad = gradients[-1].detach()
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * act).sum(dim=1, keepdim=True))
        cam = torch.nn.functional.interpolate(cam, size=batch[modality].shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min()
        cam /= max(cam.max(), 1e-8)
        return cam.astype(np.float32)
    finally:
        h1.remove()
        h2.remove()


@torch.no_grad()
def grouped_permutation_attribution(model, loader, device: torch.device, repeats: int = 5, seed: int = 0) -> Dict[str, float]:
    """Estimate grouped importance as increase in risk-prediction MSE after permutation."""
    model.eval()
    cached = []
    for batch in loader:
        cached.append({k: v.clone() for k, v in batch.items()})
    if not cached:
        return {name: 0.0 for name in MODALITY_NAMES}

    def mse_for_batches(batches):
        sq = []
        for batch in batches:
            out = model(
                batch["mri"].to(device),
                batch["ct"].to(device),
                batch["xray"].to(device),
                batch["clinical"].to(device),
                batch["biomechanical"].to(device),
                batch["modality_mask"].to(device),
            )
            sq.extend(((out["risk"].cpu() - batch["risk"]) ** 2).numpy().tolist())
        return float(np.mean(sq))

    baseline = mse_for_batches(cached)
    rng = np.random.default_rng(seed)
    raw = {}
    key_map = {"mri": "mri", "ct": "ct", "xray": "xray", "clinical": "clinical", "biomechanical": "biomechanical"}
    for name, key in key_map.items():
        increases = []
        for _ in range(repeats):
            perturbed = [{k: v.clone() for k, v in batch.items()} for batch in cached]
            full = torch.cat([b[key] for b in perturbed], dim=0)
            perm = torch.from_numpy(rng.permutation(len(full)))
            shuffled = full[perm]
            start = 0
            for b in perturbed:
                n = len(b[key])
                b[key] = shuffled[start:start+n]
                start += n
            increases.append(max(0.0, mse_for_batches(perturbed) - baseline))
        raw[name] = float(np.mean(increases))
    total = sum(raw.values())
    if total <= 0:
        return {k: 0.0 for k in raw}
    return {k: float(v / total) for k, v in raw.items()}
