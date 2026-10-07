"""Bounded, deterministic IMAGE/MASK input validation. No ComfyUI imports."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch

VERSION = "0.1.0"
VERDICTS = {"FAIL": 0, "REVIEW": 1, "PASS": 2}
CONVENTIONS = ("white_selects_edit", "white_protects", "white_selects_source")
HARD_MAX_PIXELS = 16_777_216
HARD_MAX_FRAMES = 256


@dataclass(frozen=True)
class Policy:
    white_convention: str = "white_selects_edit"
    allow_batch1_broadcast: bool = False
    active_threshold: float = 0.5
    min_coverage: float = 0.0
    max_coverage: float = 1.0
    extreme_policy: str = "review"
    max_frames: int = 64
    max_total_pixels: int = 8_388_608

    def validate(self):
        if self.white_convention not in CONVENTIONS:
            raise ValueError("Unknown white_convention")
        if type(self.allow_batch1_broadcast) is not bool:
            raise ValueError("allow_batch1_broadcast must be BOOLEAN")
        values = (self.active_threshold, self.min_coverage, self.max_coverage)
        if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) for v in values):
            raise ValueError("Threshold and coverage bounds must be finite numbers")
        if not 0 < self.active_threshold <= 1 or not 0 <= self.min_coverage <= self.max_coverage <= 1:
            raise ValueError("Require 0 < active_threshold <= 1 and 0 <= min_coverage <= max_coverage <= 1")
        if self.extreme_policy not in ("review", "fail", "allow"):
            raise ValueError("extreme_policy must be review, fail or allow")
        if type(self.max_frames) is not int or not 1 <= self.max_frames <= HARD_MAX_FRAMES:
            raise ValueError(f"max_frames must be between 1 and {HARD_MAX_FRAMES}")
        if type(self.max_total_pixels) is not int or not 1 <= self.max_total_pixels <= HARD_MAX_PIXELS:
            raise ValueError(f"max_total_pixels must be between 1 and {HARD_MAX_PIXELS}")


def _description(value):
    if not isinstance(value, torch.Tensor):
        return {"type": type(value).__name__, "shape": None}
    return {"type": "Tensor", "shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device)}


def evaluate(images, mask, policy: Policy):
    """Return JSON-safe evidence and an explicitly broadcast mask, or None."""
    policy.validate()
    report = {
        "schema_version": "1.0", "package_version": VERSION,
        "inputs": {"image": _description(images), "mask": _description(mask)},
        "policy": asdict(policy), "issues": [], "frames": [],
        "broadcast_applied": False, "measurements_available": False,
        "notes": [
            "White convention is configured, never inferred. Polarity and semantic intent are not automatically detected.",
            "Coordinates refer to the IMAGE pixel canvas, not latent cells or a cropped mask canvas.",
            "Coverage counts mask values >= active_threshold. Bboxes use exclusive right/bottom edges.",
            "Inputs are never resized, inverted, clipped or normalized.",
        ],
    }

    def issue(code, message, severity="FAIL", frame=None):
        item = {"code": code, "severity": severity, "message": message}
        if frame is not None:
            item["frame_index"] = frame
        report["issues"].append(item)

    def finish(effective=None):
        severity = {i["severity"] for i in report["issues"]}
        report["verdict"] = "FAIL" if "FAIL" in severity else "REVIEW" if "REVIEW" in severity else "PASS"
        report["verdict_code"] = VERDICTS[report["verdict"]]
        return report, effective

    for label, value in (("image", images), ("mask", mask)):
        if not isinstance(value, torch.Tensor):
            issue(label + "_type", label + " must be a torch.Tensor")
    if report["issues"]:
        return finish()
    image_shape_ok = images.ndim == 4 and min(images.shape[:3], default=0) > 0 and images.shape[-1] in (3, 4)
    mask_shape_ok = mask.ndim == 3 and min(mask.shape, default=0) > 0
    if not image_shape_ok:
        issue("image_shape", "Expected IMAGE [batch,height,width,3 or 4] with nonempty dimensions")
    if not mask_shape_ok:
        issue("mask_shape", "Expected MASK [batch,height,width]; a 2D mask must be explicitly given a batch axis upstream")
    # Bound work before reductions, device transfers or evidence allocation.
    if images.numel() > policy.max_total_pixels * 4 or mask.numel() > policy.max_total_pixels:
        issue("resource_limit", "Input element count exceeds the configured pixel work budget")
    if image_shape_ok and (images.shape[0] > policy.max_frames or math.prod(images.shape[:3]) > policy.max_total_pixels):
        issue("resource_limit", "IMAGE batch exceeds max_frames or max_total_pixels")
    if mask_shape_ok and mask.shape[0] > policy.max_frames:
        issue("resource_limit", "MASK batch exceeds max_frames")
    if report["issues"]:
        return finish()
    if images.shape[1:3] != mask.shape[1:3]:
        issue("canvas_mismatch", "MASK height/width must match the IMAGE pixel canvas exactly")
    same_batch = images.shape[0] == mask.shape[0]
    broadcast = images.shape[0] > 1 and mask.shape[0] == 1 and policy.allow_batch1_broadcast
    if not same_batch and not broadcast:
        issue("batch_mismatch", "Batch counts differ; only an explicitly allowed MASK batch of 1 can broadcast")
    valid_numbers = {}
    for label, value in (("image", images), ("mask", mask)):
        if not value.is_floating_point():
            issue(label + "_dtype", label + " must contain floating-point values")
            valid_numbers[label] = False
        elif value.device.type == "meta":
            issue(label + "_device", "Meta tensors do not contain inspectable pixel values")
            valid_numbers[label] = False
        elif not bool(torch.isfinite(value).all()):
            issue(label + "_nonfinite", label + " contains NaN or infinity")
            valid_numbers[label] = False
        else:
            minimum, maximum = float(value.min()), float(value.max())
            report["inputs"][label].update({"minimum": minimum, "maximum": maximum})
            valid_numbers[label] = minimum >= 0 and maximum <= 1
            if not valid_numbers[label]:
                issue(label + "_range", label + " values must be within [0,1]; no implicit normalization")
    report["image_values_valid"] = valid_numbers.get("image", False)
    if report["issues"]:
        return finish()
    effective = mask.expand(images.shape[0], -1, -1) if broadcast else mask
    report["broadcast_applied"] = broadcast
    report["effective_mask_shape"] = list(effective.shape)
    report["measurements_available"] = True
    for index, frame in enumerate(effective):
        selected = frame >= policy.active_threshold
        count = int(selected.sum())
        pixels = frame.numel()
        coverage = count / pixels
        bbox = None
        if count:
            # Row/column reductions avoid materializing every selected coordinate.
            rows = torch.where(selected.any(dim=1))[0]
            columns = torch.where(selected.any(dim=0))[0]
            bbox = [int(columns[0]), int(rows[0]), int(columns[-1]) + 1, int(rows[-1]) + 1]
        all_empty = bool((frame == 0).all())
        all_full = bool((frame == 1).all())
        report["frames"].append({
            "frame_index": index, "source_mask_index": 0 if broadcast else index,
            "selected_pixels": count, "canvas_pixels": pixels, "coverage": coverage,
            "mean_mask_value": float(frame.mean(dtype=torch.float32)), "bbox_xyxy": bbox,
            "all_empty": all_empty, "all_full": all_full,
            "threshold_selects_none": count == 0, "threshold_selects_all": count == pixels,
        })
        if coverage < policy.min_coverage or coverage > policy.max_coverage:
            issue("coverage_outside_bounds", "Selected coverage falls outside the configured inclusive bounds", frame=index)
        if count in (0, pixels) and policy.extreme_policy != "allow":
            severity = "REVIEW" if policy.extreme_policy == "review" else "FAIL"
            issue("empty_selection" if count == 0 else "full_selection", "The configured threshold selects no pixels" if count == 0 else "The configured threshold selects every pixel", severity, index)
    return finish(effective)


def evidence_overlay(images, effective_mask, report, opacity=0.4):
    """CPU float32 evidence only; never modify or replace gate inputs."""
    if not math.isfinite(opacity) or not 0 <= opacity <= 1:
        raise ValueError("overlay_opacity must be within [0,1]")
    if not report.get("image_values_valid"):
        report["overlay"] = {"kind": "placeholder", "reason": "IMAGE invalid, nonfinite, or work budget exceeded; no input canvas rendered"}
        result = torch.zeros((1, 32, 32, 3), dtype=torch.float32)
        result[:, :, :, 0] = 0.55
        return result
    output = images.detach().to(device="cpu", dtype=torch.float32).clone()
    report["overlay"] = {"kind": "image_canvas", "mask_tint_applied": effective_mask is not None, "opacity": opacity}
    if effective_mask is not None:
        color = (0.65, 0.3, 1.0) if report["policy"]["white_convention"] == "white_protects" else (0.0, 0.85, 1.0)
        tint = torch.tensor(color, dtype=torch.float32)
        for index, frame in enumerate(effective_mask):
            alpha = frame.detach().to(device="cpu", dtype=torch.float32).unsqueeze(-1) * opacity
            output[index, :, :, :3] = output[index, :, :, :3] * (1 - alpha) + tint * alpha
    border = {"FAIL": (1.0, 0.1, 0.1), "REVIEW": (1.0, 0.65, 0.0), "PASS": (0.0, 0.8, 0.45)}
    for index in range(output.shape[0]):
        severities = {issue["severity"] for issue in report["issues"] if issue.get("frame_index", index) == index}
        verdict = "FAIL" if "FAIL" in severities else "REVIEW" if "REVIEW" in severities else "PASS"
        color = torch.tensor(border[verdict])
        output[index, 0, :, :3] = color
        output[index, -1, :, :3] = color
        output[index, :, 0, :3] = color
        output[index, :, -1, :3] = color
    return output
