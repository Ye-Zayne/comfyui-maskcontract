"""ComfyUI V1 input inspection and explicit strict gating."""
import json

from .core import CONVENTIONS, HARD_MAX_FRAMES, HARD_MAX_PIXELS, Policy, evaluate, evidence_overlay


def inputs():
    return {"required": {
        "images": ("IMAGE",), "mask": ("MASK",),
        "white_convention": (list(CONVENTIONS),),
        "allow_batch1_broadcast": ("BOOLEAN", {"default": False}),
        "active_threshold": ("FLOAT", {"default": 0.5, "min": 0.000001, "max": 1.0, "step": 0.01}),
        "min_coverage": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
        "max_coverage": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
        "extreme_policy": (["review", "fail", "allow"],),
        "max_frames": ("INT", {"default": 64, "min": 1, "max": HARD_MAX_FRAMES}),
        "max_total_pixels": ("INT", {"default": 8_388_608, "min": 1, "max": HARD_MAX_PIXELS}),
    }}


def serialized(report):
    return json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)


class MaskContractInspect:
    @classmethod
    def INPUT_TYPES(cls):
        schema = inputs()
        schema["required"]["overlay_opacity"] = ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.05})
        return schema

    RETURN_TYPES = ("IMAGE", "STRING", "INT")
    RETURN_NAMES = ("overlay", "report_json", "verdict")
    FUNCTION = "inspect"
    CATEGORY = "MaskContract"
    OUTPUT_NODE = True

    def inspect(self, images, mask, white_convention="white_selects_edit", allow_batch1_broadcast=False,
                active_threshold=0.5, min_coverage=0.0, max_coverage=1.0, extreme_policy="review",
                max_frames=64, max_total_pixels=8_388_608, overlay_opacity=0.4):
        policy = Policy(white_convention, allow_batch1_broadcast, active_threshold, min_coverage,
                        max_coverage, extreme_policy, max_frames, max_total_pixels)
        report, effective = evaluate(images, mask, policy)
        overlay = evidence_overlay(images, effective, report, overlay_opacity)
        text = serialized(report)
        return {"ui": {"text": [f"MaskContract: {report['verdict']} ({report['verdict_code']})", text]},
                "result": (overlay, text, report["verdict_code"])}


class MaskContractGate:
    @classmethod
    def INPUT_TYPES(cls):
        schema = inputs()
        schema["required"]["allow_review"] = ("BOOLEAN", {"default": False})
        return schema

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "INT")
    RETURN_NAMES = ("images", "mask", "report_json", "verdict")
    FUNCTION = "gate"
    CATEGORY = "MaskContract"
    OUTPUT_NODE = True

    def gate(self, images, mask, white_convention="white_selects_edit", allow_batch1_broadcast=False,
             active_threshold=0.5, min_coverage=0.0, max_coverage=1.0, extreme_policy="review",
             max_frames=64, max_total_pixels=8_388_608, allow_review=False):
        if type(allow_review) is not bool:
            raise ValueError("allow_review must be BOOLEAN")
        policy = Policy(white_convention, allow_batch1_broadcast, active_threshold, min_coverage,
                        max_coverage, extreme_policy, max_frames, max_total_pixels)
        report, effective = evaluate(images, mask, policy)
        text = serialized(report)
        if report["verdict_code"] == 0 or (report["verdict_code"] == 1 and not allow_review):
            raise ValueError(f"MaskContract gate blocked {report['verdict']}. Inspect the same inputs and policy for evidence.\n{text}")
        return (images, effective, text, report["verdict_code"])
