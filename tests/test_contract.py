import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from maskcontract_test_package.core import Policy, evaluate, evidence_overlay
from maskcontract_test_package.nodes import MaskContractGate, MaskContractInspect


def sample(batch=1, channels=3):
    image = torch.full((batch, 8, 10, channels), 0.25)
    mask = torch.zeros((batch, 8, 10))
    mask[:, 2:6, 3:8] = 1
    return image, mask


def codes(report):
    return {issue["code"] for issue in report["issues"]}


def test_region_geometry_and_inclusive_coverage_bounds():
    image, mask = sample()
    report, effective = evaluate(image, mask, Policy(min_coverage=0.25, max_coverage=0.25))
    assert report["verdict_code"] == 2
    assert effective is mask
    frame = report["frames"][0]
    assert (frame["coverage"], frame["selected_pixels"], frame["bbox_xyxy"]) == (0.25, 20, [3, 2, 8, 6])
    assert not frame["all_empty"] and not frame["all_full"]


@pytest.mark.parametrize("images,mask,expected", [
    (torch.zeros(8, 10, 3), torch.zeros(1, 8, 10), "image_shape"),
    (torch.zeros(1, 8, 10, 1), torch.zeros(1, 8, 10), "image_shape"),
    (torch.zeros(1, 8, 10, 3), torch.zeros(8, 10), "mask_shape"),
    (torch.zeros(1, 8, 10, 3), torch.zeros(1, 10, 8), "canvas_mismatch"),
    (torch.zeros(0, 8, 10, 3), torch.zeros(0, 8, 10), "image_shape"),
    (None, torch.zeros(1, 8, 10), "image_type"),
])
def test_invalid_contracts_fail_with_json_safe_evidence(images, mask, expected):
    report, effective = evaluate(images, mask, Policy())
    assert report["verdict_code"] == 0 and expected in codes(report)
    assert effective is None
    json.dumps(report, allow_nan=False)


def test_single_mask_broadcast_requires_explicit_choice():
    image, _ = sample(batch=3)
    _, mask = sample()
    report, effective = evaluate(image, mask, Policy())
    assert "batch_mismatch" in codes(report) and effective is None
    report, effective = evaluate(image, mask, Policy(allow_batch1_broadcast=True))
    assert report["verdict_code"] == 2 and report["broadcast_applied"]
    assert tuple(effective.shape) == (3, 8, 10)
    assert [frame["source_mask_index"] for frame in report["frames"]] == [0, 0, 0]
    assert effective.data_ptr() == mask.data_ptr()


def test_broadcast_permission_does_not_repeat_multiple_masks():
    image, _ = sample(batch=3)
    _, mask = sample(batch=2)
    report, _ = evaluate(image, mask, Policy(allow_batch1_broadcast=True))
    assert "batch_mismatch" in codes(report)


@pytest.mark.parametrize("target,bad,code", [
    ("mask", float("nan"), "mask_nonfinite"),
    ("mask", float("inf"), "mask_nonfinite"),
    ("image", float("nan"), "image_nonfinite"),
    ("image", float("-inf"), "image_nonfinite"),
    ("mask", -0.001, "mask_range"),
    ("image", 1.001, "image_range"),
])
def test_invalid_numeric_values_are_not_normalized(target, bad, code):
    image, mask = sample()
    value = image if target == "image" else mask
    value.reshape(-1)[0] = bad
    report, effective = evaluate(image, mask, Policy())
    assert report["verdict_code"] == 0 and code in codes(report) and effective is None
    json.dumps(report, allow_nan=False)
    if math_is_nan(bad):
        assert torch.isnan(value.reshape(-1)[0])
    else:
        assert float(value.reshape(-1)[0]) == pytest.approx(bad)


def math_is_nan(value):
    return value != value


def test_integer_mask_is_rejected():
    image, mask = sample()
    report, _ = evaluate(image, mask.to(torch.uint8), Policy())
    assert "mask_dtype" in codes(report)


@pytest.mark.parametrize("fill,expected", [(0, "empty_selection"), (1, "full_selection")])
def test_extreme_masks_review_by_default_and_can_be_explicitly_allowed(fill, expected):
    image, mask = sample()
    mask.fill_(fill)
    report, _ = evaluate(image, mask, Policy())
    assert report["verdict_code"] == 1 and expected in codes(report)
    assert report["frames"][0]["all_empty"] == (fill == 0)
    assert report["frames"][0]["all_full"] == (fill == 1)
    assert evaluate(image, mask, Policy(extreme_policy="allow"))[0]["verdict_code"] == 2
    assert evaluate(image, mask, Policy(extreme_policy="fail"))[0]["verdict_code"] == 0


def test_soft_mask_threshold_and_weighted_mean_are_distinct():
    image, mask = sample()
    mask = mask * 0.4 + 0.2
    report, effective = evaluate(image, mask, Policy(active_threshold=0.5))
    frame = report["frames"][0]
    assert report["verdict_code"] == 2 and effective is mask
    assert frame["coverage"] == 0.25 and frame["mean_mask_value"] == pytest.approx(0.3)
    assert not frame["all_empty"] and not frame["all_full"]


def test_subthreshold_nonzero_mask_reports_empty_selection_without_claiming_all_zero():
    image, mask = sample()
    mask.fill_(0.25)
    report, _ = evaluate(image, mask, Policy())
    assert report["verdict_code"] == 1
    frame = report["frames"][0]
    assert frame["bbox_xyxy"] is None and frame["threshold_selects_none"] and not frame["all_empty"]


def test_batch_records_each_frame_and_worst_verdict():
    image, mask = sample(batch=2)
    mask[1].fill_(1)
    report, _ = evaluate(image, mask, Policy(max_coverage=0.5))
    assert report["verdict_code"] == 0
    assert [frame["coverage"] for frame in report["frames"]] == [0.25, 1]
    assert next(issue for issue in report["issues"] if issue["code"] == "coverage_outside_bounds")["frame_index"] == 1


def test_inversion_is_visible_in_coverage_but_polarity_is_never_inferred():
    image, mask = sample()
    a, _ = evaluate(image, mask, Policy(white_convention="white_protects"))
    b, _ = evaluate(image, 1 - mask, Policy(white_convention="white_protects"))
    assert (a["frames"][0]["coverage"], b["frames"][0]["coverage"]) == (0.25, 0.75)
    assert a["verdict_code"] == b["verdict_code"] == 2
    assert a["policy"]["white_convention"] == b["policy"]["white_convention"] == "white_protects"


def test_work_budget_fails_before_value_reductions_or_overlay_copy(monkeypatch):
    image, mask = sample()
    monkeypatch.setattr(torch, "isfinite", lambda _: pytest.fail("Oversized input must not be scanned"))
    report, effective = evaluate(image, mask, Policy(max_total_pixels=4))
    assert "resource_limit" in codes(report)
    overlay = evidence_overlay(image, effective, report)
    assert tuple(overlay.shape) == (1, 32, 32, 3) and report["overlay"]["kind"] == "placeholder"


def test_frame_budget_is_checked():
    image, mask = sample(batch=3)
    report, _ = evaluate(image, mask, Policy(max_frames=2))
    assert "resource_limit" in codes(report)


@pytest.mark.parametrize("kwargs", [{"active_threshold": 0}, {"min_coverage": 0.8, "max_coverage": 0.2},
    {"max_total_pixels": 16_777_217}, {"max_frames": 257}, {"max_frames": True}, {"white_convention": "auto"}])
def test_invalid_policy_rejected(kwargs):
    with pytest.raises(ValueError):
        evaluate(*sample(), Policy(**kwargs))


def test_evidence_preserves_input_and_rgba_alpha():
    image, mask = sample(channels=4)
    before = image.clone()
    report, effective = evaluate(image, mask, Policy())
    overlay = evidence_overlay(image, effective, report)
    assert torch.equal(image, before)
    assert torch.equal(overlay[:, :, :, 3], before[:, :, :, 3])
    assert not torch.equal(overlay[:, 3, 4, :3], before[:, 3, 4, :3])
    assert overlay.dtype == torch.float32 and overlay.device.type == "cpu"


def test_gate_returns_original_tensors_and_stops_fail_or_unapproved_review():
    image, mask = sample()
    outputs = MaskContractGate().gate(image, mask)
    assert outputs[0] is image and outputs[1] is mask and outputs[-1] == 2
    with pytest.raises(ValueError, match="blocked FAIL"):
        MaskContractGate().gate(image, mask[:, :, :-1])
    with pytest.raises(ValueError, match="blocked REVIEW"):
        MaskContractGate().gate(image, torch.zeros_like(mask))
    assert MaskContractGate().gate(image, torch.zeros_like(mask), allow_review=True)[-1] == 1


def test_gate_broadcast_output_matches_report_without_casting_or_normalizing():
    image, _ = sample(batch=2)
    _, mask = sample()
    image, mask = image.to(torch.float64), mask.to(torch.float16)
    out_image, out_mask, text, verdict = MaskContractGate().gate(image, mask, allow_batch1_broadcast=True)
    assert out_image is image and out_mask.dtype == mask.dtype
    assert tuple(out_mask.shape) == (2, 8, 10) and out_mask.data_ptr() == mask.data_ptr()
    assert json.loads(text)["broadcast_applied"] and verdict == 2


def test_inspector_is_output_node_and_reports_mismatch_with_canvas_evidence():
    image, mask = sample()
    result = MaskContractInspect().inspect(image, mask[:, :, :-1])
    overlay, text, verdict = result["result"]
    assert MaskContractInspect.OUTPUT_NODE and MaskContractGate.OUTPUT_NODE
    assert verdict == 0 and tuple(overlay.shape) == tuple(image.shape)
    assert json.loads(text)["overlay"]["mask_tint_applied"] is False
    assert result["ui"]["text"]


@pytest.mark.parametrize("device", ["cuda", "mps"])
def test_available_gpu_statistics_and_cpu_evidence(device):
    available = torch.cuda.is_available() if device == "cuda" else torch.backends.mps.is_available()
    if not available:
        pytest.skip(f"{device} is unavailable in this test environment")
    image, mask = sample()
    image, mask = image.to(device), mask.to(device)
    report, effective = evaluate(image, mask, Policy())
    assert report["verdict_code"] == 2
    assert report["frames"][0]["mean_mask_value"] == pytest.approx(0.25)
    assert evidence_overlay(image, effective, report).device.type == "cpu"
    out_image, out_mask, _text, _verdict = MaskContractGate().gate(image, mask)
    assert out_image.device.type == out_mask.device.type == device


def test_supplied_api_examples_have_valid_node_inputs_and_explicit_dependency_chain():
    directory = Path(__file__).resolve().parents[1] / "examples"
    image = torch.from_numpy(np.asarray(Image.open(directory / "maskcontract_image.png").convert("RGB"), dtype=np.float32) / 255).unsqueeze(0)
    alpha = np.asarray(Image.open(directory / "maskcontract_alpha_mask.png"), dtype=np.float32)[:, :, 3] / 255
    mask = torch.from_numpy(1 - alpha).unsqueeze(0)
    report, _ = evaluate(image, mask, Policy(min_coverage=0.05, max_coverage=0.4))
    assert report["verdict_code"] == 2
    assert report["frames"][0]["coverage"] == 0.1875
    assert report["frames"][0]["bbox_xyxy"] == [48, 32, 120, 96]
    for filename, node_type in (("inspect_api.json", MaskContractInspect), ("gate_composite_api.json", MaskContractGate)):
        prompt = json.loads((directory / filename).read_text())
        assert set(prompt["3"]["inputs"]) == set(node_type.INPUT_TYPES()["required"])
        for node in prompt.values():
            if node["class_type"] == "LoadImage":
                assert (directory / node["inputs"]["image"]).is_file()
    prompt = json.loads((directory / "gate_composite_api.json").read_text())
    assert prompt["5"]["inputs"]["source"] == ["3", 0]
    assert prompt["5"]["inputs"]["mask"] == ["3", 1]
    wrong = Image.open(directory / "maskcontract_wrong_canvas.png")
    wrong_mask = torch.zeros((1, wrong.height, wrong.width))
    assert "canvas_mismatch" in codes(evaluate(image, wrong_mask, Policy())[0])
