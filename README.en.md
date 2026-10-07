# MaskContract

Validate IMAGE + MASK before inpainting or masked compositing: detect canvas/batch mismatches, nonfinite or out-of-range values, empty/full selections and per-frame coverage outside your bounds. Emit overlay evidence, JSON, and `0=FAIL / 1=REVIEW / 2=PASS`, or stop execution through a strict gate.

Install under `ComfyUI/custom_nodes/comfyui-maskcontract` and restart, or use Manager after Registry publication. Uses ComfyUI's torch, NumPy and Pillow environment; no models, keys or service. Author A ad钙; version 0.1.0; MIT. [中文说明](README.md).

| Node | Outputs |
| --- | --- |
| `MaskContractInspect` | overlay IMAGE, report STRING, verdict INT |
| `MaskContractGate` | validated IMAGE, MASK, report STRING, verdict INT |

Both are OUTPUT_NODEs. Inspect can run independently; wire its overlay to PreviewImage for visible evidence. Gate outputs must feed the editing node you want to guard. Gate blocks FAIL and, by default, REVIEW; `allow_review=true` permits REVIEW only. Frontend text display depends on ComfyUI version; the STRING report remains available.

Wire Gate into the protected downstream dependency chain. A disconnected check branch cannot guarantee it runs before independent sampling branches; Inspect never blocks execution.

## Contract

- Floating IMAGE `[B,H,W,3 or 4]`, floating MASK `[B,H,W]`, positive dimensions, finite values within `[0,1]`. Soft masks are accepted. No implicit reshape, resize, crop, inversion, clipping or normalization.
- H/W must match the **IMAGE pixel canvas**, not latent cells, crop dimensions or display dimensions. For composite, check the source image against its mask. Destination placement/overlap and latent compatibility are outside this contract.
- Batch counts must match. `allow_batch1_broadcast=true` explicitly permits a single mask to cover the image batch; Gate returns its expanded `[B,H,W]` view, with original values/dtype/device. Multiple masks are never cyclically repeated.
- `white_convention` declares white as edit/protect/source-selection. It does not change values and never infers polarity or semantic intent. Coverage bounds can catch an unexpected inversion, but PASS does not prove the intended region was selected.
- Coverage counts `mask >= active_threshold` (default 0.5); soft-mask mean is reported separately. Coverage bounds are inclusive and checked per frame. Bboxes `[x1,y1,x2,y2]` use exclusive right/bottom edges and are null for no selected pixels.
- Exact `all_empty/all_full` mean all values are 0/1. Separate threshold-none/all fields describe selection. Threshold-none/all defaults to REVIEW; `extreme_policy=fail` blocks, and `allow` explicitly accepts them.

Structural/numeric/coverage violations FAIL. The worst per-frame verdict determines the batch verdict. Inspect tints white-valued regions cyan (purple for protect), preserving RGBA alpha; green/amber/red frame borders indicate verdicts. An invalid MASK with valid IMAGE yields untinted image-canvas evidence. An invalid/oversized IMAGE yields a documented red 32×32 placeholder. Evidence is CPU float32 and is never substituted for Gate inputs.

Work budgets are checked before reductions, transfer or evidence allocation: default 64 frames and 8,388,608 total batch pixels; hard ceilings 256 frames and 16,777,216 pixels. Split large batches or adjust the explicit budget. Validation describes these tensors now, not later node behavior or concurrent external mutation.

## Existing tools and scope

Official ComfyUI [MaskComposite](https://docs.comfy.org/built-in-nodes/MaskComposite), InvertMask and MaskToImage manipulate or display masks. [ImageCompositeMasked](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_mask.py) interpolates masks and repeats batches for compositing. MaskContract checks an explicit input contract before those operations; it does not replace them or claim that no similar checker exists. [EditFence](https://github.com/Ye-Zayne/comfyui-editfence) checks changes outside permitted regions after an edit. MaskContract checks the input only.

## Examples

Copy `examples/maskcontract_image.png`, `maskcontract_alpha_mask.png` and `maskcontract_background.png` into ComfyUI input. `inspect_api.json` previews evidence. `gate_composite_api.json` sends Gate outputs into native ImageCompositeMasked and previews the result. The mask PNG stores `1-mask` in alpha for LoadImage's MASK output; an opaque white RGB PNG is not a white MASK.

These are **API-format prompts**, sent to `POST /prompt` as `{"prompt": <JSON content>}`, not canvas workflow exports. Expected coverage is 0.1875 with bbox `[48,32,120,96]`. Swap the mask file for the supplied 64×64 `maskcontract_wrong_canvas.png` to inspect a FAIL or verify gate blocking. The generator is development-only and excluded from Registry archives.

```sh
python -m pip install pytest ruff torch numpy Pillow
python -m pytest --rootdir=tests --confcutdir=tests tests --import-mode=importlib -q
python -m ruff check . --select S102,S307,E702
```

Runtime has no dynamic imports, environment-variable reads, network/database/CLI operations, or persistent filesystem outputs.
