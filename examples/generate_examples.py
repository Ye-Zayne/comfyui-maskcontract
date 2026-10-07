"""Development-only generation of committed synthetic PNGs and API prompts."""
import json
from pathlib import Path

from PIL import Image, ImageDraw


def main():
    directory = Path(__file__).resolve().parent
    image = Image.new("RGB", (192, 128), (45, 65, 85))
    draw = ImageDraw.Draw(image)
    draw.rectangle((48, 32, 119, 95), fill=(235, 175, 65))
    image.save(directory / "maskcontract_image.png")
    Image.new("RGB", image.size, (65, 25, 85)).save(directory / "maskcontract_background.png")
    mask = Image.new("RGBA", image.size, (255, 255, 255, 255))
    ImageDraw.Draw(mask).rectangle((48, 32, 119, 95), fill=(255, 255, 255, 0))
    mask.save(directory / "maskcontract_alpha_mask.png")
    wrong = Image.new("RGBA", (64, 64), (255, 255, 255, 255))
    ImageDraw.Draw(wrong).rectangle((16, 16, 47, 47), fill=(255, 255, 255, 0))
    wrong.save(directory / "maskcontract_wrong_canvas.png")
    settings = {"images": ["1", 0], "mask": ["2", 1], "white_convention": "white_selects_edit",
                "allow_batch1_broadcast": False, "active_threshold": 0.5, "min_coverage": 0.05,
                "max_coverage": 0.4, "extreme_policy": "review", "max_frames": 64, "max_total_pixels": 8_388_608}
    loads = {"1": {"class_type": "LoadImage", "inputs": {"image": "maskcontract_image.png"}},
             "2": {"class_type": "LoadImage", "inputs": {"image": "maskcontract_alpha_mask.png"}}}
    inspect = dict(loads, **{
        "3": {"class_type": "MaskContractInspect", "inputs": dict(settings, overlay_opacity=0.4)},
        "4": {"class_type": "PreviewImage", "inputs": {"images": ["3", 0]}},
    })
    gate = dict(loads, **{
        "3": {"class_type": "MaskContractGate", "inputs": dict(settings, allow_review=False)},
        "4": {"class_type": "LoadImage", "inputs": {"image": "maskcontract_background.png"}},
        "5": {"class_type": "ImageCompositeMasked", "inputs": {"destination": ["4", 0], "source": ["3", 0], "mask": ["3", 1], "x": 0, "y": 0, "resize_source": False}},
        "6": {"class_type": "PreviewImage", "inputs": {"images": ["5", 0]}},
    })
    for name, prompt in (("inspect_api.json", inspect), ("gate_composite_api.json", gate)):
        (directory / name).write_text(json.dumps(prompt, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
