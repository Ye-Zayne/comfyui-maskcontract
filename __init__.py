"""MaskContract: explicit IMAGE/MASK contracts before editing."""
from .nodes import MaskContractGate, MaskContractInspect

NODE_CLASS_MAPPINGS = {"MaskContractInspect": MaskContractInspect, "MaskContractGate": MaskContractGate}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskContractInspect": "MaskContract · Inspect Image + Mask",
    "MaskContractGate": "MaskContract · Strict Image + Mask Gate",
}
__version__ = "0.1.0"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
