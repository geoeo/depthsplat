"""Single definition of the float32 arithmetic mode.

Previously each entry point set this for itself -- `depth_inference.py` and
`main.py` both hardcoded `set_float32_matmul_precision("high")`, and the export
scripts set something stricter -- which meant the Python reference and the
compiled artifact were quietly running different arithmetic.

The distinction matters more than usual for this model. The cost-volume
regression amplifies TF32 rounding, so on a 3-view 480x640 scene:

  * TF32 vs full fp32, both in eager: 1.30 m max, 0.105 m mean
  * an AOTInductor build vs eager, both TF32: 6.48 m max, 0.48 m mean
  * an AOTInductor build vs eager, both full fp32: 8.7e-04 m max

Only the last is agreement. TF32 is not wrong -- it is a different, faster
arithmetic -- but a reference generated under it cannot be used to validate a
compiled build numerically.

Note that `set_float32_matmul_precision` alone does not cover cudnn, whose
`allow_tf32` defaults to True. Both flags are set explicitly here so neither
mode depends on a torch default.
"""

import torch


def set_fp32_precision(full_fp32: bool) -> None:
    """True: full fp32 everywhere. False: allow TF32 (faster, ~1.3 m different)."""
    if full_fp32:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def describe_precision() -> str:
    return (
        f"matmul={torch.get_float32_matmul_precision()} "
        f"cuda.matmul.allow_tf32={torch.backends.cuda.matmul.allow_tf32} "
        f"cudnn.allow_tf32={torch.backends.cudnn.allow_tf32}"
    )
