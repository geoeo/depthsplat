"""Op substitutions that keep the depth predictor exportable.

Same idea as `XFORMERS_DISABLED` (see
`src/model/encoder/unimatch/ldm_unet/cross_attention.py`) and
`TYPECHECK_DISABLED` (see `src/typecheck.py`): an operation that is correct in
eager but cannot survive `torch.export()` + AOTInductor is swapped for an
equivalent that can. The switch is read from the environment at import time, so
`scripts/depth_export_common.py` sets it before anything under `src` is
imported, and the normal Python pipeline is untouched.

`AOTI_SAFE_INVERSE` exists because `aten.linalg_inv_ex` -- which
`torch.inverse()`, `Tensor.inverse()` and `torch.linalg.inv()` all decompose to
-- has no c-shim implementation in torch 2.6. AOTInductor routes it through the
proxy executor instead, and that path mis-deserializes the op's `check_errors`
argument:

    Exception in aoti_torch: INTERNAL ASSERT FAILED at ivalue.h:676,
    please report a bug to PyTorch. expected bool

Nothing was removed in 2.6; the op never had a c-shim. What changed is that
ABI-compatible codegen became unconditional. Under 2.4 the generated code was
free to call ATen's C++ API directly, and did -- the 2.4 build of this model
carries the symbol `at::_ops::linalg_inv_ex::call(at::Tensor const&, bool)`
alongside its `aoti_torch_*` shim calls. 2.6 emits shim calls only (verified:
zero `at::_ops` symbols in the new .so, 24 `aoti_torch_*`), and the
`abi_compatible` / `c_shim_version` config knobs that used to switch modes are
gone. So an op absent from the shim can no longer be called at all except
through the proxy executor.

The shim's op list lives in `torchgen/aoti/fallback_ops.py` (136 ops for CUDA in
2.6). `linalg_inv_ex` is not in it, and `linalg` appears nowhere in the
`aoti_torch/` headers -- though `cholesky_inverse`, `cholesky_solve`, `geqrf`,
`lu_unpack`, `ormqr` and `triangular_solve` are all present, so the omission
looks like an oversight rather than a policy. Of the seven ops the 2.4 build
called directly, this is the only one 2.6 left uncovered.

The call yields an undefined tensor and inference then dies a few ops later on
whatever consumed it (`Cannot access storage of UndefinedTensorImpl`, or a shape
assert inside an unrelated bmm). Two things make this expensive to diagnose:

  * it fails at *runtime*, not at compile time. The compile succeeds and only
    warns: "aten.linalg_inv_ex.default is missing a c-shim implementation,
    using proxy executor as fallback".
  * it is not caused by the packaging API. Reproduced on 2.6.0+cu126 with a
    two-line model through both `aoti_compile_and_package()` and the deprecated
    `torch._export.aot_compile()`; the latter fares worse still, being unable to
    locate a proxy executor at all.

Recheck this on the next torch upgrade. If the c-shim lands, the substitution
can be dropped and the two call sites can go back to `torch.inverse`.

The replacements are closed-form cofactor inverses, not approximations. They are
exact for any invertible input, so agreement with `torch.inverse` is a matter of
fp32 rounding only -- `aot_compile_depth.py` reports the measured gap against
eager on every build.
"""

import os

import torch

# Presence of the variable enables the substitution; unset means use torch.inverse
# (the default, for the eager pipeline).
SAFE_INVERSE = os.environ.get("AOTI_SAFE_INVERSE") is not None


def inverse(matrix: torch.Tensor) -> torch.Tensor:
    """`torch.inverse`, or an export-safe equivalent when AOTI_SAFE_INVERSE is set.

    Accepts a batch of 3x3 matrices, or of 4x4 matrices whose last row is
    [0, 0, 0, 1] (any affine transform, which every camera-to-world extrinsic in
    this pipeline is). Anything else has no substitution and would be silently
    wrong, so it raises rather than guessing.
    """
    if not SAFE_INVERSE:
        return torch.inverse(matrix)

    n = matrix.shape[-1]
    if matrix.shape[-2] != n:
        raise ValueError(f"expected square matrices, got {tuple(matrix.shape)}")
    if n == 3:
        return _inverse_3x3(matrix)
    if n == 4:
        return _inverse_affine_4x4(matrix)
    raise NotImplementedError(
        f"AOTI_SAFE_INVERSE has no {n}x{n} substitution; either add one or export "
        f"with the inverse hoisted out of the traced region"
    )


def _inverse_3x3(m: torch.Tensor) -> torch.Tensor:
    """Cofactor inverse of a batch of 3x3 matrices, [..., 3, 3] -> [..., 3, 3].

    adj(M)^T / det(M), written out. Only mul/sub/div/stack, so it lowers to
    plain ATen ops with no c-shim gap and no proxy executor.
    """
    a, b, c = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    d, e, f = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    g, h, i = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]

    # Cofactors, already transposed into the adjugate's layout.
    c00, c01, c02 = e * i - f * h, c * h - b * i, b * f - c * e
    c10, c11, c12 = f * g - d * i, a * i - c * g, c * d - a * f
    c20, c21, c22 = d * h - e * g, b * g - a * h, a * e - b * d

    det = a * c00 + b * c10 + c * c20
    adjugate = torch.stack([
        torch.stack([c00, c01, c02], dim=-1),
        torch.stack([c10, c11, c12], dim=-1),
        torch.stack([c20, c21, c22], dim=-1),
    ], dim=-2)
    return adjugate / det[..., None, None]


def _inverse_affine_4x4(m: torch.Tensor) -> torch.Tensor:
    """Inverse of a batch of 4x4 affine matrices, [..., 4, 4] -> [..., 4, 4].

    Requires the last row to be [0, 0, 0, 1], which lets the inverse be written
    as [[A^-1, -A^-1 t], [0, 1]]. Note that A is inverted properly rather than
    transposed: a transpose would assume an orthonormal rotation, and this way
    the result stays correct if a matrix ever carries scale.
    """
    a = m[..., :3, :3]
    t = m[..., :3, 3:]
    a_inv = _inverse_3x3(a)
    top = torch.cat([a_inv, -a_inv @ t], dim=-1)   # [..., 3, 4]
    bottom = m[..., 3:, :]                         # [..., 1, 4], i.e. [0, 0, 0, 1]
    return torch.cat([top, bottom], dim=-2)
