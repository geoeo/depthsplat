"""Opt-out gate for the jaxtyping/beartype import hook.

The hook rewrites every `forward` under `src` with a runtime type checker as the
modules are imported. That happens at import time and cannot be undone afterwards,
so it is controlled by the process environment rather than by a function argument:
`TYPECHECK_DISABLED` must be set before the first `src` import, exactly like
`XFORMERS_DISABLED`.

torch.export() traces through the beartype wrappers, so an export script must
import with the hook disabled.
"""

import contextlib
import os
from typing import ContextManager

from jaxtyping import install_import_hook

# Presence of the variable disables the hook; unset means enabled (the default).
TYPECHECK_ENABLED = os.environ.get("TYPECHECK_DISABLED") is None


def typecheck_hook() -> ContextManager:
    """Wrap `src` imports in the beartype/jaxtyping hook, or in a no-op if disabled."""
    if not TYPECHECK_ENABLED:
        return contextlib.nullcontext()
    return install_import_hook(("src",), ("beartype", "beartype"))
