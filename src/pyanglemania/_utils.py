"""Backend dispatch and small shared helpers.

The rest of the package never imports numpy/cupy directly; it asks
:func:`get_array_module` for the right array namespace given whatever array
the caller's ``AnnData`` already holds. This mirrors how rapids-singlecell
operates on GPU-resident AnnData objects: nothing here moves data to the GPU
itself, it just dispatches to numpy or cupy depending on what is already
there (e.g. after ``rapids_singlecell.get.anndata_to_GPU(adata)``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse as sp


def _cupy_modules():
    try:
        import cupy as cp
        from cupyx.scipy import sparse as csp
    except ImportError:
        return None, None
    return cp, csp


def is_sparse(x: Any) -> bool:
    if sp.issparse(x):
        return True
    _, csp = _cupy_modules()
    return csp is not None and csp.issparse(x)


def get_array_module(x: Any):
    """Return the (dense, sparse) array namespace pair backing ``x``.

    Falls back to ``(numpy, scipy.sparse)`` for anything that isn't a cupy
    array/sparse matrix, including plain numpy arrays and scipy sparse
    matrices.
    """
    cp, csp = _cupy_modules()
    if cp is not None:
        if isinstance(x, cp.ndarray) or (csp is not None and csp.issparse(x)):
            return cp, csp
    return np, sp


def to_dense(x: Any, xp):
    """Densify ``x`` (sparse or dense) into an ``xp`` array."""
    if sp.issparse(x):
        return xp.asarray(x.toarray())
    _, csp = _cupy_modules()
    if csp is not None and csp.issparse(x):
        return x.toarray()
    return xp.asarray(x)


def to_numpy(x: Any) -> np.ndarray:
    """Pull a (small, host-side) array back to numpy regardless of backend."""
    cp, _ = _cupy_modules()
    if cp is not None and isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    return np.asarray(x)


def vmessage(verbose: bool, *parts: object) -> None:
    if verbose:
        print(*parts)
