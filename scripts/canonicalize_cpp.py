"""C++ canonicalization helper via ctypes.

Expects shared library at build/libcanonicalize.so relative to repo root.
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Iterable, Sequence, Tuple

Cell = Tuple[int, int, int]
Cluster = Tuple[Cell, ...]


def _load_lib() -> ctypes.CDLL:
    lib_path = Path(__file__).resolve().parent.parent / "build" / "libcanonicalize.so"
    lib = ctypes.CDLL(str(lib_path))
    lib.canonicalize_shape.argtypes = [
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int32),
    ]
    lib.canonicalize_shape.restype = None
    return lib


_LIB = _load_lib()


def canonicalize_shape_cpp(cells: Sequence[Cell]) -> Cluster:
    n = len(cells)
    arr = (ctypes.c_int32 * (3 * n))()
    for i, (x, y, z) in enumerate(cells):
        arr[3 * i] = x
        arr[3 * i + 1] = y
        arr[3 * i + 2] = z
    out = (ctypes.c_int32 * (3 * n))()
    _LIB.canonicalize_shape(arr, n, out)
    return tuple((int(out[3 * i]), int(out[3 * i + 1]), int(out[3 * i + 2])) for i in range(n))

