""" backend selection """

import os

BACKEND = os.environ.get("NEEDLE_BACKEND", "np")
# 可选后端有np和np
# np使用numpy ndarray 和 numpy api 作为后端
# nd使用自定义NDArray 和 自定义api 作为后端

if BACKEND == 'nd':
    print("Using needle backend")
    from . import backend_ndarray as array_api
    from .backend_ndarray import (
        all_devices,
        cuda,
        cpu,
        cpu_numpy,
        default_device,
        BackendDevice as Device
    )
    
    NDArray = array_api.NDArray
elif BACKEND == 'np':
    print("using numpy backend")
    import numpy as array_api
    from .backend_numpy import all_devices, cpu, default_device, Device
    
    NDArray = array_api.ndarray
else:
    raise RuntimeError(f"Unknown needle array backend: {BACKEND}")
    