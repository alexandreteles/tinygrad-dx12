"""DXGI adapter declarations (Windows SDK C ABI)."""

import ctypes as C
from tinygrad_dx12.runtime.autogen.win32 import UINT, SIZE_T, LUID, GUID

IID_IDXGIFactory1 = GUID("770aae78-f26f-4dba-a829-253c83d1b387")
IID_IDXGIAdapter1 = GUID("29038f61-3839-4626-91fd-086879011a05")
DXGI_ERROR_NOT_FOUND = 0x887A0002
DXGI_ADAPTER_FLAG_SOFTWARE = 2


class DXGI_ADAPTER_DESC1(C.Structure):
    _fields_ = [
        ("Description", C.c_wchar * 128),
        ("VendorId", UINT),
        ("DeviceId", UINT),
        ("SubSysId", UINT),
        ("Revision", UINT),
        ("DedicatedVideoMemory", SIZE_T),
        ("DedicatedSystemMemory", SIZE_T),
        ("SharedSystemMemory", SIZE_T),
        ("AdapterLuid", LUID),
        ("Flags", UINT),
    ]
