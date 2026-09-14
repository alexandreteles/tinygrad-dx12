"""DXC C ABI declarations from Microsoft.Direct3D.DXC 1.10.2605.37-preview."""

import ctypes as C
from tinygrad_dx12.runtime.autogen.win32 import GUID, UINT, SIZE_T

CLSID_DxcUtils = GUID("6245d6af-66e0-48fd-80b4-4d271796748c")
CLSID_DxcCompiler = GUID("73e22d93-e6ce-47f3-b5bf-f0664f39c1b0")
IID_IDxcUtils = GUID("4605c4cb-2019-492a-ada4-65f20bb7d67f")
IID_IDxcCompiler3 = GUID("228b4687-5a6a-4730-900c-9702b2203f54")
IID_IDxcResult = GUID("58346cda-dde7-4497-9461-6f87af5e0659")
DXC_CP_UTF8 = 65001


class DxcBuffer(C.Structure):
    _fields_ = [("Ptr", C.c_void_p), ("Size", SIZE_T), ("Encoding", UINT)]
