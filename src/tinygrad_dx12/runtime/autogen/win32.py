"""Win32 ABI declarations used by the DX12 backend."""

import ctypes as C
import uuid

HRESULT = LONG = BOOL = C.c_int32
UINT = DWORD = C.c_uint32
UINT64 = C.c_uint64
SIZE_T = C.c_size_t
HANDLE = C.c_void_p
WINFUNCTYPE = getattr(C, "WINFUNCTYPE", C.CFUNCTYPE)


class GUID(C.Structure):
    _fields_ = [("Data1", C.c_uint32), ("Data2", C.c_uint16), ("Data3", C.c_uint16), ("Data4", C.c_uint8 * 8)]

    def __init__(self, value: str = "00000000-0000-0000-0000-000000000000") -> None:
        """Construct a GUID from its standard text representation in Windows byte order."""
        super().__init__()
        C.memmove(C.byref(self), uuid.UUID(value).bytes_le, 16)


class LUID(C.Structure):
    _fields_ = [("LowPart", DWORD), ("HighPart", LONG)]


INFINITE = 0xFFFFFFFF
WAIT_OBJECT_0 = 0
WAIT_FAILED = 0xFFFFFFFF
"""ABI declarations, kept deliberately limited to interfaces used by this backend."""
