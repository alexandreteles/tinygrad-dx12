"""Minimal C ABI declarations from Microsoft.Direct3D.D3D12 1.721.3-preview.

COM slots include inherited methods. Structure-return methods use their explicit
C ABI result pointer (resource GetDesc=10; descriptor handles=9,10).
"""

import ctypes as C
from tinygrad_dx12.runtime.autogen.win32 import UINT, UINT64, SIZE_T, BOOL, GUID

IID_ID3D12Device = GUID("189819f1-1db6-4b57-be54-1821339b85f7")
IID_ID3D12SDKConfiguration1 = GUID("8aaf9303-ad25-48b9-9a57-d9c37e009d9f")
IID_ID3D12DeviceFactory = GUID("61f307d3-d34e-4e7c-8374-3ba4de23cccb")
IID_ID3D12CommandQueue = GUID("0ec870a6-5d7e-4c22-8cfc-5baae07616ed")
IID_ID3D12CommandAllocator = GUID("6102dee4-af59-4b09-b999-b44d73f09b24")
IID_ID3D12GraphicsCommandList7 = GUID("dd171223-8b61-4769-90e3-160ccde4e2c1")
IID_ID3D12Resource = GUID("696442be-a72e-4059-bc79-5b5c98040fad")
IID_ID3D12Fence = GUID("0a753dcf-c4d8-4b91-adf6-be5a60d95a76")
IID_ID3D12RootSignature = GUID("c54a6b66-72df-4ee8-8be5-a946a1429214")
IID_ID3D12PipelineState = GUID("765a30f3-f624-4c6f-a828-ace948622445")
IID_ID3D12DescriptorHeap = GUID("8efb471d-616c-4f49-90f7-127bb763fa51")
IID_ID3D12QueryHeap = GUID("0d9658ae-ed45-469e-a61d-970ec583cab4")
IID_ID3D12CommandSignature = GUID("c36a797c-ec80-4f0a-8985-a7b2475082d1")
IID_ID3D12Heap = GUID("6b3b2502-6e51-45b3-90ee-9884265e8df3")
IID_ID3D12Debug = GUID("344488b7-6846-474b-b989-f027448245e0")
CLSID_D3D12Debug = GUID("f2352aeb-dd84-49fe-b97b-a9dcfdcc1b4f")
CLSID_D3D12SDKConfiguration = GUID("7cda6aca-a03e-49c8-9458-0334d20e07ce")
D3D12ExperimentalShaderModels = GUID("76f5573e-f13a-40f5-b297-81ce9e18933f")

D3D_FEATURE_LEVEL_11_0 = 0xB000
D3D_ROOT_SIGNATURE_VERSION_1_1 = 2
D3D12_COMMAND_LIST_TYPE_COMPUTE = 2
D3D12_COMMAND_QUEUE_FLAG_NONE = 0
D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_TYPE_UPLOAD, D3D12_HEAP_TYPE_READBACK = 1, 2, 3
D3D12_HEAP_FLAG_NONE, D3D12_HEAP_FLAG_SHARED, D3D12_HEAP_FLAG_SHARED_CROSS_ADAPTER = 0, 1, 0x20
D3D12_RESOURCE_DIMENSION_BUFFER = 1
D3D12_TEXTURE_LAYOUT_ROW_MAJOR = 1
D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER = 0, 4, 0x10
D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_GENERIC_READ = 0, 0x400, 0xAC3
D3D12_FENCE_FLAG_NONE, D3D12_FENCE_FLAG_SHARED, D3D12_FENCE_FLAG_SHARED_CROSS_ADAPTER = 0, 1, 2
D3D12_BARRIER_TYPE_GLOBAL, D3D12_BARRIER_TYPE_TEXTURE, D3D12_BARRIER_TYPE_BUFFER = 0, 1, 2
D3D12_BARRIER_SYNC_NONE, D3D12_BARRIER_SYNC_COMPUTE_SHADING, D3D12_BARRIER_SYNC_COPY, D3D12_BARRIER_SYNC_EXECUTE_INDIRECT = 0, 0x80, 0x200, 0x800
D3D12_BARRIER_ACCESS_COMMON, D3D12_BARRIER_ACCESS_UNORDERED_ACCESS, D3D12_BARRIER_ACCESS_SHADER_RESOURCE = 0, 0x10, 0x80
D3D12_BARRIER_ACCESS_INDIRECT_ARGUMENT, D3D12_BARRIER_ACCESS_COPY_DEST, D3D12_BARRIER_ACCESS_COPY_SOURCE = 0x200, 0x400, 0x800
D3D12_BARRIER_ACCESS_NO_ACCESS = 0x80000000
D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE, D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS = 0, 1
D3D12_ROOT_PARAMETER_TYPE_CBV, D3D12_ROOT_PARAMETER_TYPE_SRV, D3D12_ROOT_PARAMETER_TYPE_UAV = 2, 3, 4
D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE = 2
D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE, D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE = 1, 2
D3D12_DESCRIPTOR_RANGE_TYPE_SRV, D3D12_DESCRIPTOR_RANGE_TYPE_UAV = 0, 1
D3D12_SHADER_VISIBILITY_ALL, D3D12_ROOT_SIGNATURE_FLAG_NONE = 0, 0
D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE = 0, 1
D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING = 5768
D3D12_SRV_DIMENSION_BUFFER, D3D12_UAV_DIMENSION_BUFFER = 1, 1
D3D12_BUFFER_SRV_FLAG_RAW, D3D12_BUFFER_UAV_FLAG_RAW = 1, 1
DXGI_FORMAT_UNKNOWN, DXGI_FORMAT_R32_TYPELESS = 0, 39
D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH, D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT = 2, 5
D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT_BUFFER_VIEW = 6
D3D12_INDIRECT_ARGUMENT_TYPE_SHADER_RESOURCE_VIEW, D3D12_INDIRECT_ARGUMENT_TYPE_UNORDERED_ACCESS_VIEW = 7, 8
D3D12_QUERY_HEAP_TYPE_TIMESTAMP, D3D12_QUERY_TYPE_TIMESTAMP = 1, 2
D3D12_FEATURE_D3D12_OPTIONS, D3D12_FEATURE_SHADER_MODEL, D3D12_FEATURE_D3D12_OPTIONS1 = 0, 7, 8
D3D12_FEATURE_ROOT_SIGNATURE, D3D12_FEATURE_D3D12_OPTIONS4, D3D12_FEATURE_D3D12_OPTIONS12 = 12, 23, 41
D3D12_FEATURE_LINEAR_ALGEBRA_SUPPORT = 77


class DXGI_SAMPLE_DESC(C.Structure):
    _fields_ = [("Count", UINT), ("Quality", UINT)]


class D3D12_COMMAND_QUEUE_DESC(C.Structure):
    _fields_ = [("Type", UINT), ("Priority", C.c_int32), ("Flags", UINT), ("NodeMask", UINT)]


class D3D12_HEAP_PROPERTIES(C.Structure):
    _fields_ = [("Type", UINT), ("CPUPageProperty", UINT), ("MemoryPoolPreference", UINT), ("CreationNodeMask", UINT), ("VisibleNodeMask", UINT)]


class D3D12_RESOURCE_DESC(C.Structure):
    _fields_ = [
        ("Dimension", UINT),
        ("Alignment", UINT64),
        ("Width", UINT64),
        ("Height", UINT),
        ("DepthOrArraySize", C.c_uint16),
        ("MipLevels", C.c_uint16),
        ("Format", UINT),
        ("SampleDesc", DXGI_SAMPLE_DESC),
        ("Layout", UINT),
        ("Flags", UINT),
    ]


class D3D12_HEAP_DESC(C.Structure):
    _fields_ = [("SizeInBytes", UINT64), ("Properties", D3D12_HEAP_PROPERTIES), ("Alignment", UINT64), ("Flags", UINT)]


class D3D12_RANGE(C.Structure):
    _fields_ = [("Begin", SIZE_T), ("End", SIZE_T)]


class D3D12_RESOURCE_ALLOCATION_INFO(C.Structure):
    _fields_ = [("SizeInBytes", UINT64), ("Alignment", UINT64)]


class D3D12_SHADER_BYTECODE(C.Structure):
    _fields_ = [("pShaderBytecode", C.c_void_p), ("BytecodeLength", SIZE_T)]


class D3D12_CACHED_PIPELINE_STATE(C.Structure):
    _fields_ = [("pCachedBlob", C.c_void_p), ("CachedBlobSizeInBytes", SIZE_T)]


class D3D12_COMPUTE_PIPELINE_STATE_DESC(C.Structure):
    _fields_ = [
        ("pRootSignature", C.c_void_p),
        ("CS", D3D12_SHADER_BYTECODE),
        ("NodeMask", UINT),
        ("CachedPSO", D3D12_CACHED_PIPELINE_STATE),
        ("Flags", UINT),
    ]


class D3D12_DESCRIPTOR_RANGE1(C.Structure):
    _fields_ = [
        ("RangeType", UINT),
        ("NumDescriptors", UINT),
        ("BaseShaderRegister", UINT),
        ("RegisterSpace", UINT),
        ("Flags", UINT),
        ("OffsetInDescriptorsFromTableStart", UINT),
    ]


class D3D12_ROOT_DESCRIPTOR_TABLE1(C.Structure):
    _fields_ = [("NumDescriptorRanges", UINT), ("pDescriptorRanges", C.POINTER(D3D12_DESCRIPTOR_RANGE1))]


class D3D12_ROOT_CONSTANTS(C.Structure):
    _fields_ = [("ShaderRegister", UINT), ("RegisterSpace", UINT), ("Num32BitValues", UINT)]


class D3D12_ROOT_DESCRIPTOR1(C.Structure):
    _fields_ = [("ShaderRegister", UINT), ("RegisterSpace", UINT), ("Flags", UINT)]


class D3D12_ROOT_PARAMETER1_UNION(C.Union):
    _fields_ = [("DescriptorTable", D3D12_ROOT_DESCRIPTOR_TABLE1), ("Constants", D3D12_ROOT_CONSTANTS), ("Descriptor", D3D12_ROOT_DESCRIPTOR1)]


class D3D12_ROOT_PARAMETER1(C.Structure):
    _anonymous_ = ("u",)
    _fields_ = [("ParameterType", UINT), ("u", D3D12_ROOT_PARAMETER1_UNION), ("ShaderVisibility", UINT)]


class D3D12_ROOT_SIGNATURE_DESC1(C.Structure):
    _fields_ = [
        ("NumParameters", UINT),
        ("pParameters", C.POINTER(D3D12_ROOT_PARAMETER1)),
        ("NumStaticSamplers", UINT),
        ("pStaticSamplers", C.c_void_p),
        ("Flags", UINT),
    ]


class D3D12_VERSIONED_ROOT_SIGNATURE_DESC_UNION(C.Union):
    _fields_ = [("Desc_1_1", D3D12_ROOT_SIGNATURE_DESC1)]


class D3D12_VERSIONED_ROOT_SIGNATURE_DESC(C.Structure):
    _anonymous_ = ("u",)
    _fields_ = [("Version", UINT), ("u", D3D12_VERSIONED_ROOT_SIGNATURE_DESC_UNION)]


class D3D12_DESCRIPTOR_HEAP_DESC(C.Structure):
    _fields_ = [("Type", UINT), ("NumDescriptors", UINT), ("Flags", UINT), ("NodeMask", UINT)]


class D3D12_CPU_DESCRIPTOR_HANDLE(C.Structure):
    _fields_ = [("ptr", SIZE_T)]


class D3D12_GPU_DESCRIPTOR_HANDLE(C.Structure):
    _fields_ = [("ptr", UINT64)]


class D3D12_BUFFER_SRV(C.Structure):
    _fields_ = [("FirstElement", UINT64), ("NumElements", UINT), ("StructureByteStride", UINT), ("Flags", UINT)]


class D3D12_SHADER_RESOURCE_VIEW_DESC_UNION(C.Union):
    _fields_ = [("Buffer", D3D12_BUFFER_SRV), ("_size", UINT64 * 3)]


class D3D12_SHADER_RESOURCE_VIEW_DESC(C.Structure):
    _anonymous_ = ("u",)
    _fields_ = [("Format", UINT), ("ViewDimension", UINT), ("Shader4ComponentMapping", UINT), ("u", D3D12_SHADER_RESOURCE_VIEW_DESC_UNION)]


class D3D12_BUFFER_UAV(C.Structure):
    _fields_ = [("FirstElement", UINT64), ("NumElements", UINT), ("StructureByteStride", UINT), ("CounterOffsetInBytes", UINT64), ("Flags", UINT)]


class D3D12_UNORDERED_ACCESS_VIEW_DESC_UNION(C.Union):
    _fields_ = [("Buffer", D3D12_BUFFER_UAV), ("_size", UINT64 * 5)]


class D3D12_UNORDERED_ACCESS_VIEW_DESC(C.Structure):
    _anonymous_ = ("u",)
    _fields_ = [("Format", UINT), ("ViewDimension", UINT), ("u", D3D12_UNORDERED_ACCESS_VIEW_DESC_UNION)]


class D3D12_GLOBAL_BARRIER(C.Structure):
    _fields_ = [("SyncBefore", UINT), ("SyncAfter", UINT), ("AccessBefore", UINT), ("AccessAfter", UINT)]


class D3D12_BUFFER_BARRIER(C.Structure):
    _fields_ = [
        ("SyncBefore", UINT),
        ("SyncAfter", UINT),
        ("AccessBefore", UINT),
        ("AccessAfter", UINT),
        ("pResource", C.c_void_p),
        ("Offset", UINT64),
        ("Size", UINT64),
    ]


class D3D12_BARRIER_GROUP_UNION(C.Union):
    _fields_ = [("pGlobalBarriers", C.POINTER(D3D12_GLOBAL_BARRIER)), ("pBufferBarriers", C.POINTER(D3D12_BUFFER_BARRIER))]


class D3D12_BARRIER_GROUP(C.Structure):
    _anonymous_ = ("u",)
    _fields_ = [("Type", UINT), ("NumBarriers", UINT), ("u", D3D12_BARRIER_GROUP_UNION)]


class D3D12_INDIRECT_CONSTANT(C.Structure):
    _fields_ = [("RootParameterIndex", UINT), ("DestOffsetIn32BitValues", UINT), ("Num32BitValuesToSet", UINT)]


class D3D12_INDIRECT_ROOT_DESCRIPTOR(C.Structure):
    _fields_ = [("RootParameterIndex", UINT)]


class D3D12_INDIRECT_ARGUMENT_DESC_UNION(C.Union):
    _fields_ = [
        ("Constant", D3D12_INDIRECT_CONSTANT),
        ("ConstantBufferView", D3D12_INDIRECT_ROOT_DESCRIPTOR),
        ("ShaderResourceView", D3D12_INDIRECT_ROOT_DESCRIPTOR),
        ("UnorderedAccessView", D3D12_INDIRECT_ROOT_DESCRIPTOR),
    ]


class D3D12_INDIRECT_ARGUMENT_DESC(C.Structure):
    _anonymous_ = ("u",)
    _fields_ = [("Type", UINT), ("u", D3D12_INDIRECT_ARGUMENT_DESC_UNION)]


class D3D12_COMMAND_SIGNATURE_DESC(C.Structure):
    _fields_ = [("ByteStride", UINT), ("NumArgumentDescs", UINT), ("pArgumentDescs", C.POINTER(D3D12_INDIRECT_ARGUMENT_DESC)), ("NodeMask", UINT)]


class D3D12_QUERY_HEAP_DESC(C.Structure):
    _fields_ = [("Type", UINT), ("Count", UINT), ("NodeMask", UINT)]


class D3D12_FEATURE_DATA_D3D12_OPTIONS(C.Structure):
    _fields_ = [
        ("DoublePrecisionFloatShaderOps", BOOL),
        ("OutputMergerLogicOp", BOOL),
        ("MinPrecisionSupport", UINT),
        ("TiledResourcesTier", UINT),
        ("ResourceBindingTier", UINT),
        ("PSSpecifiedStencilRefSupported", BOOL),
        ("TypedUAVLoadAdditionalFormats", BOOL),
        ("ROVsSupported", BOOL),
        ("ConservativeRasterizationTier", UINT),
        ("MaxGPUVirtualAddressBitsPerResource", UINT),
        ("StandardSwizzle64KBSupported", BOOL),
        ("CrossNodeSharingTier", UINT),
        ("CrossAdapterRowMajorTextureSupported", BOOL),
        ("VPAndRTArrayIndexFromAnyShaderFeedingRasterizerSupportedWithoutGSEmulation", BOOL),
        ("ResourceHeapTier", UINT),
    ]


class D3D12_FEATURE_DATA_D3D12_OPTIONS1(C.Structure):
    _fields_ = [
        ("WaveOps", BOOL),
        ("WaveLaneCountMin", UINT),
        ("WaveLaneCountMax", UINT),
        ("TotalLaneCount", UINT),
        ("ExpandedComputeResourceStates", BOOL),
        ("Int64ShaderOps", BOOL),
    ]


class D3D12_FEATURE_DATA_D3D12_OPTIONS4(C.Structure):
    _fields_ = [("MSAA64KBAlignedTextureSupported", BOOL), ("SharedResourceCompatibilityTier", UINT), ("Native16BitShaderOpsSupported", BOOL)]


class D3D12_FEATURE_DATA_D3D12_OPTIONS12(C.Structure):
    _fields_ = [
        ("MSPrimitivesPipelineStatisticIncludesCulledPrimitives", UINT),
        ("EnhancedBarriersSupported", BOOL),
        ("RelaxedFormatCastingSupported", BOOL),
    ]


class D3D12_FEATURE_DATA_SHADER_MODEL(C.Structure):
    _fields_ = [("HighestShaderModel", UINT)]


class D3D12_FEATURE_DATA_ROOT_SIGNATURE(C.Structure):
    _fields_ = [("HighestVersion", UINT)]


class D3D12_FEATURE_DATA_LINEAR_ALGEBRA_SUPPORT(C.Structure):
    _fields_ = [("LinearAlgebraTier", UINT)]


# Preview matrix-query datatype values differ from HLSL dxil::ComponentType.
D3D12_FEATURE_LINEAR_ALGEBRA_LINEAR_ALGEBRA_MATRIX_OPERATION_SUPPORT = 78
D3D12_LINEAR_ALGEBRA_DATATYPE_SINT16, D3D12_LINEAR_ALGEBRA_DATATYPE_UINT16 = 2, 3
D3D12_LINEAR_ALGEBRA_DATATYPE_SINT32, D3D12_LINEAR_ALGEBRA_DATATYPE_UINT32 = 4, 5
D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT16, D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT32 = 7, 8
D3D12_LINEAR_ALGEBRA_DATATYPE_SINT8, D3D12_LINEAR_ALGEBRA_DATATYPE_UINT8 = 18, 19
D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT8_E4M3FN, D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT8_E5M2 = 20, 21
D3D12_LINEAR_ALGEBRA_MULTIPLICATION_SUPPORT_FLAG_SUPPORTED = 1


class D3D12_LINEAR_ALGEBRA_MATRIX_MULTIPLY_SHAPE(C.Structure):
    _fields_ = [("M", UINT), ("K", UINT), ("N", UINT)]


class D3D12_LINEAR_ALGEBRA_WAVE_MATRIX_MULTIPLY_INPUTS(C.Structure):
    _fields_ = [("WaveSize", UINT), ("MatrixAComponentType", UINT), ("MatrixBComponentType", UINT), ("AccumulatorComponentType", UINT)]


class D3D12_LINEAR_ALGEBRA_WAVE_MATRIX_MULTIPLY_SUPPORT(C.Structure):
    _fields_ = [
        ("Inputs", D3D12_LINEAR_ALGEBRA_WAVE_MATRIX_MULTIPLY_INPUTS),
        ("SupportFlags", UINT),
        ("NumShapes", UINT),
        ("Shapes", C.POINTER(D3D12_LINEAR_ALGEBRA_MATRIX_MULTIPLY_SHAPE)),
    ]


class D3D12_LINEAR_ALGEBRA_THREADGROUP_MATRIX_MULTIPLY_SUPPORT(C.Structure):
    _fields_ = [
        ("WaveInputs", D3D12_LINEAR_ALGEBRA_WAVE_MATRIX_MULTIPLY_INPUTS),
        ("Shape", D3D12_LINEAR_ALGEBRA_MATRIX_MULTIPLY_SHAPE),
        ("SupportFlags", UINT),
        ("MinThreadGroupSize", UINT),
        ("MaxThreadGroupSize", UINT),
        ("PreferredThreadGroupSize", UINT),
    ]


class D3D12_LINEAR_ALGEBRA_THREAD_VECTOR_MATRIX_MULTIPLY_SUPPORT(C.Structure):
    _fields_ = [("VectorInputType", UINT), ("MatrixInputType", UINT), ("BiasInputType", UINT), ("VectorResultType", UINT), ("SupportFlags", UINT)]


class D3D12_FEATURE_DATA_LINEAR_ALGEBRA_MATRIX_OPERATION_SUPPORT_UNION(C.Union):
    _fields_ = [
        ("WaveMatrixMultiply", D3D12_LINEAR_ALGEBRA_WAVE_MATRIX_MULTIPLY_SUPPORT),
        ("ThreadGroupMatrixMultiply", D3D12_LINEAR_ALGEBRA_THREADGROUP_MATRIX_MULTIPLY_SUPPORT),
        ("ThreadVectorMatrixMultiply", D3D12_LINEAR_ALGEBRA_THREAD_VECTOR_MATRIX_MULTIPLY_SUPPORT),
    ]


class D3D12_FEATURE_DATA_LINEAR_ALGEBRA_MATRIX_OPERATION_SUPPORT(C.Structure):
    _anonymous_ = ("u",)
    _fields_ = [("OperationType", UINT), ("u", D3D12_FEATURE_DATA_LINEAR_ALGEBRA_MATRIX_OPERATION_SUPPORT_UNION)]
