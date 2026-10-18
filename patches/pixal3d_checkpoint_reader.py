"""Read safetensors one tensor at a time without mapping the whole file.

Format: https://github.com/safetensors/safetensors#format
"""

import json
import math

import torch


def _unique_fields(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate safetensors header key: {key}")
        result[key] = value
    return result


def load_checkpoint(path, expected_state, device="cpu"):
    dtype_names = {
        "F64": "float64", "F32": "float32", "F16": "float16", "BF16": "bfloat16",
        "I64": "int64", "I32": "int32", "I16": "int16", "I8": "int8",
        "U8": "uint8", "BOOL": "bool", "C64": "complex64", "C128": "complex128",
        "F8_E4M3": "float8_e4m3fn", "F8_E5M2": "float8_e5m2",
        "U16": "uint16", "U32": "uint32", "U64": "uint64",
    }
    with open(path, "rb") as checkpoint:
        checkpoint.seek(0, 2)
        file_size = checkpoint.tell()
        checkpoint.seek(0)
        prefix = checkpoint.read(8)
        if len(prefix) != 8:
            raise ValueError("Truncated safetensors header")
        header_size = int.from_bytes(prefix, "little")
        if not 0 < header_size <= min(100_000_000, file_size - 8):
            raise ValueError("Invalid safetensors header size")
        header = json.loads(checkpoint.read(header_size), object_pairs_hook=_unique_fields)
        if not isinstance(header, dict):
            raise ValueError("Invalid safetensors header")
        metadata = header.pop("__metadata__", {})
        if not isinstance(metadata, dict) or not all(isinstance(v, str) for v in metadata.values()):
            raise ValueError("Invalid safetensors metadata")
        if set(header) != set(expected_state):
            raise ValueError("Pixal3D checkpoint keys do not match the model")
        data_start = 8 + header_size
        entries = []
        for key, entry in header.items():
            shape = entry["shape"]
            if not isinstance(shape, list) or any(type(d) is not int or d < 0 for d in shape):
                raise ValueError(f"Invalid checkpoint shape for {key}")
            if tuple(shape) != tuple(expected_state[key].shape):
                raise ValueError(f"Checkpoint shape does not match the model for {key}")
            dtype = getattr(torch, dtype_names.get(entry["dtype"], ""), None)
            if dtype is None:
                raise ValueError(f"Unsupported checkpoint dtype for {key}: {entry['dtype']}")
            offsets = entry["data_offsets"]
            if not isinstance(offsets, list) or len(offsets) != 2 or any(type(v) is not int for v in offsets):
                raise ValueError(f"Invalid checkpoint offsets for {key}")
            begin, end = offsets
            byte_size = math.prod(shape) * torch.empty(0, dtype=dtype).element_size()
            if not 0 <= begin <= end <= file_size - data_start or end - begin != byte_size:
                raise ValueError(f"Invalid checkpoint data range for {key}")
            entries.append((begin, end, key, shape, dtype))
        cursor = 0
        for begin, end, *_ in sorted(entries):
            if begin != cursor:
                raise ValueError("Checkpoint data ranges overlap or contain holes")
            cursor = end
        if cursor != file_size - data_start:
            raise ValueError("Checkpoint contains unindexed data")

        state = {}
        for begin, end, key, shape, dtype in sorted(entries):
            checkpoint.seek(data_start + begin)
            if end == begin:
                tensor = torch.empty(shape, dtype=dtype, device="cpu")
            else:
                buffer = bytearray(end - begin)
                if checkpoint.readinto(buffer) != len(buffer):
                    raise ValueError(f"Truncated checkpoint tensor: {key}")
                tensor = torch.frombuffer(buffer, dtype=dtype).reshape(shape)
            state[key] = tensor.to(device=device, dtype=expected_state[key].dtype)
        return state
