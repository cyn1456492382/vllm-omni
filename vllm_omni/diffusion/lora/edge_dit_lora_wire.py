# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Wire codecs and TCP frames for edge-owned DiT LoRA residuals."""

from __future__ import annotations

import socket
import struct

import numpy as np
import torch

FRAME_HEADER = struct.Struct("!I")


def encode_activation(x: torch.Tensor, precision: str) -> bytes:
    """Encode activation/delta tensor to wire bytes.

    Args:
        x: Tensor on any device; copied to CPU float32 first.
        precision: One of ``fp16``/``float16``, ``int8``, ``int4``, or fp32.

    Returns:
        Encoded payload bytes.
    """
    arr = x.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    p = precision.lower()
    if p in ("fp16", "float16"):
        return arr.astype(np.float16).tobytes()
    if p == "int8":
        flat = arr.reshape(-1)
        max_abs = float(np.max(np.abs(flat))) if flat.size else 0.0
        scale = (max_abs / 127.0) if max_abs else 1.0
        q = np.clip(np.round(flat / scale), -127, 127).astype(np.int8)
        return q.tobytes() + struct.pack("<f", float(scale)) + struct.pack(
            "<I", int(flat.size)
        )
    if p == "int4":
        flat = arr.reshape(-1)
        max_abs = float(np.max(np.abs(flat))) if flat.size else 0.0
        scale = (max_abs / 7.0) if max_abs else 1.0
        q = np.clip(np.round(flat / scale), -7, 7).astype(np.int8)
        if q.size % 2 == 1:
            q = np.concatenate([q, np.zeros(1, dtype=np.int8)])
        # Two's-complement nibble packing for values in [-7, 7].
        hi = (q[0::2] & 0x0F).astype(np.uint8)
        lo = (q[1::2] & 0x0F).astype(np.uint8)
        packed = (hi << 4) | lo
        return (
            packed.tobytes()
            + struct.pack("<f", float(scale))
            + struct.pack("<I", int(flat.size))
        )
    return arr.astype(np.float32).tobytes()


def decode_activation(
    payload: bytes, shape: tuple[int, ...], precision: str
) -> torch.Tensor:
    """Decode wire bytes to a CPU float32 tensor of ``shape``.

    Args:
        payload: Bytes from ``encode_activation``.
        shape: Expected tensor shape.
        precision: Matching encode precision.

    Returns:
        Float32 CPU tensor.

    Raises:
        ValueError: If the payload is too short for ``shape``.
    """
    p = precision.lower()
    numel = int(np.prod(shape))
    if p == "int8":
        body, scale_b, n_b = payload[:-8], payload[-8:-4], payload[-4:]
        scale = struct.unpack("<f", scale_b)[0]
        n = struct.unpack("<I", n_b)[0]
        q = np.frombuffer(body, dtype=np.int8)[:n].astype(np.float32) * scale
        return torch.from_numpy(q.reshape(shape).copy())
    if p == "int4":
        body, scale_b, n_b = payload[:-8], payload[-8:-4], payload[-4:]
        scale = struct.unpack("<f", scale_b)[0]
        n = struct.unpack("<I", n_b)[0]
        packed = np.frombuffer(body, dtype=np.uint8)
        hi = ((packed >> 4) & 0x0F).astype(np.int8)
        lo = (packed & 0x0F).astype(np.int8)
        hi = np.where(hi > 7, hi - 16, hi)
        lo = np.where(lo > 7, lo - 16, lo)
        q = np.empty(packed.size * 2, dtype=np.int8)
        q[0::2], q[1::2] = hi, lo
        q = q[:n].astype(np.float32) * scale
        return torch.from_numpy(q.reshape(shape).copy())
    if p in ("fp16", "float16"):
        arr = np.frombuffer(payload, dtype=np.float16).astype(np.float32)
    else:
        arr = np.frombuffer(payload, dtype=np.float32).copy()
    if arr.size < numel:
        raise ValueError(f"payload too short: {arr.size} < {numel}")
    return torch.from_numpy(arr[:numel].reshape(shape).copy())


encode_delta = encode_activation
decode_delta = decode_activation


def send_frame(sock: socket.socket, payload: bytes) -> None:
    """Send a length-prefixed TCP frame."""
    sock.sendall(FRAME_HEADER.pack(len(payload)) + payload)


def recv_frame(sock: socket.socket) -> bytes:
    """Receive a length-prefixed TCP frame."""
    header = _recv_exact(sock, FRAME_HEADER.size)
    (n,) = FRAME_HEADER.unpack(header)
    return _recv_exact(sock, n)


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    left = size
    while left:
        chunk = sock.recv(min(65536, left))
        if not chunk:
            raise ConnectionError("socket closed")
        chunks.append(chunk)
        left -= len(chunk)
    return b"".join(chunks)
