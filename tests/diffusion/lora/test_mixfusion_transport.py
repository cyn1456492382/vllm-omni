from __future__ import annotations

import pytest
import torch

from vllm_omni.diffusion.lora.mixfusion_transport import (
    TransportDescriptor,
    TransportError,
    TransportUnavailableError,
    create_transport_endpoint,
    inspect_transport_capability,
)


def _descriptor(*, request_id: str = "request-0") -> TransportDescriptor:
    return TransportDescriptor(
        request_id=request_id,
        step_id=1,
        block_id="block-0",
        tile_id=0,
        shape=(2, 3),
        dtype="torch.float32",
        adapter_ids=("adapter-0",),
        ranks=(2,),
        payload_bytes=24,
        buffer_token="buffer-0",
        completion_sequence=7,
    )


def test_loopback_transfers_registered_tensor() -> None:
    endpoint = create_transport_endpoint("p2p")
    endpoint.connect()
    source = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    destination = torch.zeros_like(source)
    registration = endpoint.register_buffer("buffer-0", destination)

    assert registration.bytes == source.numel() * source.element_size()
    assert registration.registered_gpu_buffer is False
    assert registration.payload_not_host_staged is False

    descriptor = _descriptor()
    send = endpoint.send_async(descriptor, source)
    receive = endpoint.recv_async(descriptor, destination)
    endpoint.wait(send, timeout=1.0)
    endpoint.wait(receive, timeout=1.0)

    torch.testing.assert_close(destination, source)
    endpoint.deregister_buffer("buffer-0")
    endpoint.close()


def test_loopback_rejects_mismatched_work_metadata() -> None:
    endpoint = create_transport_endpoint("loopback")
    endpoint.connect()
    source = torch.ones((2, 3), dtype=torch.float32)
    destination = torch.zeros_like(source)

    send = endpoint.send_async(_descriptor(), source)
    endpoint.recv_async(_descriptor(request_id="other-request"), destination)

    with pytest.raises(TransportError, match="same work"):
        endpoint.wait(send, timeout=1.0)


def test_rdma_capability_fails_closed_without_declaration() -> None:
    capability = inspect_transport_capability("rdma")

    assert capability.available is False
    assert capability.registered_gpu_buffer is False
    assert "not declared" in capability.reason
    with pytest.raises(TransportUnavailableError):
        create_transport_endpoint("rdma")


def test_p2p_contract_does_not_claim_hardware_data_plane() -> None:
    capability = inspect_transport_capability("p2p")

    assert capability.available is True
    assert capability.hardware_data_plane is False
    assert capability.payload_not_host_staged is False
    assert "contract only" in capability.reason


def test_descriptor_rejects_adapter_rank_mismatch() -> None:
    descriptor = _descriptor()
    invalid = TransportDescriptor(
        **{
            **descriptor.__dict__,
            "ranks": (),
        }
    )

    with pytest.raises(ValueError, match="adapter and rank"):
        invalid.validate()
