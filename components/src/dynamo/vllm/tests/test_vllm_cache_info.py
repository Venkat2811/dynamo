# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from dynamo.vllm.cache_info import (
    DYNAMO_KV_EVENT_BLOCK_SIZE_KEY,
    configure_kv_event_block_size,
    get_configured_kv_event_block_size,
    kv_event_block_size,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.xpu_1,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.timeout(180),  # 0-GiB unit tests, floor 180s
    pytest.mark.pre_merge,
]


def _vllm_config(
    *,
    block_size: int = 16,
    dcp_size: int | None = 1,
    additional_config: dict | None = None,
):
    return SimpleNamespace(
        additional_config=additional_config,
        cache_config=SimpleNamespace(block_size=block_size),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=dcp_size,
        ),
    )


@pytest.mark.parametrize(
    "vllm_config, physical_block_size, expected",
    [
        (SimpleNamespace(), 64, 64),
        (_vllm_config(dcp_size=None), 64, 64),
        (_vllm_config(dcp_size=1), 64, 64),
        (_vllm_config(dcp_size=8), 64, 512),
    ],
)
def test_kv_event_block_size_accounts_for_dcp(
    vllm_config, physical_block_size, expected
):
    assert kv_event_block_size(vllm_config, physical_block_size) == expected


def test_get_configured_kv_event_block_size_does_not_multiply_cached_value_twice():
    vllm_config = _vllm_config(
        dcp_size=8,
        additional_config={DYNAMO_KV_EVENT_BLOCK_SIZE_KEY: 128},
    )

    assert get_configured_kv_event_block_size(vllm_config) == 128


def test_get_configured_kv_event_block_size_uses_dcp_aware_fallback():
    vllm_config = _vllm_config(dcp_size=8)

    assert get_configured_kv_event_block_size(vllm_config) == 128


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind, physical_block_size",
    [
        ("full_attention", 16),
        ("mla_attention", 64),
        ("sink_full_attention", 128),
    ],
)
async def test_configure_kv_event_block_size_applies_dcp_to_main_attention_group(
    kind, physical_block_size
):
    group_metadata = [
        {"kind": "mamba", "block_size": 4096},
        {"kind": kind, "block_size": physical_block_size},
    ]
    engine = SimpleNamespace(
        engine_core=SimpleNamespace(
            call_utility_async=AsyncMock(return_value=group_metadata)
        )
    )
    vllm_config = _vllm_config(dcp_size=2)

    configured = await configure_kv_event_block_size(engine, vllm_config)

    assert configured == physical_block_size * 2
    assert (
        vllm_config.additional_config[DYNAMO_KV_EVENT_BLOCK_SIZE_KEY]
        == physical_block_size * 2
    )
    engine.engine_core.call_utility_async.assert_awaited_once_with(
        "get_kv_cache_group_metadata"
    )


@pytest.mark.asyncio
async def test_configure_kv_event_block_size_applies_dcp_to_fallback():
    engine = SimpleNamespace(
        engine_core=SimpleNamespace(
            call_utility_async=AsyncMock(side_effect=RuntimeError("unsupported"))
        )
    )
    vllm_config = _vllm_config(block_size=16, dcp_size=8)

    configured = await configure_kv_event_block_size(engine, vllm_config)

    assert configured == 128
    assert vllm_config.additional_config[DYNAMO_KV_EVENT_BLOCK_SIZE_KEY] == 128


@pytest.mark.asyncio
async def test_configure_kv_event_block_size_hybrid_mla_with_dcp8():
    """Kimi-K3 geometry on 8 x B300 (vLLM 0.17.2rc1 nightly, TP8 x DCP8): the hybrid KV
    cache manager pads three mamba (KDA) groups and the MLA group to a 1,536-token page,
    while the cache manager hashes attention blocks at 1,536 x 8 = 12,288 tokens.
    Without the DCP factor every attention BlockStored event was dropped by the
    publisher ("Block size must be 1536 tokens to be published. Block size is: 12288")
    and the router indexed nothing."""
    group_metadata = [
        {"group_idx": 0, "kind": "mamba", "block_size": 1536, "sliding_window": None},
        {"group_idx": 1, "kind": "mamba", "block_size": 1536, "sliding_window": None},
        {"group_idx": 2, "kind": "mamba", "block_size": 1536, "sliding_window": None},
        {"group_idx": 3, "kind": "mla_attention", "block_size": 1536, "sliding_window": None},
    ]
    engine = SimpleNamespace(
        engine_core=SimpleNamespace(
            call_utility_async=AsyncMock(return_value=group_metadata)
        )
    )
    vllm_config = _vllm_config(block_size=1536, dcp_size=8)

    configured = await configure_kv_event_block_size(engine, vllm_config)

    assert configured == 12288
    assert vllm_config.additional_config[DYNAMO_KV_EVENT_BLOCK_SIZE_KEY] == 12288
    assert get_configured_kv_event_block_size(vllm_config) == 12288
