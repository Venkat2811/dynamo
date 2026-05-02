# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from dynamo.vllm.main import get_wombatkv_shared_cache_runtime_data

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
]


def _config(model: str = "Qwen/Qwen3-0.6B", served_model_name: str | None = None):
    return SimpleNamespace(model=model, served_model_name=served_model_name)


def _vllm_config(
    *,
    backend: str = "wombatkv",
    offloading_size: float | None = 1.0,
    extra_config: dict[str, object] | None = None,
    tp_size: int = 1,
    pp_size: int = 1,
):
    return SimpleNamespace(
        cache_config=SimpleNamespace(
            kv_offloading_backend=backend,
            kv_offloading_size=offloading_size,
        ),
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config=extra_config or {},
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp_size,
            pipeline_parallel_size=pp_size,
        ),
    )


def test_wombatkv_runtime_data_is_absent_without_wombatkv_offload():
    assert (
        get_wombatkv_shared_cache_runtime_data(
            _config(),
            _vllm_config(backend="native"),
            {"block_size": 16},
        )
        is None
    )
    assert (
        get_wombatkv_shared_cache_runtime_data(
            _config(),
            _vllm_config(backend="wombatkv", offloading_size=None),
            {"block_size": 16},
        )
        is None
    )


def test_wombatkv_runtime_data_uses_extra_config():
    data = get_wombatkv_shared_cache_runtime_data(
        _config(served_model_name="served-qwen"),
        _vllm_config(
            backend="tensorpuffer",
            extra_config={
                "namespace": "bench-ns",
                "shared_cache_endpoint": "http://127.0.0.1:9234/check_blocks",
                "shared_cache_timeout_ms": "75",
                "model_fingerprint": "model-digest",
                "layout_fingerprint": "layout-digest",
            },
            tp_size=2,
        ),
        {"block_size": 16},
    )

    assert data == {
        "backend": "wombatkv",
        "block_size": 16,
        "namespace": "bench-ns",
        "endpoint": "http://127.0.0.1:9234/check_blocks",
        "timeout_ms": 75,
        "model_fingerprint": "model-digest",
        "layout_fingerprint": "layout-digest",
    }


def test_wombatkv_runtime_data_env_overrides(monkeypatch):
    monkeypatch.setenv("DYN_WOMBATKV_NAMESPACE", "env-ns")
    monkeypatch.setenv("DYN_WOMBATKV_SHARED_CACHE_ENDPOINT", "127.0.0.1:9999")
    monkeypatch.setenv("DYN_WOMBATKV_MODEL_FINGERPRINT", "env-model")
    monkeypatch.setenv("DYN_WOMBATKV_LAYOUT_FINGERPRINT", "env-layout")

    data = get_wombatkv_shared_cache_runtime_data(
        _config(model="base-model"),
        _vllm_config(extra_config={"namespace": "ignored"}),
        {"block_size": 32},
    )

    assert data == {
        "backend": "wombatkv",
        "block_size": 32,
        "namespace": "env-ns",
        "endpoint": "127.0.0.1:9999",
        "model_fingerprint": "env-model",
        "layout_fingerprint": "env-layout",
    }
