// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! TensorPuffer/WombatKV implementation of [`ObjectBlockOps`].
//!
//! This backend uses TensorPuffer's ABI 1.2 namespace/key byte-store calls
//! directly. Dynamo still owns the KVBM block identity (`SequenceHash`) and
//! physical layout movement; WombatKV supplies the object-tier persistence and
//! local Foyer acceleration behind `(namespace, key)`.

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};
use std::ptr::NonNull;
use std::sync::Arc;

use anyhow::{Result, anyhow};
use bytes::Bytes;
use futures::future::BoxFuture;
use futures::stream::StreamExt;
use libloading::Library;

use crate::object::{DefaultKeyFormatter, KeyFormatter, LayoutConfigExt, ObjectBlockOps};
use crate::{BlockId, SequenceHash};
use kvbm_common::LogicalLayoutHandle;
use kvbm_physical::transfer::PhysicalLayout;

type TpufHandle = c_void;
type TpufBorrow = c_void;
type TpufAbiVersion = unsafe extern "C" fn() -> u32;
type TpufInitFromEnv = unsafe extern "C" fn() -> *mut TpufHandle;
type TpufFree = unsafe extern "C" fn(*mut TpufHandle);
type TpufLastError = unsafe extern "C" fn() -> *const c_char;
type TpufPutKv =
    unsafe extern "C" fn(*mut TpufHandle, *const c_char, *const c_char, *const u8, usize) -> i64;
type TpufGetKvBorrowed = unsafe extern "C" fn(
    *mut TpufHandle,
    *const c_char,
    *const c_char,
    *mut *const u8,
    *mut usize,
    *mut *mut TpufBorrow,
) -> i32;
type TpufReleaseBorrow = unsafe extern "C" fn(*mut TpufBorrow);
type TpufExistsKv = unsafe extern "C" fn(*mut TpufHandle, *const c_char, *const c_char) -> i32;
type TpufRestoreNamespace = unsafe extern "C" fn(*mut TpufHandle, *const c_char) -> i64;

struct TensorPufferAbi {
    _lib: Library,
    tpuf_free: TpufFree,
    tpuf_last_error: TpufLastError,
    tpuf_put_kv: TpufPutKv,
    tpuf_get_kv_borrowed: TpufGetKvBorrowed,
    tpuf_release_borrow: TpufReleaseBorrow,
    tpuf_exists_kv: TpufExistsKv,
    tpuf_restore_namespace: TpufRestoreNamespace,
}

impl TensorPufferAbi {
    fn load(lib_path: Option<&str>) -> Result<(Arc<Self>, *mut TpufHandle)> {
        let lib = unsafe {
            match lib_path {
                Some(path) => Library::new(path)
                    .map_err(|e| anyhow!("failed to load TensorPuffer library {path:?}: {e}"))?,
                None => Library::new("libtensorpuffer.so")
                    .or_else(|_| Library::new("libtensorpuffer.dylib"))
                    .map_err(|e| anyhow!("failed to load libtensorpuffer from loader path: {e}"))?,
            }
        };

        let tpuf_abi_version: TpufAbiVersion = unsafe { load_symbol(&lib, b"tpuf_abi_version")? };
        let version = unsafe { tpuf_abi_version() };
        let major = version >> 16;
        let minor = version & 0xFFFF;
        if major != 1 || minor < 2 {
            anyhow::bail!("libtensorpuffer ABI {major}.{minor} lacks namespace/key KV calls");
        }

        let tpuf_init_from_env: TpufInitFromEnv =
            unsafe { load_symbol(&lib, b"tpuf_init_from_env")? };
        let abi = Arc::new(Self {
            tpuf_free: unsafe { load_symbol(&lib, b"tpuf_free")? },
            tpuf_last_error: unsafe { load_symbol(&lib, b"tpuf_last_error")? },
            tpuf_put_kv: unsafe { load_symbol(&lib, b"tpuf_put_kv")? },
            tpuf_get_kv_borrowed: unsafe { load_symbol(&lib, b"tpuf_get_kv_borrowed")? },
            tpuf_release_borrow: unsafe { load_symbol(&lib, b"tpuf_release_borrow")? },
            tpuf_exists_kv: unsafe { load_symbol(&lib, b"tpuf_exists_kv")? },
            tpuf_restore_namespace: unsafe { load_symbol(&lib, b"tpuf_restore_namespace")? },
            _lib: lib,
        });

        let handle = unsafe { tpuf_init_from_env() };
        if handle.is_null() {
            anyhow::bail!("{}", abi.last_error("tpuf_init_from_env failed"));
        }

        Ok((abi, handle))
    }

    fn last_error(&self, fallback: &str) -> String {
        let ptr = unsafe { (self.tpuf_last_error)() };
        if ptr.is_null() {
            fallback.to_string()
        } else {
            unsafe { CStr::from_ptr(ptr) }
                .to_string_lossy()
                .into_owned()
        }
    }
}

unsafe fn load_symbol<T: Copy>(lib: &Library, name: &[u8]) -> Result<T> {
    let symbol = unsafe { lib.get::<T>(name) }
        .map_err(|e| anyhow!("failed to load TensorPuffer symbol {:?}: {e}", name))?;
    Ok(*symbol)
}

struct TensorPufferHandle {
    abi: Arc<TensorPufferAbi>,
    handle: NonNull<TpufHandle>,
}

// TensorPuffer's public ABI documents that handles are internally synchronized
// and may be shared by multiple threads.
unsafe impl Send for TensorPufferHandle {}
unsafe impl Sync for TensorPufferHandle {}

impl TensorPufferHandle {
    fn new(lib_path: Option<&str>) -> Result<Self> {
        let (abi, handle) = TensorPufferAbi::load(lib_path)?;
        let handle =
            NonNull::new(handle).ok_or_else(|| anyhow!("tpuf_init_from_env returned null"))?;
        Ok(Self { abi, handle })
    }

    fn put_kv(&self, namespace: &str, key: &str, payload: &[u8]) -> Result<usize> {
        let namespace = cstring("namespace", namespace)?;
        let key = cstring("key", key)?;
        let payload_ptr = if payload.is_empty() {
            NonNull::<u8>::dangling().as_ptr() as *const u8
        } else {
            payload.as_ptr()
        };

        let rc = unsafe {
            (self.abi.tpuf_put_kv)(
                self.handle.as_ptr(),
                namespace.as_ptr(),
                key.as_ptr(),
                payload_ptr,
                payload.len(),
            )
        };
        if rc < 0 {
            anyhow::bail!("{}", self.abi.last_error("tpuf_put_kv failed"));
        }
        Ok(rc as usize)
    }

    fn get_kv(&self, namespace: &str, key: &str) -> Result<Option<Bytes>> {
        let namespace = cstring("namespace", namespace)?;
        let key = cstring("key", key)?;
        let mut out_ptr: *const u8 = std::ptr::null();
        let mut out_len: usize = 0;
        let mut borrow: *mut TpufBorrow = std::ptr::null_mut();

        let rc = unsafe {
            (self.abi.tpuf_get_kv_borrowed)(
                self.handle.as_ptr(),
                namespace.as_ptr(),
                key.as_ptr(),
                &mut out_ptr,
                &mut out_len,
                &mut borrow,
            )
        };

        if rc == 0 {
            return Ok(None);
        }
        if rc < 0 {
            anyhow::bail!("{}", self.abi.last_error("tpuf_get_kv_borrowed failed"));
        }

        let bytes = if out_len == 0 {
            Bytes::new()
        } else if out_ptr.is_null() {
            unsafe { (self.abi.tpuf_release_borrow)(borrow) };
            anyhow::bail!("tpuf_get_kv_borrowed returned null pointer for {out_len} bytes");
        } else {
            let slice = unsafe { std::slice::from_raw_parts(out_ptr, out_len) };
            Bytes::copy_from_slice(slice)
        };
        unsafe { (self.abi.tpuf_release_borrow)(borrow) };
        Ok(Some(bytes))
    }

    fn exists_kv(&self, namespace: &str, key: &str) -> Result<bool> {
        let namespace = cstring("namespace", namespace)?;
        let key = cstring("key", key)?;
        let rc = unsafe {
            (self.abi.tpuf_exists_kv)(self.handle.as_ptr(), namespace.as_ptr(), key.as_ptr())
        };
        if rc < 0 {
            anyhow::bail!("{}", self.abi.last_error("tpuf_exists_kv failed"));
        }
        Ok(rc == 1)
    }

    fn restore_namespace(&self, namespace: &str) -> Result<usize> {
        let namespace = cstring("namespace", namespace)?;
        let rc =
            unsafe { (self.abi.tpuf_restore_namespace)(self.handle.as_ptr(), namespace.as_ptr()) };
        if rc < 0 {
            anyhow::bail!("{}", self.abi.last_error("tpuf_restore_namespace failed"));
        }
        Ok(rc as usize)
    }
}

impl Drop for TensorPufferHandle {
    fn drop(&mut self) {
        unsafe { (self.abi.tpuf_free)(self.handle.as_ptr()) };
    }
}

fn cstring(label: &str, value: &str) -> Result<CString> {
    CString::new(value).map_err(|_| anyhow!("{label} contains an embedded NUL byte"))
}

/// WombatKV-backed KVBM object-tier client.
pub struct WombatKvObjectBlockClient {
    store: Arc<TensorPufferHandle>,
    namespace: String,
    key_prefix: Option<String>,
    max_concurrent_requests: usize,
    key_formatter: Arc<dyn KeyFormatter>,
}

impl WombatKvObjectBlockClient {
    /// Create a new client with default key formatting.
    pub fn new(config: kvbm_config::WombatKvObjectConfig) -> Result<Self> {
        Self::with_key_formatter(config, Arc::new(DefaultKeyFormatter))
    }

    /// Create a new client with a custom Dynamo key formatter.
    pub fn with_key_formatter(
        config: kvbm_config::WombatKvObjectConfig,
        key_formatter: Arc<dyn KeyFormatter>,
    ) -> Result<Self> {
        let store = Arc::new(TensorPufferHandle::new(config.lib_path.as_deref())?);
        if config.restore_on_init {
            let restored = store.restore_namespace(&config.namespace)?;
            tracing::info!(
                namespace = %config.namespace,
                restored,
                "restored WombatKV namespace for KVBM object client"
            );
        }

        Ok(Self {
            store,
            namespace: config.namespace,
            key_prefix: config.key_prefix,
            max_concurrent_requests: config.max_concurrent_requests.max(1),
            key_formatter,
        })
    }

    fn object_key(&self, hash: &SequenceHash) -> String {
        format_wombatkv_key(&*self.key_formatter, self.key_prefix.as_deref(), hash)
    }
}

impl ObjectBlockOps for WombatKvObjectBlockClient {
    fn has_blocks(
        &self,
        keys: Vec<SequenceHash>,
    ) -> BoxFuture<'static, Vec<(SequenceHash, Option<usize>)>> {
        let namespace = self.namespace.clone();
        let store = self.store.clone();
        let max_concurrent = self.max_concurrent_requests;
        let work_items: Vec<_> = keys
            .into_iter()
            .map(|key| {
                let object_key = self.object_key(&key);
                (key, object_key)
            })
            .collect();

        Box::pin(async move {
            let tasks = work_items.into_iter().map(|(hash, object_key)| {
                let store = store.clone();
                let namespace = namespace.clone();

                async move {
                    match tokio::task::spawn_blocking(move || {
                        store.exists_kv(&namespace, &object_key)
                    })
                    .await
                    {
                        Ok(Ok(true)) => (hash, Some(0)),
                        Ok(Ok(false)) => (hash, None),
                        Ok(Err(e)) => {
                            tracing::warn!(key = %hash, error = %e, "WombatKV exists failed");
                            (hash, None)
                        }
                        Err(e) => {
                            tracing::warn!(key = %hash, error = %e, "WombatKV exists task failed");
                            (hash, None)
                        }
                    }
                }
            });

            futures::stream::iter(tasks)
                .buffer_unordered(max_concurrent)
                .collect()
                .await
        })
    }

    fn put_blocks(
        &self,
        keys: Vec<SequenceHash>,
        _src_layout: LogicalLayoutHandle,
        _block_ids: Vec<BlockId>,
    ) -> BoxFuture<'static, Vec<Result<SequenceHash, SequenceHash>>> {
        tracing::error!(
            "WombatKvObjectBlockClient::put_blocks called with LogicalLayoutHandle - \
             use put_blocks_with_layout() via a worker that can resolve layouts"
        );
        Box::pin(async move { keys.into_iter().map(Err).collect() })
    }

    fn get_blocks(
        &self,
        keys: Vec<SequenceHash>,
        _dst_layout: LogicalLayoutHandle,
        _block_ids: Vec<BlockId>,
    ) -> BoxFuture<'static, Vec<Result<SequenceHash, SequenceHash>>> {
        tracing::error!(
            "WombatKvObjectBlockClient::get_blocks called with LogicalLayoutHandle - \
             use get_blocks_with_layout() via a worker that can resolve layouts"
        );
        Box::pin(async move { keys.into_iter().map(Err).collect() })
    }

    fn put_blocks_with_layout(
        &self,
        keys: Vec<SequenceHash>,
        layout: PhysicalLayout,
        block_ids: Vec<BlockId>,
    ) -> BoxFuture<'static, Vec<Result<SequenceHash, SequenceHash>>> {
        let config = layout.layout().config();
        let block_size = config.block_size_bytes();
        let region_size = config.region_size();
        let is_contiguous = layout.layout().is_fully_contiguous();
        let namespace = self.namespace.clone();
        let store = self.store.clone();
        let max_concurrent = self.max_concurrent_requests;
        let work_items: Vec<_> = keys
            .into_iter()
            .zip(block_ids)
            .map(|(key, block_id)| {
                let object_key = self.object_key(&key);
                (key, object_key, block_id)
            })
            .collect();

        Box::pin(async move {
            let tasks = work_items.into_iter().map(|(hash, object_key, block_id)| {
                let layout = layout.clone();
                let store = store.clone();
                let namespace = namespace.clone();

                async move {
                    let copy_result = tokio_rayon::spawn(move || {
                        copy_block_to_bytes(
                            &layout,
                            block_id,
                            block_size,
                            region_size,
                            is_contiguous,
                        )
                    })
                    .await;

                    let data = match copy_result {
                        Ok(data) => data,
                        Err(e) => {
                            tracing::warn!(key = %hash, error = %e, "WombatKV block copy failed");
                            return Err(hash);
                        }
                    };
                    let expected_len = data.len();

                    match tokio::task::spawn_blocking(move || {
                        store.put_kv(&namespace, &object_key, &data)
                    })
                    .await
                    {
                        Ok(Ok(written)) if written == expected_len => Ok(hash),
                        Ok(Ok(written)) => {
                            tracing::warn!(
                                key = %hash,
                                written,
                                expected_len,
                                "WombatKV put wrote unexpected byte count"
                            );
                            Err(hash)
                        }
                        Ok(Err(e)) => {
                            tracing::warn!(key = %hash, error = %e, "WombatKV put failed");
                            Err(hash)
                        }
                        Err(e) => {
                            tracing::warn!(key = %hash, error = %e, "WombatKV put task failed");
                            Err(hash)
                        }
                    }
                }
            });

            futures::stream::iter(tasks)
                .buffer_unordered(max_concurrent)
                .collect()
                .await
        })
    }

    fn get_blocks_with_layout(
        &self,
        keys: Vec<SequenceHash>,
        layout: PhysicalLayout,
        block_ids: Vec<BlockId>,
    ) -> BoxFuture<'static, Vec<Result<SequenceHash, SequenceHash>>> {
        let config = layout.layout().config();
        let block_size = config.block_size_bytes();
        let region_size = config.region_size();
        let is_contiguous = layout.layout().is_fully_contiguous();
        let namespace = self.namespace.clone();
        let store = self.store.clone();
        let max_concurrent = self.max_concurrent_requests;
        let work_items: Vec<_> = keys
            .into_iter()
            .zip(block_ids)
            .map(|(key, block_id)| {
                let object_key = self.object_key(&key);
                (key, object_key, block_id)
            })
            .collect();

        Box::pin(async move {
            let tasks = work_items.into_iter().map(|(hash, object_key, block_id)| {
                let layout = layout.clone();
                let store = store.clone();
                let namespace = namespace.clone();

                async move {
                    let data = match tokio::task::spawn_blocking(move || {
                        store.get_kv(&namespace, &object_key)
                    })
                    .await
                    {
                        Ok(Ok(Some(data))) => data,
                        Ok(Ok(None)) => {
                            tracing::warn!(key = %hash, "WombatKV get missed");
                            return Err(hash);
                        }
                        Ok(Err(e)) => {
                            tracing::warn!(key = %hash, error = %e, "WombatKV get failed");
                            return Err(hash);
                        }
                        Err(e) => {
                            tracing::warn!(key = %hash, error = %e, "WombatKV get task failed");
                            return Err(hash);
                        }
                    };

                    match tokio_rayon::spawn(move || {
                        copy_bytes_to_block(
                            &data,
                            &layout,
                            block_id,
                            block_size,
                            region_size,
                            is_contiguous,
                        )
                    })
                    .await
                    {
                        Ok(()) => Ok(hash),
                        Err(e) => {
                            tracing::warn!(key = %hash, error = %e, "WombatKV write-to-layout failed");
                            Err(hash)
                        }
                    }
                }
            });

            futures::stream::iter(tasks)
                .buffer_unordered(max_concurrent)
                .collect()
                .await
        })
    }
}

fn format_wombatkv_key(
    formatter: &dyn KeyFormatter,
    key_prefix: Option<&str>,
    hash: &SequenceHash,
) -> String {
    let formatted = formatter.format_key(hash);
    match key_prefix.filter(|prefix| !prefix.is_empty()) {
        Some(prefix) => format!("{}/{}", prefix.trim_end_matches('/'), formatted),
        None => formatted,
    }
}

fn copy_block_to_bytes(
    layout: &PhysicalLayout,
    block_id: BlockId,
    block_size: usize,
    region_size: usize,
    is_contiguous: bool,
) -> Result<Bytes> {
    if is_contiguous {
        let region = layout.memory_region(block_id, 0, 0)?;
        let slice = unsafe { std::slice::from_raw_parts(region.addr() as *const u8, block_size) };
        Ok(Bytes::copy_from_slice(slice))
    } else {
        let mut buf = Vec::with_capacity(block_size);
        let inner_layout = layout.layout();
        for layer_id in 0..inner_layout.num_layers() {
            for outer_id in 0..inner_layout.outer_dim() {
                let region = layout.memory_region(block_id, layer_id, outer_id)?;
                if region.size() < region_size {
                    return Err(anyhow!(
                        "memory region too small: got {} bytes, need {}",
                        region.size(),
                        region_size
                    ));
                }
                let slice =
                    unsafe { std::slice::from_raw_parts(region.addr() as *const u8, region_size) };
                buf.extend_from_slice(slice);
            }
        }
        Ok(Bytes::from(buf))
    }
}

fn copy_bytes_to_block(
    data: &[u8],
    layout: &PhysicalLayout,
    block_id: BlockId,
    block_size: usize,
    region_size: usize,
    is_contiguous: bool,
) -> Result<()> {
    if is_contiguous {
        if data.len() < block_size {
            return Err(anyhow!(
                "WombatKV data too short: got {} bytes, expected {}",
                data.len(),
                block_size
            ));
        }
        let region = layout.memory_region(block_id, 0, 0)?;
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), region.addr() as *mut u8, block_size);
        }
    } else {
        let mut offset = 0;
        let inner_layout = layout.layout();
        for layer_id in 0..inner_layout.num_layers() {
            for outer_id in 0..inner_layout.outer_dim() {
                if offset + region_size > data.len() {
                    return Err(anyhow!(
                        "WombatKV data too short at offset {}: need {} more bytes, only {} remain",
                        offset,
                        region_size,
                        data.len().saturating_sub(offset)
                    ));
                }
                let region = layout.memory_region(block_id, layer_id, outer_id)?;
                if region.size() < region_size {
                    return Err(anyhow!(
                        "memory region too small: got {} bytes, need {}",
                        region.size(),
                        region_size
                    ));
                }
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        data[offset..].as_ptr(),
                        region.addr() as *mut u8,
                        region_size,
                    );
                }
                offset += region_size;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::RankPrefixedKeyFormatter;

    #[test]
    fn test_format_wombatkv_key_without_prefix() {
        let formatter = DefaultKeyFormatter;
        let hash = SequenceHash::new(0x1234_u64, None, 7);
        assert_eq!(
            format_wombatkv_key(&formatter, None, &hash),
            formatter.format_key(&hash)
        );
    }

    #[test]
    fn test_format_wombatkv_key_with_prefix_and_rank() {
        let formatter = RankPrefixedKeyFormatter::new(3);
        let hash = SequenceHash::new(0x1234_u64, None, 7);
        let key = format_wombatkv_key(&formatter, Some("g4/"), &hash);
        assert!(key.starts_with("g4/3/"));
        assert!(key.ends_with(&hash.to_string()));
    }
}

#[cfg(all(test, feature = "testing"))]
mod live_tests {
    use super::*;
    use kvbm_physical::testing::{create_fc_layout, create_test_agent};
    use kvbm_physical::transfer::StorageKind;

    fn live_enabled() -> bool {
        std::env::var("TPUF_LIVE_WOMBATKV").as_deref() == Ok("1")
    }

    fn test_lib_path() -> Option<String> {
        std::env::var("KVBM_TEST_WOMBATKV_LIB").ok()
    }

    #[test]
    fn live_wombatkv_raw_put_get_exists_roundtrip() {
        if !live_enabled() {
            eprintln!("skipping live WombatKV test; set TPUF_LIVE_WOMBATKV=1");
            return;
        }

        let store = TensorPufferHandle::new(test_lib_path().as_deref())
            .expect("create TensorPuffer handle");
        let namespace = std::env::var("KVBM_TEST_WOMBATKV_NAMESPACE")
            .unwrap_or_else(|_| "kvbm-live-test".to_string());
        let key = format!(
            "raw-roundtrip/{}/{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        );
        let payload = b"dynamo-wombatkv-object-client-live-smoke".to_vec();

        assert!(
            !store.exists_kv(&namespace, &key).expect("initial exists"),
            "fresh key unexpectedly exists"
        );
        let written = store
            .put_kv(&namespace, &key, &payload)
            .expect("put raw WombatKV payload");
        assert_eq!(written, payload.len());
        assert!(store.exists_kv(&namespace, &key).expect("post-put exists"));
        let actual = store
            .get_kv(&namespace, &key)
            .expect("get raw WombatKV payload")
            .expect("payload should exist");
        assert_eq!(actual.as_ref(), &payload[..]);
    }

    #[tokio::test]
    async fn live_wombatkv_put_has_get_roundtrip() {
        if !live_enabled() {
            eprintln!("skipping live WombatKV test; set TPUF_LIVE_WOMBATKV=1");
            return;
        }
        if dynamo_memory::nixl::is_stub() {
            eprintln!("skipping ObjectBlockOps live test; NIXL is in stub mode");
            return;
        }

        let run_id = format!("roundtrip-{}-{}", std::process::id(), uuid::Uuid::new_v4());
        let client = WombatKvObjectBlockClient::new(kvbm_config::WombatKvObjectConfig {
            namespace: std::env::var("KVBM_TEST_WOMBATKV_NAMESPACE")
                .unwrap_or_else(|_| "kvbm-live-test".to_string()),
            key_prefix: Some(run_id),
            lib_path: test_lib_path(),
            max_concurrent_requests: 2,
            restore_on_init: false,
        })
        .expect("create WombatKV object client");

        let agent = create_test_agent("wombatkv_live_roundtrip");
        let layout = create_fc_layout(agent, StorageKind::System, 2);
        let config = layout.layout().config();
        let block_size = config.block_size_bytes();
        let region_size = config.region_size();
        let hash = SequenceHash::new(0xD1E5_EA5E_u64, None, 13);
        let payload = (0..block_size)
            .map(|idx| (idx % 251) as u8)
            .collect::<Vec<_>>();

        copy_bytes_to_block(&payload, &layout, 0, block_size, region_size, true)
            .expect("seed source block");

        let put = client
            .put_blocks_with_layout(vec![hash], layout.clone(), vec![0])
            .await;
        assert!(put[0].is_ok(), "put failed: {:?}", put);

        let exists = client.has_blocks(vec![hash]).await;
        assert_eq!(exists[0].0, hash);
        assert!(exists[0].1.is_some(), "has_blocks missed after put");

        let get = client
            .get_blocks_with_layout(vec![hash], layout.clone(), vec![1])
            .await;
        assert!(get[0].is_ok(), "get failed: {:?}", get);

        let actual = copy_block_to_bytes(&layout, 1, block_size, region_size, true)
            .expect("read back block");
        assert_eq!(actual.as_ref(), &payload[..]);
    }
}
