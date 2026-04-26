//! Safe wrappers for AOCL-Compression.

#![warn(missing_debug_implementations)]
#![cfg_attr(docsrs, feature(doc_cfg))]

use aocl_compression_sys as sys;
pub use aocl_error::{Error, Result};

/// Compression algorithms exposed by AOCL's unified API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Codec {
    Lz4,
    Lz4Hc,
    Lzma,
    Bzip2,
    Snappy,
    Zlib,
    Zstd,
}

impl Codec {
    fn raw(self) -> sys::aocl_compression_type {
        match self {
            Codec::Lz4 => sys::aocl_compression_type_LZ4,
            Codec::Lz4Hc => sys::aocl_compression_type_LZ4HC,
            Codec::Lzma => sys::aocl_compression_type_LZMA,
            Codec::Bzip2 => sys::aocl_compression_type_BZIP2,
            Codec::Snappy => sys::aocl_compression_type_SNAPPY,
            Codec::Zlib => sys::aocl_compression_type_ZLIB,
            Codec::Zstd => sys::aocl_compression_type_ZSTD,
        }
    }
}

fn map_error(err: i64) -> Error {
    let message = match err as i32 {
        sys::aocl_error_type_ERR_MEMORY_ALLOC => "memory allocation failure",
        sys::aocl_error_type_ERR_INVALID_INPUT => "invalid input",
        sys::aocl_error_type_ERR_UNSUPPORTED_METHOD => "unsupported method",
        sys::aocl_error_type_ERR_EXCLUDED_METHOD => "method excluded from this build",
        sys::aocl_error_type_ERR_COMPRESSION_FAILED => "compression failed",
        sys::aocl_error_type_ERR_COMPRESSION_INVALID_OUTPUT => "invalid compression output",
        _ => "unknown compression error",
    }
    .to_string();
    Error::Status {
        component: "compression",
        code: err,
        message,
    }
}

fn empty_desc() -> sys::aocl_compression_desc {
    // SAFETY: zeroed bytes are a valid representation for this struct.
    unsafe { std::mem::zeroed() }
}

/// Compute an upper bound on the compressed size for `src_len` bytes.
pub fn compress_bound(codec: Codec, src_len: usize) -> Result<usize> {
    let bound = unsafe { sys::aocl_llc_compressBound(codec.raw(), src_len) };
    if bound < 0 {
        return Err(map_error(bound));
    }
    Ok(bound as usize)
}

/// Compress `src` and return the compressed bytes.
pub fn compress(codec: Codec, src: &[u8], level: u32) -> Result<Vec<u8>> {
    let bound = compress_bound(codec, src.len())?;
    let mut out = vec![0u8; bound];
    let mut desc = empty_desc();
    desc.inBuf = src.as_ptr() as *mut std::os::raw::c_char;
    desc.outBuf = out.as_mut_ptr() as *mut std::os::raw::c_char;
    desc.inSize = src.len();
    desc.outSize = out.len();
    desc.level = level as usize;
    desc.optLevel = 2;
    let raw_codec = codec.raw();

    let setup_status = unsafe { sys::aocl_llc_setup(&mut desc, raw_codec) };
    if setup_status < 0 {
        return Err(map_error(setup_status as i64));
    }
    let written = unsafe { sys::aocl_llc_compress(&mut desc, raw_codec) };
    unsafe { sys::aocl_llc_destroy(&mut desc, raw_codec) };

    if written < 0 {
        return Err(map_error(written));
    }
    out.truncate(written as usize);
    Ok(out)
}

/// Decompress `src` into a buffer of at least `dst_len` bytes.
pub fn decompress(codec: Codec, src: &[u8], dst_len: usize) -> Result<Vec<u8>> {
    let mut out = vec![0u8; dst_len];
    let mut desc = empty_desc();
    desc.inBuf = src.as_ptr() as *mut std::os::raw::c_char;
    desc.outBuf = out.as_mut_ptr() as *mut std::os::raw::c_char;
    desc.inSize = src.len();
    desc.outSize = out.len();
    desc.optLevel = 2;
    let raw_codec = codec.raw();

    let setup_status = unsafe { sys::aocl_llc_setup(&mut desc, raw_codec) };
    if setup_status < 0 {
        return Err(map_error(setup_status as i64));
    }
    let written = unsafe { sys::aocl_llc_decompress(&mut desc, raw_codec) };
    unsafe { sys::aocl_llc_destroy(&mut desc, raw_codec) };

    if written < 0 {
        return Err(map_error(written));
    }
    out.truncate(written as usize);
    Ok(out)
}

/// AOCL-Compression library version string.
pub fn version() -> Option<String> {
    unsafe {
        let p = sys::aocl_llc_version();
        if p.is_null() {
            return None;
        }
        Some(std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(codec: Codec, level: u32, payload: &[u8]) {
        let compressed = compress(codec, payload, level).unwrap();
        let recovered = decompress(codec, &compressed, payload.len()).unwrap();
        assert_eq!(recovered, payload, "{codec:?} round-trip mismatch");
    }

    #[test]
    fn lz4_round_trip() {
        let payload = b"the quick brown fox jumps over the lazy dog\n".repeat(64);
        round_trip(Codec::Lz4, 0, &payload);
    }

    #[test]
    fn zlib_round_trip() {
        let payload = b"AOCL compression test payload, AOCL compression test payload, ".repeat(32);
        round_trip(Codec::Zlib, 6, &payload);
    }

    #[test]
    fn zstd_round_trip() {
        let payload: Vec<u8> = (0..4096_u32).flat_map(|i| (i as u8).to_le_bytes()).collect();
        round_trip(Codec::Zstd, 3, &payload);
    }

    #[test]
    fn snappy_round_trip() {
        let payload = vec![0xAB_u8; 16 * 1024];
        round_trip(Codec::Snappy, 0, &payload);
    }

    #[test]
    fn compress_bound_is_sensible() {
        let n = 4096usize;
        let bound = compress_bound(Codec::Zstd, n).unwrap();
        assert!(bound >= n, "bound {bound} should be at least src len {n}");
    }

    #[test]
    fn version_string_present() {
        if let Some(v) = version() {
            assert!(!v.is_empty());
        }
    }
}
