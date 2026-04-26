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

// =========================================================================
//   Per-codec native APIs
// =========================================================================

/// LZ4 codec: fast, byte-stream compression.
pub mod lz4 {
    use super::*;

    /// Maximum possible compressed size for `src_len` bytes of input.
    pub fn compress_bound(src_len: usize) -> Result<usize> {
        let n: i32 = src_len.try_into().map_err(|_| {
            Error::InvalidArgument(format!(
                "lz4::compress_bound: src_len {src_len} exceeds i32::MAX"
            ))
        })?;
        let bound = unsafe { sys::LZ4_compressBound(n) };
        if bound <= 0 {
            return Err(Error::InvalidArgument(format!(
                "lz4::compress_bound: input too large for LZ4 (bound={bound})"
            )));
        }
        Ok(bound as usize)
    }

    /// Compress with the default acceleration. Returns the number of bytes
    /// written to `dst`.
    pub fn compress(src: &[u8], dst: &mut [u8]) -> Result<usize> {
        let src_size: i32 = src.len().try_into().map_err(|_| {
            Error::InvalidArgument("lz4::compress: src too large for LZ4".into())
        })?;
        let dst_cap: i32 = dst.len().try_into().map_err(|_| {
            Error::InvalidArgument("lz4::compress: dst too large for LZ4".into())
        })?;
        let n = unsafe {
            sys::LZ4_compress_default(
                src.as_ptr() as *const std::os::raw::c_char,
                dst.as_mut_ptr() as *mut std::os::raw::c_char,
                src_size,
                dst_cap,
            )
        };
        if n <= 0 {
            return Err(Error::Status {
                component: "compression",
                code: n as i64,
                message: "LZ4_compress_default returned non-positive size".into(),
            });
        }
        Ok(n as usize)
    }

    /// Compress with the high-compression algorithm. `level` is in 1..=12;
    /// higher = better ratio at higher cost.
    pub fn compress_hc(src: &[u8], dst: &mut [u8], level: u8) -> Result<usize> {
        let src_size: i32 = src.len().try_into().map_err(|_| {
            Error::InvalidArgument("lz4::compress_hc: src too large for LZ4".into())
        })?;
        let dst_cap: i32 = dst.len().try_into().map_err(|_| {
            Error::InvalidArgument("lz4::compress_hc: dst too large for LZ4".into())
        })?;
        if !(1..=12).contains(&level) {
            return Err(Error::InvalidArgument(format!(
                "lz4::compress_hc: level={level} out of [1, 12]"
            )));
        }
        let n = unsafe {
            sys::LZ4_compress_HC(
                src.as_ptr() as *const std::os::raw::c_char,
                dst.as_mut_ptr() as *mut std::os::raw::c_char,
                src_size,
                dst_cap,
                level as i32,
            )
        };
        if n <= 0 {
            return Err(Error::Status {
                component: "compression",
                code: n as i64,
                message: "LZ4_compress_HC returned non-positive size".into(),
            });
        }
        Ok(n as usize)
    }

    /// Decompress LZ4 stream into `dst`. Returns the number of bytes
    /// written.
    pub fn decompress(src: &[u8], dst: &mut [u8]) -> Result<usize> {
        let src_size: i32 = src.len().try_into().map_err(|_| {
            Error::InvalidArgument("lz4::decompress: src too large for LZ4".into())
        })?;
        let dst_cap: i32 = dst.len().try_into().map_err(|_| {
            Error::InvalidArgument("lz4::decompress: dst too large for LZ4".into())
        })?;
        let n = unsafe {
            sys::LZ4_decompress_safe(
                src.as_ptr() as *const std::os::raw::c_char,
                dst.as_mut_ptr() as *mut std::os::raw::c_char,
                src_size,
                dst_cap,
            )
        };
        if n < 0 {
            return Err(Error::Status {
                component: "compression",
                code: n as i64,
                message: format!("LZ4_decompress_safe returned {n}"),
            });
        }
        Ok(n as usize)
    }
}

/// Snappy codec: fast byte-stream compression with explicit validation.
pub mod snappy {
    use super::*;

    fn check_status(status: sys::snappy_status) -> Result<()> {
        if status == sys::snappy_status_SNAPPY_OK {
            return Ok(());
        }
        let msg = if status == sys::snappy_status_SNAPPY_INVALID_INPUT {
            "invalid Snappy input"
        } else if status == sys::snappy_status_SNAPPY_BUFFER_TOO_SMALL {
            "Snappy output buffer too small"
        } else {
            "unknown Snappy status"
        };
        Err(Error::Status {
            component: "compression",
            code: status as i64,
            message: msg.into(),
        })
    }

    /// Maximum possible compressed size for `src_len` bytes.
    pub fn max_compressed_length(src_len: usize) -> usize {
        unsafe { sys::snappy_max_compressed_length(src_len) }
    }

    /// Recover the size required to decompress `src`.
    pub fn uncompressed_length(src: &[u8]) -> Result<usize> {
        let mut out: usize = 0;
        let status = unsafe {
            sys::snappy_uncompressed_length(
                src.as_ptr() as *const std::os::raw::c_char,
                src.len(),
                &mut out,
            )
        };
        check_status(status)?;
        Ok(out)
    }

    /// Validate that `src` is a complete Snappy stream.
    pub fn validate(src: &[u8]) -> Result<()> {
        let status = unsafe {
            sys::snappy_validate_compressed_buffer(
                src.as_ptr() as *const std::os::raw::c_char,
                src.len(),
            )
        };
        check_status(status)
    }

    /// Compress `src` into `dst`. Returns the number of bytes written.
    /// `dst` must be at least [`max_compressed_length`].
    pub fn compress(src: &[u8], dst: &mut [u8]) -> Result<usize> {
        let mut written: usize = dst.len();
        let status = unsafe {
            sys::snappy_compress(
                src.as_ptr() as *const std::os::raw::c_char,
                src.len(),
                dst.as_mut_ptr() as *mut std::os::raw::c_char,
                &mut written,
            )
        };
        check_status(status)?;
        Ok(written)
    }

    /// Decompress `src` into `dst`. Pre-size `dst` using
    /// [`uncompressed_length`].
    pub fn uncompress(src: &[u8], dst: &mut [u8]) -> Result<usize> {
        let mut written: usize = dst.len();
        let status = unsafe {
            sys::snappy_uncompress(
                src.as_ptr() as *const std::os::raw::c_char,
                src.len(),
                dst.as_mut_ptr() as *mut std::os::raw::c_char,
                &mut written,
            )
        };
        check_status(status)?;
        Ok(written)
    }
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

    // --- Native codec APIs ----------------------------------------------

    #[test]
    fn lz4_native_round_trip() {
        let payload = b"hello world".repeat(100);
        let bound = lz4::compress_bound(payload.len()).unwrap();
        let mut compressed = vec![0u8; bound];
        let nc = lz4::compress(&payload, &mut compressed).unwrap();
        compressed.truncate(nc);

        let mut decompressed = vec![0u8; payload.len()];
        let nd = lz4::decompress(&compressed, &mut decompressed).unwrap();
        assert_eq!(nd, payload.len());
        assert_eq!(&decompressed[..nd], &payload[..]);
    }

    #[test]
    fn lz4_hc_higher_ratio() {
        let payload = b"the quick brown fox\n".repeat(200);
        let bound = lz4::compress_bound(payload.len()).unwrap();
        let mut default_buf = vec![0u8; bound];
        let mut hc_buf = vec![0u8; bound];
        let n_default = lz4::compress(&payload, &mut default_buf).unwrap();
        let n_hc = lz4::compress_hc(&payload, &mut hc_buf, 12).unwrap();
        // HC should compress at least as well as default for repetitive
        // input.
        assert!(n_hc <= n_default, "HC={n_hc} > default={n_default}");
    }

    #[test]
    fn lz4_hc_rejects_bad_level() {
        let payload = b"abc";
        let mut buf = vec![0u8; 64];
        let err = lz4::compress_hc(payload, &mut buf, 0).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)));
        let err = lz4::compress_hc(payload, &mut buf, 13).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)));
    }

    #[test]
    fn snappy_native_round_trip() {
        let payload = b"snappy sample input".repeat(50);
        let mut compressed = vec![0u8; snappy::max_compressed_length(payload.len())];
        let nc = snappy::compress(&payload, &mut compressed).unwrap();
        compressed.truncate(nc);

        // Validation should accept it.
        snappy::validate(&compressed).unwrap();

        // Recover size first, then decompress.
        let n_uncomp = snappy::uncompressed_length(&compressed).unwrap();
        assert_eq!(n_uncomp, payload.len());
        let mut decompressed = vec![0u8; n_uncomp];
        let nd = snappy::uncompress(&compressed, &mut decompressed).unwrap();
        assert_eq!(nd, payload.len());
        assert_eq!(decompressed, payload);
    }

    #[test]
    fn snappy_validate_rejects_garbage() {
        let bogus = vec![0xFF_u8; 10];
        assert!(snappy::validate(&bogus).is_err());
    }
}
