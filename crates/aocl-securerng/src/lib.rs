//! Safe wrappers for AOCL-SecureRNG.
//!
//! Wraps the AMD secure-RNG entry points around the x86 `RDRAND`
//! (already conditioned, suitable for general-purpose random) and
//! `RDSEED` (raw entropy, suitable for seeding cryptographic generators)
//! instructions.

#![warn(missing_debug_implementations)]
#![cfg_attr(docsrs, feature(doc_cfg))]

pub use aocl_error::{Error, Result};
use aocl_securerng_sys as sys;

const SECRNG_SUCCESS: i32 = 2;
const SECRNG_SUPPORTED: i32 = 1;
const SECRNG_NOT_SUPPORTED: i32 = -1;
const SECRNG_FAILURE: i32 = -2;
const SECRNG_INVALID_INPUT: i32 = -3;

/// Default retry budget AOCL recommends for transient failures of `RDRAND` /
/// `RDSEED`.
pub const DEFAULT_RETRY_COUNT: u32 = 10;

fn check_status(component: &'static str, raw: i32) -> Result<()> {
    if raw == SECRNG_SUCCESS {
        return Ok(());
    }
    let message = match raw {
        SECRNG_NOT_SUPPORTED => "instruction not supported on this CPU",
        SECRNG_FAILURE => "RNG instruction failed after all retries",
        SECRNG_INVALID_INPUT => "invalid input",
        _ => "unknown SecureRNG status",
    }
    .to_string();
    Err(Error::Status {
        component,
        code: raw as i64,
        message,
    })
}

/// Returns `true` if the host CPU exposes the `RDRAND` instruction.
pub fn is_rdrand_supported() -> bool {
    unsafe { sys::is_RDRAND_supported() == SECRNG_SUPPORTED }
}

/// Returns `true` if the host CPU exposes the `RDSEED` instruction.
pub fn is_rdseed_supported() -> bool {
    unsafe { sys::is_RDSEED_supported() == SECRNG_SUPPORTED }
}

/// Read a single random `u16` from `RDRAND`.
pub fn rdrand_u16(retry_count: u32) -> Result<u16> {
    let mut v: u16 = 0;
    let raw = unsafe { sys::get_rdrand16u(&mut v, retry_count) };
    check_status("securerng", raw)?;
    Ok(v)
}

/// Read a single random `u32` from `RDRAND`.
pub fn rdrand_u32(retry_count: u32) -> Result<u32> {
    let mut v: u32 = 0;
    let raw = unsafe { sys::get_rdrand32u(&mut v, retry_count) };
    check_status("securerng", raw)?;
    Ok(v)
}

/// Read a single random `u64` from `RDRAND`.
pub fn rdrand_u64(retry_count: u32) -> Result<u64> {
    let mut v: u64 = 0;
    let raw = unsafe { sys::get_rdrand64u(&mut v, retry_count) };
    check_status("securerng", raw)?;
    Ok(v)
}

/// Read a single random `u16` from `RDSEED`.
pub fn rdseed_u16(retry_count: u32) -> Result<u16> {
    let mut v: u16 = 0;
    let raw = unsafe { sys::get_rdseed16u(&mut v, retry_count) };
    check_status("securerng", raw)?;
    Ok(v)
}

/// Read a single random `u32` from `RDSEED`.
pub fn rdseed_u32(retry_count: u32) -> Result<u32> {
    let mut v: u32 = 0;
    let raw = unsafe { sys::get_rdseed32u(&mut v, retry_count) };
    check_status("securerng", raw)?;
    Ok(v)
}

/// Read a single random `u64` from `RDSEED`.
pub fn rdseed_u64(retry_count: u32) -> Result<u64> {
    let mut v: u64 = 0;
    let raw = unsafe { sys::get_rdseed64u(&mut v, retry_count) };
    check_status("securerng", raw)?;
    Ok(v)
}

/// Fill `out` with random bytes from `RDRAND`.
pub fn rdrand_bytes(out: &mut [u8], retry_count: u32) -> Result<()> {
    if out.is_empty() {
        return Ok(());
    }
    let n: u32 = out.len().try_into().map_err(|_| {
        Error::InvalidArgument(format!(
            "rdrand_bytes: length {} exceeds u32::MAX",
            out.len()
        ))
    })?;
    let raw = unsafe { sys::get_rdrand_bytes_arr(out.as_mut_ptr(), n, retry_count) };
    check_status("securerng", raw)
}

/// Fill `out` with random bytes from `RDSEED`.
pub fn rdseed_bytes(out: &mut [u8], retry_count: u32) -> Result<()> {
    if out.is_empty() {
        return Ok(());
    }
    let n: u32 = out.len().try_into().map_err(|_| {
        Error::InvalidArgument(format!(
            "rdseed_bytes: length {} exceeds u32::MAX",
            out.len()
        ))
    })?;
    let raw = unsafe { sys::get_rdseed_bytes_arr(out.as_mut_ptr(), n, retry_count) };
    check_status("securerng", raw)
}

/// Fill `out` with random `u32`s from `RDRAND`.
pub fn rdrand_u32_array(out: &mut [u32], retry_count: u32) -> Result<()> {
    if out.is_empty() {
        return Ok(());
    }
    let n: u32 = out.len().try_into().map_err(|_| {
        Error::InvalidArgument(format!(
            "rdrand_u32_array: length {} exceeds u32::MAX",
            out.len()
        ))
    })?;
    let raw = unsafe { sys::get_rdrand32u_arr(out.as_mut_ptr(), n, retry_count) };
    check_status("securerng", raw)
}

/// Fill `out` with random `u64`s from `RDRAND`.
pub fn rdrand_u64_array(out: &mut [u64], retry_count: u32) -> Result<()> {
    if out.is_empty() {
        return Ok(());
    }
    let n: u32 = out.len().try_into().map_err(|_| {
        Error::InvalidArgument(format!(
            "rdrand_u64_array: length {} exceeds u32::MAX",
            out.len()
        ))
    })?;
    let raw = unsafe { sys::get_rdrand64u_arr(out.as_mut_ptr(), n, retry_count) };
    check_status("securerng", raw)
}

/// Library version reported by AOCL-SecureRNG.
pub fn version() -> Option<String> {
    unsafe {
        let p = sys::get_secrngversion();
        if p.is_null() {
            return None;
        }
        Some(std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_support_query_is_consistent() {
        let _ = is_rdrand_supported();
        let _ = is_rdseed_supported();
    }

    #[test]
    fn rdrand_yields_distinct_values_when_supported() {
        if !is_rdrand_supported() {
            eprintln!("RDRAND not supported on this CPU; skipping");
            return;
        }
        let a = rdrand_u64(DEFAULT_RETRY_COUNT).unwrap();
        let b = rdrand_u64(DEFAULT_RETRY_COUNT).unwrap();
        assert_ne!(a, b, "RDRAND returned the same u64 twice");
    }

    #[test]
    fn rdrand_bytes_fills_buffer() {
        if !is_rdrand_supported() {
            return;
        }
        let mut buf = [0u8; 32];
        rdrand_bytes(&mut buf, DEFAULT_RETRY_COUNT).unwrap();
        assert!(buf.iter().any(|&b| b != 0));
    }

    #[test]
    fn empty_inputs_are_ok() {
        let mut empty: [u8; 0] = [];
        rdrand_bytes(&mut empty, DEFAULT_RETRY_COUNT).unwrap();
        let mut empty32: [u32; 0] = [];
        rdrand_u32_array(&mut empty32, DEFAULT_RETRY_COUNT).unwrap();
    }

    #[test]
    fn version_string_is_non_empty() {
        if let Some(v) = version() {
            assert!(!v.is_empty());
        }
    }
}
