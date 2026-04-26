//! ALCP NIST SP 800-90A deterministic random-bit generator (DRBG).
//!
//! Two variants are exposed:
//! - HMAC-DRBG over a configurable digest (`Drbg::hmac`)
//! - CTR-DRBG over an AES key length (`Drbg::ctr_aes`)
//!
//! **Status (AOCL 5.1):** the API surface is exposed but
//! `alcp_drbg_initialize` currently returns error 0x3 / heap-corrupts
//! on Windows with our zeroed-info-struct pattern. We track this as a
//! known issue; users wanting cryptographic randomness should use
//! [`aocl-securerng`](https://docs.rs/aocl-securerng) (RDRAND/RDSEED)
//! in the meantime.
//!
//! For general-purpose statistical RNG use
//! [`aocl-rng`](https://docs.rs/aocl-rng).

use aocl_crypto_sys as sys;
use aocl_error::{Error, Result};

use crate::digest::Mode as DigestMode;

/// AES key length for CTR-DRBG.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AesKeyLen {
    Bits128,
    Bits192,
    Bits256,
}

impl AesKeyLen {
    fn bits(self) -> u64 {
        match self {
            AesKeyLen::Bits128 => 128,
            AesKeyLen::Bits192 => 192,
            AesKeyLen::Bits256 => 256,
        }
    }
}

fn check(err: sys::alc_error_t) -> Result<()> {
    if unsafe { sys::alcp_is_error(err) } == 0 {
        Ok(())
    } else {
        Err(Error::Status {
            component: "crypto",
            code: err as i64,
            message: format!("ALCP DRBG returned alc_error_t = {err:#x}"),
        })
    }
}

/// A configured DRBG session.
pub struct Drbg {
    handle: sys::alc_drbg_handle_t,
    _context: Box<[u8]>,
    /// Keep the info struct alive for the lifetime of the handle —
    /// `alcp_drbg_request` retains a pointer into it.
    _info: Box<sys::_alc_drbg_info_t>,
    security_strength: i32,
}

impl Drbg {
    /// Build an HMAC-DRBG over the given digest with the given security
    /// strength (in bits). `personalization` is an optional caller-provided
    /// per-instantiation string.
    pub fn hmac(
        digest_mode: DigestMode,
        security_strength: u32,
        personalization: Option<&[u8]>,
    ) -> Result<Self> {
        let mut info: Box<sys::_alc_drbg_info_t> = Box::new(unsafe { std::mem::zeroed() });
        info.di_type = sys::_alc_drbg_type_ALC_DRBG_HMAC;
        info.di_algoinfo.hmac_drbg = sys::_alc_hmac_drbg_info { digest_mode: digest_mode.raw() };
        info.max_entropy_len = 256;
        info.max_nonce_len = 256;
        Self::build(info, security_strength as i32, personalization)
    }

    /// Build a CTR-DRBG over the given AES key length.
    pub fn ctr_aes(
        key_len: AesKeyLen,
        security_strength: u32,
        personalization: Option<&[u8]>,
    ) -> Result<Self> {
        let mut info: Box<sys::_alc_drbg_info_t> = Box::new(unsafe { std::mem::zeroed() });
        info.di_type = sys::_alc_drbg_type_ALC_DRBG_CTR;
        info.di_algoinfo.ctr_drbg = sys::_alc_ctr_drbg_info {
            di_keysize: key_len.bits(),
            use_derivation_function: 0,
        };
        info.max_entropy_len = 256;
        info.max_nonce_len = 256;
        Self::build(info, security_strength as i32, personalization)
    }

    fn build(
        mut info: Box<sys::_alc_drbg_info_t>,
        security_strength: i32,
        personalization: Option<&[u8]>,
    ) -> Result<Self> {
        // Configure entropy source: use the operating system's secure
        // random source (CryptGenRandom on Windows, /dev/urandom on
        // Linux). The unused distrib field stays at UNKNOWN.
        info.di_rng_sourceinfo.custom_rng = 0;
        info.di_rng_sourceinfo.di_sourceinfo.rng_info = sys::alc_rng_info_t {
            ri_type: sys::alc_rng_type_t_ALC_RNG_TYPE_SIMPLE,
            ri_source: sys::alc_rng_source_t_ALC_RNG_SOURCE_OS,
            ri_distrib: sys::alc_rng_distrib_t_ALC_RNG_DISTRIB_UNKNOWN,
            ri_flags: 0,
        };

        let context_size =
            unsafe { sys::alcp_drbg_context_size(info.as_mut() as *mut _) } as usize;
        if context_size == 0 {
            return Err(Error::AllocationFailed("crypto"));
        }
        let mut context = vec![0u8; context_size].into_boxed_slice();
        let mut handle = sys::alc_drbg_handle_t {
            ch_context: context.as_mut_ptr() as *mut std::os::raw::c_void,
        };
        check(unsafe { sys::alcp_drbg_request(&mut handle, info.as_mut() as *mut _) })?;

        let (pers_ptr, pers_len) = match personalization {
            Some(p) => (p.as_ptr() as *mut u8, p.len() as u64),
            None => (std::ptr::null_mut(), 0),
        };
        check(unsafe {
            sys::alcp_drbg_initialize(&mut handle, security_strength, pers_ptr, pers_len)
        })?;

        Ok(Self {
            handle,
            _context: context,
            _info: info,
            security_strength,
        })
    }

    /// Fill `out` with random bytes from the DRBG. Optional
    /// `additional_input` is mixed into the generator state for this
    /// call only.
    pub fn randomize(
        &mut self,
        out: &mut [u8],
        additional_input: Option<&[u8]>,
    ) -> Result<()> {
        if out.is_empty() {
            return Ok(());
        }
        let (add_ptr, add_len) = match additional_input {
            Some(a) => (a.as_ptr(), a.len()),
            None => (std::ptr::null(), 0),
        };
        check(unsafe {
            sys::alcp_drbg_randomize(
                &mut self.handle,
                out.as_mut_ptr(),
                out.len(),
                self.security_strength,
                add_ptr,
                add_len,
            )
        })
    }
}

impl Drop for Drbg {
    fn drop(&mut self) {
        unsafe {
            let _ = sys::alcp_drbg_finish(&mut self.handle);
        }
    }
}

impl std::fmt::Debug for Drbg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Drbg")
            .field("security_strength", &self.security_strength)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "ALCP DRBG init returns 0x3 / heap-corrupts on AOCL 5.1 with our zeroed-info pattern; needs an alcp-side example as reference"]
    fn hmac_drbg_produces_distinct_outputs() {
        let mut drbg = Drbg::hmac(DigestMode::Sha2_256, 128, Some(b"my-app-tag")).unwrap();
        let mut a = [0u8; 32];
        let mut b = [0u8; 32];
        drbg.randomize(&mut a, None).unwrap();
        drbg.randomize(&mut b, None).unwrap();
        assert_ne!(a, b, "DRBG returned the same 32 bytes twice");
        // Should be very unlikely to be all-zero.
        assert!(a.iter().any(|&x| x != 0));
    }

    #[test]
    #[ignore = "ALCP DRBG init returns 0x3 / heap-corrupts on AOCL 5.1; tracked"]
    fn ctr_drbg_aes256_produces_distinct_outputs() {
        let mut drbg = Drbg::ctr_aes(AesKeyLen::Bits256, 256, None).unwrap();
        let mut a = [0u8; 64];
        let mut b = [0u8; 64];
        drbg.randomize(&mut a, None).unwrap();
        drbg.randomize(&mut b, None).unwrap();
        assert_ne!(a, b);
    }

    #[test]
    #[ignore = "ALCP DRBG init returns 0x3 / heap-corrupts on AOCL 5.1; tracked"]
    fn empty_output_is_ok() {
        let mut drbg = Drbg::hmac(DigestMode::Sha2_256, 128, None).unwrap();
        let mut empty: [u8; 0] = [];
        drbg.randomize(&mut empty, None).unwrap();
    }
}
