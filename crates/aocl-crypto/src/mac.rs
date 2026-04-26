//! ALCP Message Authentication Code primitives: HMAC, CMAC, Poly1305.

use aocl_crypto_sys as sys;
use aocl_error::{Error, Result};

use crate::digest::Mode as DigestMode;

/// MAC type and the associated parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum MacKind {
    /// HMAC over the given digest. Output size = digest output size.
    Hmac(DigestMode),
    /// AES-CMAC over the given AES key length (128/192/256-bit). Output
    /// is 16 bytes.
    Cmac(AesKeyLen),
    /// Poly1305 — 16-byte tag, 32-byte one-time key.
    Poly1305,
}

/// AES key length for CMAC.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AesKeyLen {
    Bits128,
    Bits192,
    Bits256,
}

impl MacKind {
    fn raw_type(self) -> sys::alc_mac_type_t {
        match self {
            MacKind::Hmac(_) => sys::_alc_mac_type_ALC_MAC_HMAC,
            MacKind::Cmac(_) => sys::_alc_mac_type_ALC_MAC_CMAC,
            MacKind::Poly1305 => sys::_alc_mac_type_ALC_MAC_POLY1305,
        }
    }

    fn output_len(self) -> usize {
        match self {
            MacKind::Hmac(d) => d.output_len(),
            MacKind::Cmac(_) => 16,
            MacKind::Poly1305 => 16,
        }
    }
}

fn cmac_mode(k: AesKeyLen) -> sys::alc_cipher_mode_t {
    // CMAC info specifies the underlying AES mode; CMAC always uses ECB
    // internally regardless of key length. ALCP only cares about it as
    // a placeholder to dispatch the AES variant.
    let _ = k;
    sys::_alc_cipher_mode_ALC_AES_MODE_ECB
}

fn check(err: sys::alc_error_t) -> Result<()> {
    if unsafe { sys::alcp_is_error(err) } == 0 {
        Ok(())
    } else {
        Err(Error::Status {
            component: "crypto",
            code: err as i64,
            message: format!("ALCP MAC returned alc_error_t = {err:#x}"),
        })
    }
}

/// A configured MAC session. Build with [`Mac::new`], feed bytes via
/// [`Mac::update`], call [`Mac::finalize`] for the tag.
pub struct Mac {
    handle: sys::alc_mac_handle_t,
    _context: Box<[u8]>,
    info: sys::alc_mac_info_t,
    output_len: usize,
    finished: bool,
}

impl Mac {
    /// Build a MAC session for the given kind, with `key` of the
    /// appropriate length:
    /// - HMAC: any key length (typically equal to the digest output).
    /// - CMAC: 16/24/32 bytes matching `AesKeyLen`.
    /// - Poly1305: exactly 32 bytes.
    pub fn new(kind: MacKind, key: &[u8]) -> Result<Self> {
        let context_size = unsafe { sys::alcp_mac_context_size() } as usize;
        if context_size == 0 {
            return Err(Error::AllocationFailed("crypto"));
        }
        let mut context = vec![0u8; context_size].into_boxed_slice();
        let mut handle = sys::alc_mac_handle_t {
            ch_context: context.as_mut_ptr() as *mut std::os::raw::c_void,
        };

        let mut info: sys::alc_mac_info_t = unsafe { std::mem::zeroed() };
        match kind {
            MacKind::Hmac(d) => {
                info.hmac = sys::_alc_hmac_info { digest_mode: d.raw() };
            }
            MacKind::Cmac(k) => {
                info.cmac = sys::_alc_cmac_info { ci_mode: cmac_mode(k) };
            }
            MacKind::Poly1305 => {
                // No info needed; the union is zeroed.
            }
        }

        check(unsafe { sys::alcp_mac_request(&mut handle, kind.raw_type()) })?;
        check(unsafe {
            sys::alcp_mac_init(
                &mut handle,
                key.as_ptr(),
                key.len() as u64,
                &mut info,
            )
        })?;
        Ok(Self {
            handle,
            _context: context,
            info,
            output_len: kind.output_len(),
            finished: false,
        })
    }

    /// Feed a chunk of data into the MAC.
    pub fn update(&mut self, data: &[u8]) -> Result<()> {
        if self.finished {
            return Err(Error::InvalidArgument("update after finalize".into()));
        }
        if data.is_empty() {
            return Ok(());
        }
        check(unsafe {
            sys::alcp_mac_update(&mut self.handle, data.as_ptr(), data.len() as u64)
        })
    }

    /// Finalize and return the authentication tag.
    pub fn finalize(mut self) -> Result<Vec<u8>> {
        let mut tag = vec![0u8; self.output_len];
        check(unsafe {
            sys::alcp_mac_finalize(
                &mut self.handle,
                tag.as_mut_ptr(),
                tag.len() as u64,
            )
        })?;
        self.finished = true;
        Ok(tag)
    }
}

impl Drop for Mac {
    fn drop(&mut self) {
        let _ = self.info; // keep `info` alive for the duration of the handle.
        unsafe { sys::alcp_mac_finish(&mut self.handle) };
    }
}

impl std::fmt::Debug for Mac {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Mac")
            .field("output_len", &self.output_len)
            .field("finished", &self.finished)
            .finish_non_exhaustive()
    }
}

/// One-shot convenience: compute a MAC tag over `data`.
pub fn hmac(digest_mode: DigestMode, key: &[u8], data: &[u8]) -> Result<Vec<u8>> {
    let mut mac = Mac::new(MacKind::Hmac(digest_mode), key)?;
    mac.update(data)?;
    mac.finalize()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(b: &[u8]) -> String {
        let mut s = String::with_capacity(b.len() * 2);
        for byte in b { s.push_str(&format!("{byte:02x}")); }
        s
    }

    #[test]
    fn hmac_sha256_known_answer() {
        // RFC 4231 test case 1: key = 20 × 0x0b, data = "Hi There"
        // → 0xb0344c61d8db38535ca8afceaf0bf12b
        //   0x881dc200c9833da726e9376c2e32cff7
        let key = vec![0x0bu8; 20];
        let tag = hmac(DigestMode::Sha2_256, &key, b"Hi There").unwrap();
        assert_eq!(
            hex(&tag),
            "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7"
        );
    }

    #[test]
    fn hmac_streaming_matches_one_shot() {
        let key = b"secret-key-with-some-length";
        let data = b"the quick brown fox jumps over the lazy dog".to_vec();
        let one_shot = hmac(DigestMode::Sha2_256, key, &data).unwrap();
        let mut mac = Mac::new(MacKind::Hmac(DigestMode::Sha2_256), key).unwrap();
        mac.update(&data[..10]).unwrap();
        mac.update(&data[10..]).unwrap();
        let streamed = mac.finalize().unwrap();
        assert_eq!(streamed, one_shot);
    }
}
