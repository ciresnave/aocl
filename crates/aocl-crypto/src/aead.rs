//! ALCP authenticated-encryption-with-associated-data (AEAD) primitives.
//!
//! AEAD modes encrypt plaintext to ciphertext + authentication tag in
//! one step. Optional **associated data** (AAD) is also authenticated
//! by the tag but is not encrypted. Decryption recomputes the tag and
//! returns an error on mismatch — the caller must check the returned
//! `Result` and not trust the plaintext on failure.
//!
//! Modes covered:
//! - AES-GCM (`Mode::AesGcm`) — 12-byte IV, 16-byte tag (default).
//! - AES-CCM (`Mode::AesCcm`) — variable IV (7..=13 bytes), variable
//!   tag (4..=16 bytes); requires [`Aead::set_ccm_lengths`] before
//!   `set_aad`.
//! - AES-SIV (`Mode::AesSiv`) — synthetic-IV; tag length 16 bytes.
//! - ChaCha20-Poly1305 (`Mode::ChaCha20Poly1305`) — 12-byte nonce,
//!   16-byte tag.

use aocl_crypto_sys as sys;
use aocl_error::{Error, Result};

/// AEAD mode and (where applicable) associated-block-cipher key length.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Mode {
    /// AES Galois/Counter Mode.
    AesGcm,
    /// AES Counter-with-CBC-MAC.
    AesCcm,
    /// AES Synthetic Initialisation Vector (RFC 5297).
    AesSiv,
    /// ChaCha20 + Poly1305 (RFC 8439).
    ChaCha20Poly1305,
}

impl Mode {
    fn raw(self) -> sys::alc_cipher_mode_t {
        match self {
            Mode::AesGcm => sys::_alc_cipher_mode_ALC_AES_MODE_GCM,
            Mode::AesCcm => sys::_alc_cipher_mode_ALC_AES_MODE_CCM,
            Mode::AesSiv => sys::_alc_cipher_mode_ALC_AES_MODE_SIV,
            Mode::ChaCha20Poly1305 => sys::_alc_cipher_mode_ALC_CHACHA20_POLY1305,
        }
    }

    /// Default tag length in bytes for this mode.
    pub const fn default_tag_len(self) -> usize {
        match self {
            // GCM and CCM both default to 16 bytes; CCM allows 4-16.
            // SIV is fixed 16. ChaCha20-Poly1305 is fixed 16.
            _ => 16,
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
            message: format!("ALCP AEAD returned alc_error_t = {err:#x}"),
        })
    }
}

/// AEAD session. Build with [`Aead::new`], optionally call
/// [`Aead::set_ccm_lengths`] (CCM only) and [`Aead::set_aad`], then
/// [`Aead::encrypt`] / [`Aead::get_tag`] for one direction or
/// [`Aead::decrypt`] for the other. Decryption error checking is the
/// caller's responsibility — see [`Aead::verify_and_decrypt`] for a
/// helper.
pub struct Aead {
    handle: sys::alc_cipher_handle_t,
    _context: Box<[u8]>,
}

impl Aead {
    /// Create an AEAD session for `mode` with the given key (bit-length =
    /// `key.len() * 8`) and IV / nonce.
    ///
    /// Key length per mode:
    /// - AES-GCM / AES-CCM: 16, 24, or 32 bytes
    /// - AES-SIV: 32, 48, or 64 bytes (twice the AES key length, since
    ///   SIV uses two AES keys internally)
    /// - ChaCha20-Poly1305: exactly 32 bytes
    pub fn new(mode: Mode, key: &[u8], iv: &[u8]) -> Result<Self> {
        let key_len_bits = (key.len() as u64).saturating_mul(8);
        let context_size = unsafe { sys::alcp_cipher_aead_context_size() } as usize;
        if context_size == 0 {
            return Err(Error::AllocationFailed("crypto"));
        }
        let mut context = vec![0u8; context_size].into_boxed_slice();
        let mut handle = sys::alc_cipher_handle_t {
            ch_context: context.as_mut_ptr() as *mut std::os::raw::c_void,
        };
        check(unsafe { sys::alcp_cipher_aead_request(mode.raw(), key_len_bits, &mut handle) })?;
        check(unsafe {
            sys::alcp_cipher_aead_init(
                &mut handle,
                key.as_ptr(),
                key_len_bits,
                iv.as_ptr(),
                iv.len() as u64,
            )
        })?;
        Ok(Self { handle, _context: context })
    }

    /// CCM-only: declare the plaintext length and tag length up front
    /// (CCM needs both before processing AAD/plaintext). For other
    /// modes this is a no-op.
    pub fn set_ccm_lengths(&mut self, plaintext_len: u64, tag_len: usize) -> Result<()> {
        check(unsafe {
            sys::alcp_cipher_aead_set_ccm_plaintext_length(&mut self.handle, plaintext_len)
        })?;
        check(unsafe {
            sys::alcp_cipher_aead_set_tag_length(&mut self.handle, tag_len as u64)
        })?;
        Ok(())
    }

    /// Provide associated data (authenticated, not encrypted). May be
    /// called once. Some modes (CCM) require this before encrypt /
    /// decrypt; for GCM it can be skipped if AAD is not used.
    pub fn set_aad(&mut self, aad: &[u8]) -> Result<()> {
        check(unsafe {
            sys::alcp_cipher_aead_set_aad(&mut self.handle, aad.as_ptr(), aad.len() as u64)
        })
    }

    /// Encrypt `plaintext` to `ciphertext` (same length). Call
    /// [`Aead::get_tag`] afterwards to retrieve the authentication tag.
    pub fn encrypt(&mut self, plaintext: &[u8], ciphertext: &mut [u8]) -> Result<()> {
        if plaintext.len() != ciphertext.len() {
            return Err(Error::InvalidArgument(format!(
                "encrypt: plaintext.len()={}, ciphertext.len()={}",
                plaintext.len(), ciphertext.len()
            )));
        }
        check(unsafe {
            sys::alcp_cipher_aead_encrypt(
                &mut self.handle,
                plaintext.as_ptr(),
                ciphertext.as_mut_ptr(),
                plaintext.len() as u64,
            )
        })
    }

    /// Decrypt `ciphertext` to `plaintext`. The caller must then call
    /// [`Aead::get_tag`] and compare to the expected tag bytes — or use
    /// [`Aead::verify_and_decrypt`] which does both.
    pub fn decrypt(&mut self, ciphertext: &[u8], plaintext: &mut [u8]) -> Result<()> {
        if ciphertext.len() != plaintext.len() {
            return Err(Error::InvalidArgument(format!(
                "decrypt: ciphertext.len()={}, plaintext.len()={}",
                ciphertext.len(), plaintext.len()
            )));
        }
        check(unsafe {
            sys::alcp_cipher_aead_decrypt(
                &mut self.handle,
                ciphertext.as_ptr(),
                plaintext.as_mut_ptr(),
                ciphertext.len() as u64,
            )
        })
    }

    /// Read the computed authentication tag into `out`. Call after
    /// [`Aead::encrypt`] / [`Aead::decrypt`].
    pub fn get_tag(&mut self, out: &mut [u8]) -> Result<()> {
        check(unsafe {
            sys::alcp_cipher_aead_get_tag(&mut self.handle, out.as_mut_ptr(), out.len() as u64)
        })
    }

    /// Decrypt and constant-time-compare the computed tag against the
    /// expected one. Returns `Err(Error::Status)` with a tag-mismatch
    /// message when the ciphertext or AAD has been tampered with.
    pub fn verify_and_decrypt(
        &mut self,
        ciphertext: &[u8],
        plaintext: &mut [u8],
        expected_tag: &[u8],
    ) -> Result<()> {
        self.decrypt(ciphertext, plaintext)?;
        let mut computed = vec![0u8; expected_tag.len()];
        self.get_tag(&mut computed)?;
        if !constant_time_eq(&computed, expected_tag) {
            // Zero plaintext on tag mismatch so callers can't accidentally
            // trust leaked bytes.
            for b in plaintext.iter_mut() {
                *b = 0;
            }
            return Err(Error::Status {
                component: "crypto",
                code: -1,
                message: "AEAD tag mismatch — ciphertext or AAD has been tampered with".into(),
            });
        }
        Ok(())
    }
}

impl Drop for Aead {
    fn drop(&mut self) {
        unsafe { sys::alcp_cipher_aead_finish(&mut self.handle) };
    }
}

impl std::fmt::Debug for Aead {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Aead").finish_non_exhaustive()
    }
}

/// Constant-time byte slice equality. Hand-rolled rather than pulling
/// in `subtle` to keep the dependency footprint small — for AEAD tag
/// verification this is the standard pattern.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff: u8 = 0;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aes_gcm_round_trip_no_aad() {
        let key = [0x42_u8; 32]; // AES-256
        let iv = [0x77_u8; 12];
        let plaintext = b"AEAD round-trip test payload, no AAD".to_vec();
        let mut ciphertext = vec![0u8; plaintext.len()];
        let mut tag = [0u8; 16];

        {
            let mut enc = Aead::new(Mode::AesGcm, &key, &iv).unwrap();
            enc.encrypt(&plaintext, &mut ciphertext).unwrap();
            enc.get_tag(&mut tag).unwrap();
        }

        let mut recovered = vec![0u8; plaintext.len()];
        {
            let mut dec = Aead::new(Mode::AesGcm, &key, &iv).unwrap();
            dec.verify_and_decrypt(&ciphertext, &mut recovered, &tag).unwrap();
        }
        assert_eq!(recovered, plaintext);
    }

    #[test]
    fn aes_gcm_with_aad() {
        let key = [0x33_u8; 16]; // AES-128
        let iv = [0x55_u8; 12];
        let aad = b"associated authenticated data".to_vec();
        let plaintext = b"only the plaintext gets encrypted".to_vec();
        let mut ciphertext = vec![0u8; plaintext.len()];
        let mut tag = [0u8; 16];

        {
            let mut enc = Aead::new(Mode::AesGcm, &key, &iv).unwrap();
            enc.set_aad(&aad).unwrap();
            enc.encrypt(&plaintext, &mut ciphertext).unwrap();
            enc.get_tag(&mut tag).unwrap();
        }

        let mut recovered = vec![0u8; plaintext.len()];
        {
            let mut dec = Aead::new(Mode::AesGcm, &key, &iv).unwrap();
            dec.set_aad(&aad).unwrap();
            dec.verify_and_decrypt(&ciphertext, &mut recovered, &tag).unwrap();
        }
        assert_eq!(recovered, plaintext);
    }

    #[test]
    fn aes_gcm_aad_tamper_detected() {
        let key = [0x33_u8; 16];
        let iv = [0x55_u8; 12];
        let aad = b"original AAD".to_vec();
        let plaintext = b"sensitive plaintext".to_vec();
        let mut ciphertext = vec![0u8; plaintext.len()];
        let mut tag = [0u8; 16];

        {
            let mut enc = Aead::new(Mode::AesGcm, &key, &iv).unwrap();
            enc.set_aad(&aad).unwrap();
            enc.encrypt(&plaintext, &mut ciphertext).unwrap();
            enc.get_tag(&mut tag).unwrap();
        }

        // Decrypt with *different* AAD — tag must not match.
        let mut recovered = vec![0u8; plaintext.len()];
        let mut dec = Aead::new(Mode::AesGcm, &key, &iv).unwrap();
        dec.set_aad(b"tampered AAD").unwrap();
        let err = dec.verify_and_decrypt(&ciphertext, &mut recovered, &tag).unwrap_err();
        match err {
            Error::Status { code: -1, .. } => {}
            other => panic!("expected tag-mismatch error, got {other:?}"),
        }
        // Plaintext buffer must have been zeroed.
        assert!(recovered.iter().all(|&b| b == 0));
    }

    #[test]
    fn aes_gcm_ciphertext_tamper_detected() {
        let key = [0u8; 32];
        let iv = [1u8; 12];
        let plaintext = b"payload to integrity-protect".to_vec();
        let mut ciphertext = vec![0u8; plaintext.len()];
        let mut tag = [0u8; 16];
        {
            let mut enc = Aead::new(Mode::AesGcm, &key, &iv).unwrap();
            enc.encrypt(&plaintext, &mut ciphertext).unwrap();
            enc.get_tag(&mut tag).unwrap();
        }

        // Flip one bit in the ciphertext.
        ciphertext[0] ^= 0x01;

        let mut recovered = vec![0u8; plaintext.len()];
        let mut dec = Aead::new(Mode::AesGcm, &key, &iv).unwrap();
        let err = dec.verify_and_decrypt(&ciphertext, &mut recovered, &tag).unwrap_err();
        assert!(matches!(err, Error::Status { .. }));
    }

    #[test]
    fn constant_time_eq_basic() {
        assert!(constant_time_eq(b"abc", b"abc"));
        assert!(!constant_time_eq(b"abc", b"abd"));
        assert!(!constant_time_eq(b"abc", b"abcd"));
    }
}
