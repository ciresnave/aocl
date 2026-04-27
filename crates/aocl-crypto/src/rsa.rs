//! ALCP RSA: public-key encryption / decryption and PKCS#1 v1.5 / PSS
//! signatures over 1024-bit and 2048-bit keys.
//!
//! The flow mirrors the underlying ALCP API:
//!
//! 1. [`Rsa::new`] requests a context.
//! 2. [`Rsa::set_public_key`] (with `(exponent, modulus)`) and/or
//!    [`Rsa::set_private_key`] (with the CRT components `(dp, dq, p, q,
//!    qInv, modulus)`) install the key material on the handle.
//! 3. Encrypt / decrypt with [`Rsa::encrypt`] / [`Rsa::decrypt`]
//!    (raw padding-NONE) or [`Rsa::encrypt_oaep`] /
//!    [`Rsa::decrypt_oaep`] for OAEP padding (paired with
//!    [`Rsa::add_digest`] to set the OAEP hash).
//! 4. Sign / verify with [`Rsa::sign_pss`] / [`Rsa::verify_pss`] (PSS)
//!    or the PKCS#1 v1.5 variants.

use aocl_crypto_sys as sys;
use aocl_error::{Error, Result};

use crate::digest::Mode as DigestMode;

/// Padding scheme for [`Rsa::encrypt`] / [`Rsa::decrypt`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Padding {
    /// OAEP padding (must be paired with [`Rsa::add_digest`]).
    Oaep,
    /// No padding — input length must equal the key size in bytes.
    None,
}

impl Padding {
    fn raw(self) -> sys::alc_rsa_padding {
        match self {
            Padding::Oaep => sys::alc_rsa_padding_ALCP_RSA_PADDING_OAEP,
            Padding::None => sys::alc_rsa_padding_ALCP_RSA_PADDING_NONE,
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
            message: format!("ALCP RSA returned alc_error_t = {err:#x}"),
        })
    }
}

/// An RSA session.
pub struct Rsa {
    handle: sys::_alc_rsa_handle,
    _context: Box<[u8]>,
}

impl Rsa {
    /// Request a new RSA handle.
    pub fn new() -> Result<Self> {
        let context_size = unsafe { sys::alcp_rsa_context_size() } as usize;
        if context_size == 0 {
            return Err(Error::AllocationFailed("crypto"));
        }
        let mut context = vec![0u8; context_size].into_boxed_slice();
        let mut handle = sys::_alc_rsa_handle {
            context: context.as_mut_ptr() as *mut std::os::raw::c_void,
        };
        check(unsafe { sys::alcp_rsa_request(&mut handle) })?;
        Ok(Self { handle, _context: context })
    }

    /// Install a public key. `exponent` is typically `65537`. `modulus`
    /// is the big-endian byte representation of `n`; its length (e.g.
    /// 256 for a 2048-bit key) is also the RSA "key size" in bytes.
    pub fn set_public_key(&mut self, exponent: u64, modulus: &[u8]) -> Result<()> {
        check(unsafe {
            sys::alcp_rsa_set_publickey(
                &mut self.handle,
                exponent,
                modulus.as_ptr(),
                modulus.len() as u64,
            )
        })
    }

    /// Install a private key in CRT form. `dp = d mod (p-1)`,
    /// `dq = d mod (q-1)`, `qinv = q⁻¹ mod p`. `modulus` is `n`.
    /// All byte slices must be the same length (key size in bytes / 2
    /// for `dp`, `dq`, `p`, `q`, `qinv`; full key size for `modulus`).
    #[allow(clippy::too_many_arguments)]
    pub fn set_private_key(
        &mut self,
        dp: &[u8],
        dq: &[u8],
        p: &[u8],
        q: &[u8],
        qinv: &[u8],
        modulus: &[u8],
    ) -> Result<()> {
        let size = modulus.len() as u64;
        check(unsafe {
            sys::alcp_rsa_set_privatekey(
                &mut self.handle,
                dp.as_ptr(), dq.as_ptr(), p.as_ptr(), q.as_ptr(),
                qinv.as_ptr(), modulus.as_ptr(),
                size,
            )
        })
    }

    /// Set the digest used for OAEP padding (encrypt / decrypt) and
    /// PSS sign / verify.
    pub fn add_digest(&mut self, digest: DigestMode) -> Result<()> {
        check(unsafe { sys::alcp_rsa_add_digest(&mut self.handle, digest.raw()) })
    }

    /// Set the digest used by the OAEP / PSS mask-generation function.
    pub fn add_mgf(&mut self, digest: DigestMode) -> Result<()> {
        check(unsafe { sys::alcp_rsa_add_mgf(&mut self.handle, digest.raw()) })
    }

    /// Key size in bytes.
    pub fn key_size_bytes(&mut self) -> usize {
        unsafe { sys::alcp_rsa_get_key_size(&mut self.handle) as usize }
    }

    /// Encrypt with the public key. Plaintext length must equal the key
    /// size in bytes (use OAEP for shorter inputs).
    pub fn encrypt(&mut self, plaintext: &[u8], ciphertext: &mut [u8]) -> Result<()> {
        check(unsafe {
            sys::alcp_rsa_publickey_encrypt(
                &mut self.handle,
                plaintext.as_ptr(),
                plaintext.len() as u64,
                ciphertext.as_mut_ptr(),
            )
        })
    }

    /// Decrypt with the private key. `padding` must match the encrypt
    /// side.
    pub fn decrypt(&mut self, padding: Padding, ciphertext: &[u8], plaintext: &mut [u8]) -> Result<()> {
        check(unsafe {
            sys::alcp_rsa_privatekey_decrypt(
                &mut self.handle,
                padding.raw(),
                ciphertext.as_ptr(),
                ciphertext.len() as u64,
                plaintext.as_mut_ptr(),
            )
        })
    }

    /// Encrypt with public key + OAEP padding. `seed` must be the
    /// digest output size in bytes (e.g. 32 bytes for SHA-256). Pair
    /// with [`Rsa::add_digest`] / [`Rsa::add_mgf`].
    pub fn encrypt_oaep(
        &mut self,
        plaintext: &[u8],
        label: &[u8],
        seed: &[u8],
        ciphertext: &mut [u8],
    ) -> Result<()> {
        check(unsafe {
            sys::alcp_rsa_publickey_encrypt_oaep(
                &mut self.handle,
                plaintext.as_ptr(),
                plaintext.len() as u64,
                label.as_ptr(),
                label.len() as u64,
                seed.as_ptr(),
                ciphertext.as_mut_ptr(),
            )
        })
    }

    /// Decrypt with private key + OAEP padding. Returns the recovered
    /// plaintext length (usually shorter than the key size).
    pub fn decrypt_oaep(
        &mut self,
        ciphertext: &[u8],
        label: &[u8],
        plaintext: &mut [u8],
    ) -> Result<usize> {
        let mut text_len: u64 = plaintext.len() as u64;
        check(unsafe {
            sys::alcp_rsa_privatekey_decrypt_oaep(
                &mut self.handle,
                ciphertext.as_ptr(),
                ciphertext.len() as u64,
                label.as_ptr(),
                label.len() as u64,
                plaintext.as_mut_ptr(),
                &mut text_len,
            )
        })?;
        Ok(text_len as usize)
    }

    /// PSS sign: produce a signature over `message`. `salt` length is
    /// caller's choice (typically the digest output size).
    pub fn sign_pss(
        &mut self,
        message: &[u8],
        salt: &[u8],
        signature: &mut [u8],
    ) -> Result<()> {
        check(unsafe {
            sys::alcp_rsa_privatekey_sign_pss(
                &mut self.handle,
                1, // check = TRUE per ALCP convention
                message.as_ptr(),
                message.len() as u64,
                salt.as_ptr(),
                salt.len() as u64,
                signature.as_mut_ptr(),
            )
        })
    }

    /// PSS verify: returns `Ok(())` if the signature matches.
    pub fn verify_pss(&mut self, message: &[u8], signature: &[u8]) -> Result<()> {
        check(unsafe {
            sys::alcp_rsa_publickey_verify_pss(
                &mut self.handle,
                message.as_ptr(),
                message.len() as u64,
                signature.as_ptr(),
            )
        })
    }

    /// PKCS#1 v1.5 sign.
    pub fn sign_pkcs1v15(&mut self, message: &[u8], signature: &mut [u8]) -> Result<()> {
        check(unsafe {
            sys::alcp_rsa_privatekey_sign_pkcs1v15(
                &mut self.handle,
                1,
                message.as_ptr(),
                message.len() as u64,
                signature.as_mut_ptr(),
            )
        })
    }

    /// PKCS#1 v1.5 verify.
    pub fn verify_pkcs1v15(&mut self, message: &[u8], signature: &[u8]) -> Result<()> {
        check(unsafe {
            sys::alcp_rsa_publickey_verify_pkcs1v15(
                &mut self.handle,
                message.as_ptr(),
                message.len() as u64,
                signature.as_ptr(),
            )
        })
    }
}

impl Drop for Rsa {
    fn drop(&mut self) {
        unsafe { sys::alcp_rsa_finish(&mut self.handle) };
    }
}

impl std::fmt::Debug for Rsa {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Rsa").finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rsa_request_and_drop() {
        // Smoke: just allocating + dropping the handle should not blow up.
        let _rsa = Rsa::new().unwrap();
    }
}
