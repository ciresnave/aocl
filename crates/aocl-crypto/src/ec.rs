//! ALCP elliptic-curve operations (X25519 ECDH).
//!
//! Currently exposes Diffie-Hellman key agreement over Curve25519.
//! Each party generates a 32-byte private key (e.g. via
//! [`aocl-securerng`](https://docs.rs/aocl-securerng) or any
//! cryptographic RNG), derives a 32-byte public key, exchanges it,
//! and computes a 32-byte shared secret.
//!
//! Use one [`EcDh`] handle per peer through the entire flow
//! (`set_private_key` → `derive_public_key` → `shared_secret`); ALCP
//! 5.1 stores derivation state on the handle, so splitting the
//! lifecycle across multiple handles yields non-matching secrets.

use aocl_crypto_sys as sys;
use aocl_error::{Error, Result};

/// Supported elliptic curves.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Curve {
    /// Curve25519 (Montgomery form, 32-byte keys). Used for X25519 ECDH.
    X25519,
}

impl Curve {
    fn id(self) -> sys::alc_ec_curve_id {
        match self {
            Curve::X25519 => sys::alc_ec_curve_id_ALCP_EC_CURVE25519,
        }
    }

    fn ty(self) -> sys::alc_ec_curve_type {
        match self {
            Curve::X25519 => sys::alc_ec_curve_type_ALCP_EC_CURVE_TYPE_MONTGOMERY,
        }
    }

    /// Length in bytes of private and public keys for this curve.
    pub const fn key_len(self) -> usize {
        match self {
            Curve::X25519 => 32,
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
            message: format!("ALCP EC returned alc_error_t = {err:#x}"),
        })
    }
}

/// An EC session bound to a specific curve. Build with [`EcDh::new`],
/// set the local private key, derive the local public key, share it
/// with the peer, and call [`EcDh::shared_secret`] with the peer's
/// public key to derive the agreed-upon secret.
pub struct EcDh {
    handle: sys::_alc_ec_handle,
    _context: Box<[u8]>,
    _info: Box<sys::alc_ec_info>,
    curve: Curve,
}

impl EcDh {
    /// Build an EC session for the given curve.
    pub fn new(curve: Curve) -> Result<Self> {
        let mut info: Box<sys::alc_ec_info> = Box::new(sys::alc_ec_info {
            ecCurveId: curve.id(),
            ecCurveType: curve.ty(),
            ecPointFormat:
                sys::alc_ec_point_format_id_ALCP_EC_POINT_FORMAT_UNCOMPRESSED,
        });
        let context_size =
            unsafe { sys::alcp_ec_context_size(info.as_mut() as *mut _) } as usize;
        if context_size == 0 {
            return Err(Error::AllocationFailed("crypto"));
        }
        let mut context = vec![0u8; context_size].into_boxed_slice();
        let mut handle = sys::_alc_ec_handle {
            context: context.as_mut_ptr() as *mut std::os::raw::c_void,
        };
        check(unsafe { sys::alcp_ec_request(info.as_mut() as *mut _, &mut handle) })?;
        Ok(Self { handle, _context: context, _info: info, curve })
    }

    /// Set the local private key. Must be exactly `Curve::key_len()` bytes.
    pub fn set_private_key(&mut self, private_key: &[u8]) -> Result<()> {
        if private_key.len() != self.curve.key_len() {
            return Err(Error::InvalidArgument(format!(
                "set_private_key: expected {} bytes, got {}",
                self.curve.key_len(), private_key.len()
            )));
        }
        check(unsafe { sys::alcp_ec_set_privatekey(&mut self.handle, private_key.as_ptr()) })
    }

    /// Derive the local public key from the given private key, writing
    /// `Curve::key_len()` bytes into `public_key`.
    pub fn derive_public_key(
        &mut self,
        private_key: &[u8],
        public_key: &mut [u8],
    ) -> Result<()> {
        let n = self.curve.key_len();
        if private_key.len() != n {
            return Err(Error::InvalidArgument(format!(
                "derive_public_key: private key must be {n} bytes"
            )));
        }
        if public_key.len() < n {
            return Err(Error::InvalidArgument(format!(
                "derive_public_key: public key buffer must be at least {n} bytes"
            )));
        }
        check(unsafe {
            sys::alcp_ec_get_publickey(
                &mut self.handle,
                public_key.as_mut_ptr(),
                private_key.as_ptr(),
            )
        })
    }

    /// Compute the ECDH shared secret given the peer's public key.
    /// Writes up to `Curve::key_len()` bytes into `secret`; the actual
    /// length is returned.
    pub fn shared_secret(
        &mut self,
        peer_public_key: &[u8],
        secret: &mut [u8],
    ) -> Result<usize> {
        let n = self.curve.key_len();
        if peer_public_key.len() != n {
            return Err(Error::InvalidArgument(format!(
                "shared_secret: peer public key must be {n} bytes"
            )));
        }
        if secret.len() < n {
            return Err(Error::InvalidArgument(format!(
                "shared_secret: secret buffer must be at least {n} bytes"
            )));
        }
        let mut key_len: u64 = secret.len() as u64;
        check(unsafe {
            sys::alcp_ec_get_secretkey(
                &mut self.handle,
                secret.as_mut_ptr(),
                peer_public_key.as_ptr(),
                &mut key_len,
            )
        })?;
        Ok(key_len as usize)
    }
}

impl Drop for EcDh {
    fn drop(&mut self) {
        unsafe { sys::alcp_ec_finish(&mut self.handle) };
    }
}

impl std::fmt::Debug for EcDh {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EcDh").field("curve", &self.curve).finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn x25519_dh_round_trip() {
        // Mirrors ALCP's x25519_example.c: each peer keeps a single
        // EcDh handle through set_private_key → derive_public_key →
        // shared_secret. Splitting the lifecycle across multiple
        // handles produces non-matching secrets on AOCL 5.1.
        let alice_priv = [
            0x77, 0x07, 0x6d, 0x0a, 0x73, 0x18, 0xa5, 0x7d,
            0x3c, 0x16, 0xc1, 0x72, 0x51, 0xb2, 0x66, 0x45,
            0xdf, 0x4c, 0x2f, 0x87, 0xeb, 0xc0, 0x99, 0x2a,
            0xb1, 0x77, 0xfb, 0xa5, 0x1d, 0xb9, 0x2c, 0x2a,
        ];
        let bob_priv = [
            0x5d, 0xab, 0x08, 0x7e, 0x62, 0x4a, 0x8a, 0x4b,
            0x79, 0xe1, 0x7f, 0x8b, 0x83, 0x80, 0x0e, 0xe6,
            0x6f, 0x3b, 0xb1, 0x29, 0x26, 0x18, 0xb6, 0xfd,
            0x1c, 0x2f, 0x8b, 0x27, 0xff, 0x88, 0xe0, 0xeb,
        ];

        let mut alice = EcDh::new(Curve::X25519).unwrap();
        let mut bob = EcDh::new(Curve::X25519).unwrap();

        alice.set_private_key(&alice_priv).unwrap();
        bob.set_private_key(&bob_priv).unwrap();

        let mut alice_pub = [0u8; 32];
        let mut bob_pub = [0u8; 32];
        alice.derive_public_key(&alice_priv, &mut alice_pub).unwrap();
        bob.derive_public_key(&bob_priv, &mut bob_pub).unwrap();

        // Same handles compute the shared secret with the peer's pub key.
        let mut alice_secret = [0u8; 32];
        let mut bob_secret = [0u8; 32];
        alice.shared_secret(&bob_pub, &mut alice_secret).unwrap();
        bob.shared_secret(&alice_pub, &mut bob_secret).unwrap();

        assert_eq!(alice_secret, bob_secret,
                   "X25519 shared secrets did not match");
        assert!(alice_secret.iter().any(|&b| b != 0),
                "shared secret was all zeros");
    }
}
