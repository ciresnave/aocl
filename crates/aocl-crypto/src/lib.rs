//! Safe wrappers for AOCL-Cryptography (ALCP).
//!
//! Modules:
//! - [`digest`] — MD5, SHA-1, SHA-2, SHA-3 hash functions
//! - [`cipher`] — AES (ECB / CBC / OFB / CTR / CFB / XTS) and ChaCha20
//!   stream-cipher encryption
//! - [`mac`] — Message Authentication Codes: HMAC, CMAC, Poly1305
//! - [`aead`] — AES-GCM, AES-CCM, AES-SIV, ChaCha20-Poly1305 authenticated
//!   encryption with associated data

#![warn(missing_debug_implementations)]
#![cfg_attr(docsrs, feature(doc_cfg))]

pub mod aead;
pub mod cipher;
pub mod digest;
pub mod drbg;
pub mod ec;
pub mod mac;
pub mod rsa;

pub use aocl_error::{Error, Result};
