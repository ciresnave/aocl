//! Safe wrappers for AOCL-Cryptography (ALCP).
//!
//! Modules:
//! - [`digest`] — MD5, SHA-1, SHA-2, SHA-3 hash functions
//! - [`cipher`] — AES (ECB / CBC / OFB / CTR / CFB / XTS) and ChaCha20
//!   stream-cipher encryption
//! - [`mac`] — Message Authentication Codes: HMAC, CMAC, Poly1305

#![warn(missing_debug_implementations)]
#![cfg_attr(docsrs, feature(doc_cfg))]

pub mod cipher;
pub mod digest;
pub mod mac;

pub use aocl_error::{Error, Result};
