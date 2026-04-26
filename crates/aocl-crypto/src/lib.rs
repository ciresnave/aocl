//! Safe wrappers for AOCL-Cryptography (ALCP).
//!
//! Currently exposes the digest API. Cipher / AEAD / MAC / RSA / EC will
//! follow.
//!
//! # Example
//!
//! ```no_run
//! use aocl_crypto::digest::{Digest, Mode};
//!
//! let mut d = Digest::new(Mode::Sha2_256).unwrap();
//! d.update(b"hello, ").unwrap();
//! d.update(b"world").unwrap();
//! let hash = d.finalize().unwrap();
//! assert_eq!(hash.len(), 32);
//! ```

#![warn(missing_debug_implementations)]
#![cfg_attr(docsrs, feature(doc_cfg))]

pub mod digest;

pub use aocl_error::{Error, Result};
