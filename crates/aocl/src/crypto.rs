//! Safe wrappers for AOCL-Cryptography (ALCP).
//!
//! Currently exposes the digest API: SHA-1, SHA-2 (224/256/384/512), and
//! SHA-3 family hashes. Cipher / AEAD / RSA / EC will follow.
//!
//! # Example
//!
//! ```no_run
//! # #[cfg(feature = "crypto")] {
//! use aocl::crypto::digest::{Digest, Mode};
//!
//! let mut d = Digest::new(Mode::Sha2_256).unwrap();
//! d.update(b"hello, ").unwrap();
//! d.update(b"world").unwrap();
//! let hash = d.finalize().unwrap();
//! assert_eq!(hash.len(), 32);
//! # }
//! ```

pub mod digest;
