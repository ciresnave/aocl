//! Safe wrappers for AOCL-Utils — AMD CPU identification and threading
//! helpers.

#![warn(missing_debug_implementations)]
#![cfg_attr(docsrs, feature(doc_cfg))]

pub mod cpuid;

pub use aocl_error::{Error, Result};
