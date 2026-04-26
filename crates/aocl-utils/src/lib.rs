//! Safe wrappers for AOCL-Utils — AMD CPU identification, CPU-feature
//! flag queries, thread pinning, and library version reporting.

#![warn(missing_debug_implementations)]
#![cfg_attr(docsrs, feature(doc_cfg))]

pub mod cpuid;
pub mod threads;
pub mod version;

pub use aocl_error::{Error, Result};
