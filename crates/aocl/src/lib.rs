//! `aocl` — safe, idiomatic Rust wrappers for the AMD Optimizing CPU Libraries.
//!
//! Each AOCL component is exposed as a top-level module gated behind its
//! Cargo feature. The crate forwards features to [`aocl-sys`], which provides
//! the raw FFI bindings.
//!
//! See the workspace [README](https://github.com/ciresnave/aocl) for setup
//! instructions (in particular: `AOCL_ROOT`, `LIBCLANG_PATH`).

#![warn(missing_debug_implementations)]
#![cfg_attr(docsrs, feature(doc_cfg))]

pub mod error;

#[cfg(feature = "utils")]
#[cfg_attr(docsrs, doc(cfg(feature = "utils")))]
pub mod utils;

#[cfg(feature = "blis")]
#[cfg_attr(docsrs, doc(cfg(feature = "blis")))]
pub mod blas;

#[cfg(feature = "libflame")]
#[cfg_attr(docsrs, doc(cfg(feature = "libflame")))]
pub mod lapack;

#[cfg(feature = "libm")]
#[cfg_attr(docsrs, doc(cfg(feature = "libm")))]
pub mod math;

#[cfg(feature = "fftw")]
#[cfg_attr(docsrs, doc(cfg(feature = "fftw")))]
pub mod fft;

#[cfg(feature = "sparse")]
#[cfg_attr(docsrs, doc(cfg(feature = "sparse")))]
pub mod sparse;

#[cfg(feature = "rng")]
#[cfg_attr(docsrs, doc(cfg(feature = "rng")))]
pub mod rng;

#[cfg(feature = "securerng")]
#[cfg_attr(docsrs, doc(cfg(feature = "securerng")))]
pub mod securerng;

#[cfg(feature = "compression")]
#[cfg_attr(docsrs, doc(cfg(feature = "compression")))]
pub mod compression;

#[cfg(feature = "crypto")]
#[cfg_attr(docsrs, doc(cfg(feature = "crypto")))]
pub mod crypto;

#[cfg(feature = "data-analytics")]
#[cfg_attr(docsrs, doc(cfg(feature = "data-analytics")))]
pub mod data_analytics;

pub use error::{Error, Result};

/// Re-export of the `aocl-sys` crate for users who need to drop down to the
/// raw FFI for an operation not yet covered by a safe wrapper.
pub use aocl_sys as sys;
