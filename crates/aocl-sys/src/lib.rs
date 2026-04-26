//! `aocl-sys` — umbrella re-exporting the per-component `aocl-*-sys` FFI
//! crates under cargo features.
//!
//! For most users the per-component crates are easier to depend on
//! directly. This crate exists for "I want a bunch of AOCL components
//! behind one dependency line" use-cases.

#![cfg_attr(docsrs, feature(doc_cfg))]

#[cfg(feature = "blas")]
#[cfg_attr(docsrs, doc(cfg(feature = "blas")))]
pub use aocl_blas_sys as blas;

#[cfg(feature = "lapack")]
#[cfg_attr(docsrs, doc(cfg(feature = "lapack")))]
pub use aocl_lapack_sys as lapack;

#[cfg(feature = "fft")]
#[cfg_attr(docsrs, doc(cfg(feature = "fft")))]
pub use aocl_fft_sys as fft;

#[cfg(feature = "math")]
#[cfg_attr(docsrs, doc(cfg(feature = "math")))]
pub use aocl_math_sys as math;

#[cfg(feature = "sparse")]
#[cfg_attr(docsrs, doc(cfg(feature = "sparse")))]
pub use aocl_sparse_sys as sparse;

#[cfg(feature = "rng")]
#[cfg_attr(docsrs, doc(cfg(feature = "rng")))]
pub use aocl_rng_sys as rng;

#[cfg(feature = "securerng")]
#[cfg_attr(docsrs, doc(cfg(feature = "securerng")))]
pub use aocl_securerng_sys as securerng;

#[cfg(feature = "utils")]
#[cfg_attr(docsrs, doc(cfg(feature = "utils")))]
pub use aocl_utils_sys as utils;

#[cfg(feature = "compression")]
#[cfg_attr(docsrs, doc(cfg(feature = "compression")))]
pub use aocl_compression_sys as compression;

#[cfg(feature = "crypto")]
#[cfg_attr(docsrs, doc(cfg(feature = "crypto")))]
pub use aocl_crypto_sys as crypto;

#[cfg(feature = "data-analytics")]
#[cfg_attr(docsrs, doc(cfg(feature = "data-analytics")))]
pub use aocl_data_analytics_sys as data_analytics;

#[cfg(feature = "scalapack")]
#[cfg_attr(docsrs, doc(cfg(feature = "scalapack")))]
pub use aocl_scalapack_sys as scalapack;
