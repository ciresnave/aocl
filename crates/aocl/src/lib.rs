//! `aocl` — umbrella re-exporting the per-component safe AOCL wrappers
//! under cargo features.
//!
//! Most users will prefer to depend directly on the component crate they
//! need (e.g. `aocl-blas`, `aocl-lapack`). This crate exists for "I want
//! a bunch of AOCL components behind one dependency line" use-cases.

#![cfg_attr(docsrs, feature(doc_cfg))]

pub use aocl_error::{Error, Result};
pub use aocl_types::{Diag, Layout, Side, Trans, Uplo};

#[cfg(feature = "blas")]
#[cfg_attr(docsrs, doc(cfg(feature = "blas")))]
pub use aocl_blas as blas;

#[cfg(feature = "lapack")]
#[cfg_attr(docsrs, doc(cfg(feature = "lapack")))]
pub use aocl_lapack as lapack;

#[cfg(feature = "fft")]
#[cfg_attr(docsrs, doc(cfg(feature = "fft")))]
pub use aocl_fft as fft;

#[cfg(feature = "math")]
#[cfg_attr(docsrs, doc(cfg(feature = "math")))]
pub use aocl_math as math;

#[cfg(feature = "sparse")]
#[cfg_attr(docsrs, doc(cfg(feature = "sparse")))]
pub use aocl_sparse as sparse;

#[cfg(feature = "rng")]
#[cfg_attr(docsrs, doc(cfg(feature = "rng")))]
pub use aocl_rng as rng;

#[cfg(feature = "securerng")]
#[cfg_attr(docsrs, doc(cfg(feature = "securerng")))]
pub use aocl_securerng as securerng;

#[cfg(feature = "utils")]
#[cfg_attr(docsrs, doc(cfg(feature = "utils")))]
pub use aocl_utils as utils;

#[cfg(feature = "compression")]
#[cfg_attr(docsrs, doc(cfg(feature = "compression")))]
pub use aocl_compression as compression;

#[cfg(feature = "crypto")]
#[cfg_attr(docsrs, doc(cfg(feature = "crypto")))]
pub use aocl_crypto as crypto;

#[cfg(feature = "data-analytics")]
#[cfg_attr(docsrs, doc(cfg(feature = "data-analytics")))]
pub use aocl_data_analytics as data_analytics;

#[cfg(feature = "scalapack")]
#[cfg_attr(docsrs, doc(cfg(feature = "scalapack")))]
pub use aocl_scalapack as scalapack;
