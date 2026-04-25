//! `aocl-sys` — raw FFI bindings to AMD AOCL.
//!
//! Each AOCL component is gated behind its own Cargo feature and exposed as a
//! top-level module containing the `bindgen`-generated declarations. All items
//! re-exported here are `unsafe extern "C"`; refer to the AMD AOCL API guide
//! for semantics.
//!
//! For a safe, idiomatic Rust API on top of these bindings see the
//! [`aocl`](https://docs.rs/aocl) crate.

#![allow(
    non_upper_case_globals,
    non_camel_case_types,
    non_snake_case,
    dead_code,
    improper_ctypes,
    clippy::all
)]
#![cfg_attr(docsrs, feature(doc_cfg))]

#[cfg(feature = "blis")]
#[cfg_attr(docsrs, doc(cfg(feature = "blis")))]
pub mod blis {
    //! AOCL-BLAS (BLIS) — Basic Linear Algebra Subprograms.
    include!(concat!(env!("OUT_DIR"), "/blis.rs"));
}

#[cfg(feature = "libflame")]
#[cfg_attr(docsrs, doc(cfg(feature = "libflame")))]
pub mod libflame {
    //! AOCL-LAPACK (libFLAME) — Linear Algebra PACKage.
    include!(concat!(env!("OUT_DIR"), "/libflame.rs"));
}

#[cfg(feature = "utils")]
#[cfg_attr(docsrs, doc(cfg(feature = "utils")))]
pub mod utils {
    //! AOCL-Utils — CPU identification, threading helpers.
    include!(concat!(env!("OUT_DIR"), "/utils.rs"));
}

#[cfg(feature = "libm")]
#[cfg_attr(docsrs, doc(cfg(feature = "libm")))]
pub mod libm {
    //! AOCL-LibM — vectorized math functions.
    include!(concat!(env!("OUT_DIR"), "/libm.rs"));
}

#[cfg(feature = "fftw")]
#[cfg_attr(docsrs, doc(cfg(feature = "fftw")))]
pub mod fftw {
    //! AOCL-FFTW — FFTW-compatible Fast Fourier Transform.
    include!(concat!(env!("OUT_DIR"), "/fftw.rs"));
}

#[cfg(feature = "sparse")]
#[cfg_attr(docsrs, doc(cfg(feature = "sparse")))]
pub mod sparse {
    //! AOCL-Sparse — sparse BLAS and solvers.
    include!(concat!(env!("OUT_DIR"), "/sparse.rs"));
}

#[cfg(feature = "rng")]
#[cfg_attr(docsrs, doc(cfg(feature = "rng")))]
pub mod rng {
    //! AOCL-RNG — random number generation.
    include!(concat!(env!("OUT_DIR"), "/rng.rs"));
}

#[cfg(feature = "securerng")]
#[cfg_attr(docsrs, doc(cfg(feature = "securerng")))]
pub mod securerng {
    //! AOCL-SecureRNG — hardware-backed random number generation.
    include!(concat!(env!("OUT_DIR"), "/securerng.rs"));
}

#[cfg(feature = "compression")]
#[cfg_attr(docsrs, doc(cfg(feature = "compression")))]
pub mod compression {
    //! AOCL-Compression — multi-algorithm compression library.
    include!(concat!(env!("OUT_DIR"), "/compression.rs"));
}

#[cfg(feature = "crypto")]
#[cfg_attr(docsrs, doc(cfg(feature = "crypto")))]
pub mod crypto {
    //! AOCL-Cryptography (ALCP) — cryptographic primitives.
    include!(concat!(env!("OUT_DIR"), "/crypto.rs"));
}

#[cfg(feature = "data-analytics")]
#[cfg_attr(docsrs, doc(cfg(feature = "data-analytics")))]
pub mod data_analytics {
    //! AOCL-DA — data analytics and ML primitives.
    include!(concat!(env!("OUT_DIR"), "/data_analytics.rs"));
}

// `scalapack` is link-only: no public C bindings.
