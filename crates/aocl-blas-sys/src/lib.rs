//! Raw FFI bindings to AOCL-BLAS (BLIS).
//!
//! This crate exposes the CBLAS C interface plus BLIS's native API, generated
//! by `bindgen` against the AOCL-installed headers. For a safe, idiomatic
//! API see the [`aocl-blas`](https://docs.rs/aocl-blas) crate.

#![allow(
    non_upper_case_globals,
    non_camel_case_types,
    non_snake_case,
    dead_code,
    improper_ctypes,
    clippy::all
)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
