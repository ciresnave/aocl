//! Raw FFI bindings to AOCL-LAPACK (libFLAME).
//!
//! Exposes the LAPACKE C interface and standard LAPACK Fortran-symbol
//! declarations. For a safe, idiomatic API see [`aocl-lapack`](https://docs.rs/aocl-lapack).

#![allow(
    non_upper_case_globals,
    non_camel_case_types,
    non_snake_case,
    dead_code,
    improper_ctypes,
    clippy::all
)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
