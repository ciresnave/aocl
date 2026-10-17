//! Raw FFI bindings to AOCL-RNG.
//!
//! For a safe, idiomatic API see [`aocl-rng`](https://docs.rs/aocl-rng).

#![allow(
    non_upper_case_globals,
    non_camel_case_types,
    non_snake_case,
    dead_code,
    improper_ctypes,
    clippy::all,
    reason = "bindgen output mirrors the C headers: names, unused items and C-only types are not ours to change"
)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
