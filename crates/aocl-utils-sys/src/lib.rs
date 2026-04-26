//! Raw FFI bindings to AOCL-Utils.
//!
//! For a safe, idiomatic API see [`aocl-utils`](https://docs.rs/aocl-utils).

#![allow(
    non_upper_case_globals,
    non_camel_case_types,
    non_snake_case,
    dead_code,
    improper_ctypes,
    clippy::all
)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
