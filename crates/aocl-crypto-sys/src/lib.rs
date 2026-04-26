//! Raw FFI bindings to AOCL-Cryptography (ALCP).
//!
//! For a safe, idiomatic API see [`aocl-crypto`](https://docs.rs/aocl-crypto).

#![allow(
    non_upper_case_globals,
    non_camel_case_types,
    non_snake_case,
    dead_code,
    improper_ctypes,
    clippy::all
)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
