# aocl-types

Shared types and enums used by AOCL safe wrappers — `Layout`, `Trans`, `Uplo`, `Diag`, `Side`, plus a [`sealed`] module for sealing public traits.

These appear at the safe-API boundary of multiple AOCL components (a `Layout` is meaningful for BLAS, LAPACK, sparse-BLAS, and data-analytics calls). Pulled out into its own crate so:

- Cross-library code can use one canonical `Layout` value across calls into different AOCL components without converting between per-crate definitions.
- Each safe `aocl-*` crate has a small, fixed dependency core rather than re-defining the same enums.

This crate intentionally has **no `*-sys` dependencies**. Conversion between an `aocl_types` enum and the native FFI representation happens inside each safe crate, at its boundary with its dedicated `*-sys` crate.

Dual-licensed under MIT or Apache-2.0.
