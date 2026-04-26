# aocl-sys

Umbrella crate re-exporting the individual `aocl-*-sys` FFI binding crates under cargo features. Convenient when you want most or all of AOCL behind a single dependency line. If you only need one or two components, depend directly on the corresponding `aocl-*-sys` crate.

## Cargo features

One per AOCL component (`blas`, `lapack`, `fft`, `math`, `sparse`, `rng`, `securerng`, `utils`, `compression`, `crypto`, `data-analytics`, `scalapack`), plus build modes (`ilp64`, `static-link`).

## Re-exports

When a feature is enabled, the corresponding crate is re-exported under a short name:

```rust
#[cfg(feature = "blas")]
use aocl_sys::blas;
```

For the safe wrappers, see the [`aocl`](../aocl/) umbrella.

Dual-licensed under MIT or Apache-2.0.
