# aocl

Umbrella crate re-exporting the safe Rust wrappers for the [AMD Optimizing CPU Libraries](https://www.amd.com/en/developer/aocl.html). Convenient when you want most of AOCL under a single dependency line. If you only need one or two components, depend directly on the per-component crates (e.g. `aocl-blas`, `aocl-lapack`).

```rust
#[cfg(feature = "blas")]
use aocl::blas;

#[cfg(feature = "lapack")]
use aocl::lapack;
```

## Features

One per AOCL component (`blas`, `lapack`, `fft`, `math`, `sparse`, `rng`, `securerng`, `utils`, `compression`, `crypto`, `data-analytics`, `scalapack`), plus build modes (`ilp64`, `static-link`).

Shared types (`Layout`, `Trans`, `Uplo`, `Diag`, `Side`) and the common `Error` type are re-exported from this crate's root regardless of which features you enable.

Dual-licensed under MIT or Apache-2.0.
