# aocl

Safe, idiomatic Rust wrappers for the [AMD Optimizing CPU Libraries (AOCL)](https://www.amd.com/en/developer/aocl.html), built on top of [`aocl-sys`](../aocl-sys/).

## Features

One Cargo feature per AOCL component, plus build-mode toggles. See the workspace [README](../../README.md) for the full list.

## Example

```rust
# #[cfg(feature = "utils")] {
use aocl::utils::cpuid::{Cpu, is_amd, vendor_info};

let info = vendor_info(Cpu::Current);
println!("CPU: {:?}", info);
println!("Is AMD: {}", is_amd(Cpu::Current));
# }
```

## Status

Early development. Module coverage:

- `utils::cpuid` (feature `utils`) — `is_amd`, `is_zen_family`, `zen_arch`, `x86_64_level`, `vendor_info`.
- `blas` (feature `blis`) — `Scalar` trait for `f32`/`f64`; `gemm`, `axpy`, `dot` with bounds-checked dimensions.
- `lapack` (feature `libflame`) — `gesv`, `getrf` for `f32`/`f64`; typed `Error::Status` for LAPACK info codes.
- `math` (feature `libm`) — 14 scalar wrappers (`exp/ln/pow/sin/cos/tan/sincos/sqrt/cbrt/hypot`) with `f32` + `f64` variants.
- `sparse` (feature `sparse`) — `MatDescr` RAII handle, `csrmv` (CSR mat-vec) for `f32`/`f64`.
- `fft` (feature `fftw`) — `dft_1d_inplace` 1-D complex DFT (forward / backward), planner mutex-serialized for thread safety.
- `rng` (feature `rng`) — `Rng` with `BaseGenerator` (LCG, MT, …); `uniform`, `gaussian`, `exponential` samplers.
- `data_analytics` (feature `data-analytics`) — `mean`, `variance` along row / column / global axes for `f32`/`f64`.
- `securerng`, `compression`, `crypto` — bindings only (`aocl_sys::*`); safe layer pending.
- `scalapack` — link-only; AOCL ships no public C headers (Fortran/MPI).

## License

Dual MIT / Apache-2.0; see workspace [LICENSE-MIT](../../LICENSE-MIT) and [LICENSE-APACHE](../../LICENSE-APACHE).
