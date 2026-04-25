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

| Module          | Feature           | Status            |
|-----------------|-------------------|-------------------|
| `utils::cpuid`  | `utils`           | Initial scaffold  |
| `blas`          | `blis`            | Pending           |
| `lapack`        | `libflame`        | Pending           |
| `fft`           | `fftw`            | Pending           |
| `math`          | `libm`            | Pending           |
| `sparse`        | `sparse`          | Pending           |
| `rng`           | `rng`             | Pending           |
| `securerng`     | `securerng`       | Pending           |
| `compression`   | `compression`     | Pending           |
| `crypto`        | `crypto`          | Pending           |
| `data_analytics`| `data-analytics`  | Pending           |
| `scalapack`     | `scalapack`       | Pending           |

## License

Dual MIT / Apache-2.0; see workspace [LICENSE-MIT](../../LICENSE-MIT) and [LICENSE-APACHE](../../LICENSE-APACHE).
