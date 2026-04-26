# aocl

Rust bindings for the [AMD Optimizing CPU Libraries (AOCL)](https://www.amd.com/en/developer/aocl.html), split into per-component crates so downstream users can take only what they need.

## Layout

The workspace contains 29 crates organized as three layers:

### Shared

- [`aocl-build`](crates/aocl-build/) — build-script helpers (AOCL root detection, libclang detection, bindgen+linker driver). Build-time only; never appears in a downstream user's runtime tree.
- [`aocl-error`](crates/aocl-error/) — shared `Error` and `Result` types used by every safe wrapper.
- [`aocl-types`](crates/aocl-types/) — shared `Layout` / `Trans` / `Uplo` / `Diag` / `Side` enums and the `Sealed` marker trait.

### Per-component (12 pairs)

| Component         | Sys crate (raw FFI)                                          | Safe crate                                                |
|-------------------|--------------------------------------------------------------|-----------------------------------------------------------|
| BLAS              | [`aocl-blas-sys`](crates/aocl-blas-sys/)                     | [`aocl-blas`](crates/aocl-blas/)                          |
| LAPACK            | [`aocl-lapack-sys`](crates/aocl-lapack-sys/)                 | [`aocl-lapack`](crates/aocl-lapack/)                      |
| FFT               | [`aocl-fft-sys`](crates/aocl-fft-sys/)                       | [`aocl-fft`](crates/aocl-fft/)                            |
| LibM              | [`aocl-math-sys`](crates/aocl-math-sys/)                     | [`aocl-math`](crates/aocl-math/)                          |
| Sparse            | [`aocl-sparse-sys`](crates/aocl-sparse-sys/)                 | [`aocl-sparse`](crates/aocl-sparse/)                      |
| RNG               | [`aocl-rng-sys`](crates/aocl-rng-sys/)                       | [`aocl-rng`](crates/aocl-rng/)                            |
| SecureRNG         | [`aocl-securerng-sys`](crates/aocl-securerng-sys/)           | [`aocl-securerng`](crates/aocl-securerng/)                |
| Utils             | [`aocl-utils-sys`](crates/aocl-utils-sys/)                   | [`aocl-utils`](crates/aocl-utils/)                        |
| Compression       | [`aocl-compression-sys`](crates/aocl-compression-sys/)       | [`aocl-compression`](crates/aocl-compression/)            |
| Cryptography      | [`aocl-crypto-sys`](crates/aocl-crypto-sys/)                 | [`aocl-crypto`](crates/aocl-crypto/)                      |
| Data Analytics    | [`aocl-data-analytics-sys`](crates/aocl-data-analytics-sys/) | [`aocl-data-analytics`](crates/aocl-data-analytics/)      |
| ScaLAPACK         | [`aocl-scalapack-sys`](crates/aocl-scalapack-sys/)           | [`aocl-scalapack`](crates/aocl-scalapack/)                |

### Umbrellas (convenience)

- [`aocl-sys`](crates/aocl-sys/) — re-exports each `aocl-*-sys` under a cargo feature.
- [`aocl`](crates/aocl/) — re-exports each safe crate under a cargo feature.

If you only need one or two AOCL components, depend directly on those crates. The umbrellas exist so you can write `aocl = { version = "0.1", features = ["blas", "lapack", "fft"] }` instead of three separate dependency lines.

## Cargo features (mirrored on `aocl-sys` and `aocl`)

Component features: `blas`, `lapack`, `fft`, `math`, `sparse`, `rng`, `securerng`, `utils`, `compression`, `crypto`, `data-analytics`, `scalapack`. Default: `utils` only.

Build-mode features: `ilp64` (64-bit integer indexing — default is LP64 / 32-bit), `static-link` (default is dynamic).

## Prerequisites

1. **Install AOCL** from <https://www.amd.com/en/developer/aocl.html>.
   - Windows default: `C:\Program Files\AMD\AOCL-Windows`
   - Linux default:   `/opt/AMD/aocl/aocl-linux-*`
2. **Set `AOCL_ROOT`** to your install root if it is not in the default location.
3. **Install `libclang`** (required by `bindgen`):
   - Windows: install LLVM (`winget install LLVM.LLVM`) — `aocl-build` will auto-detect it at `C:\Program Files\LLVM\bin`. Set `LIBCLANG_PATH` explicitly if it lives elsewhere.
   - Linux:   `apt install libclang-dev` (or distro equivalent).

## Build

```sh
# Just the BLAS bindings, dynamically linked, LP64.
cargo build -p aocl-blas

# Everything via the umbrella, ILP64, static linking.
cargo build -p aocl --features "blas lapack fft math sparse rng securerng utils compression crypto data-analytics ilp64 static-link"
```

## Runtime

When linking dynamically (the default), the AOCL DLLs / shared objects must be reachable at runtime:

- **Windows**: add the relevant `<AOCL_ROOT>\<component>\lib[\LP64|\ILP64]` directory to `PATH` before running. Some components live under `lib\<arch>\shared\` (sparse, fft, compression, data-analytics). For example:

  ```bat
  set PATH=C:\Program Files\AMD\AOCL-Windows\amd-utils\lib;%PATH%
  set PATH=C:\Program Files\AMD\AOCL-Windows\amd-blis\lib\LP64;%PATH%
  ```

  Many of AMD's Windows AOCL DLLs are built with the Intel compiler and depend on `libmmd.dll` and `svml_dispmd.dll` (Intel runtime). If you have Intel oneAPI installed, add `C:\Program Files (x86)\Intel\oneAPI\<version>\bin` to `PATH` as well. Some components (`compression`) also pull in LLVM's OpenMP runtime (`libomp.dll`) — add `C:\Program Files\LLVM\bin` to `PATH` if you have LLVM installed. The `crypto` ALCP DLL also depends on OpenSSL 3 (`libcrypto-3-x64.dll`).

  When mixing components from different AOCL DLLs that link different OpenMP runtimes (Intel `libiomp5md.dll` vs LLVM `libomp.dll`), set `KMP_DUPLICATE_LIB_OK=TRUE` to suppress the runtime's duplicate-library abort. AMD's libraries do not actually conflict in practice.

- **Linux**: add to `LD_LIBRARY_PATH`, or rely on the system loader cache.

For builds with `--features static-link` no runtime configuration is required.

## Licensing

This project is dual-licensed under either of:

- [MIT license](LICENSE-MIT)
- [Apache License, Version 2.0](LICENSE-APACHE)

at your option.

The AOCL libraries themselves are distributed by AMD under their own terms; this project does not redistribute AMD code or binaries. Make sure your use of the underlying AOCL libraries complies with AMD's license.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
