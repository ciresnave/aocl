# aocl roadmap

Taken over by the CPUFish lane 2026-10-30. Statuses: TODO, DOING, DONE, BLOCKED. Facts measured at `origin/main` 7fbe68f
(workspace 0.1.2) and crates.io on 2026-10-30.

Constraint: Fuel depends on `aocl-blas = "0.1"` and `aocl-types = "0.1"` from crates.io. Nothing here may break that API
without coordinating with the Fuel lane first (pre-1.0: a breaking change is 0.1.x to 0.2.0).

| # | Row | Status | Notes |
|---|-----|--------|-------|
| 1 | Release gap: crates.io has 0.1.0 of every crate, `main` is 0.1.2 | TODO | 0.1.1 (CI, CODEOWNERS) and 0.1.2 (lint) unpublished; PM publishes |
| 2 | CHANGELOG.md, `deny.toml` + cargo-deny job (repo has neither) | DONE | portfolio CI rule |
| 3 | Tests on real AOCL: CI only builds `*-sys` under a DOCS_RS stub; the safe wrappers are tested "by the contributor locally". Run them on the self-hosted runner (Windows, AOCL 5.3.2 installed on the box) | TODO | rule 5.1g/5.1h, no `#[ignore]` |
| 4 | Dependency currency: workspace pins `thiserror = "1"` (latest 2.0.21), `bindgen = "0.69"` (latest 0.73.2); MSRV 1.81 | DONE 0.1.5 (CI stub build only; bindgen 0.73.2 against the real AOCL 5.3.2 headers verified for `aocl-blas-sys` only, 2026-10-31, VS LLVM libclang) | `libc` latest on crates.io is only 1.0.0-alpha.5; stay on stable 0.2 and note it |
| 5 | Verify bindings against installed AOCL 5.3.x (pre-generated bindings vendored for docs.rs came from some earlier version: confirm which) | DONE for `aocl-blas-sys` (0.1.6): the vendored file does NOT match 5.3.2; its origin version is UNKNOWN; other `*-sys` crates not compared | Measured 2026-10-31 on `origin/main` 5d361cb: bindgen 0.73.2 against `%ProgramFiles%/AMD/AOCL-Windows/amd-blis/include/LP64` (AOCL 5.3.2) builds, and its output differs from `crates/aocl-blas-sys/bindings/bindings.rs` (bindgen 0.69.5): 265 names only in the vendored file, 568 only in the regenerated one, 30 shared constants with different values (the `arch_t_BLIS_ARCH_*` enum, e.g. `ZEN5` is 8 in the vendored file). Vendored bindings are used only for `DOCS_RS=1` builds, so the shipped link path is unaffected; docs.rs shows an older API. Nothing in the vendored file names its AOCL version. Next: regenerate the vendored files from a recorded AOCL version (a separate PR; changes docs.rs output only); licence: bindings derive from AMD/BLIS headers, keep notices |
| 6 | Linux coverage: runtime test of the wrappers on Linux AOCL | TODO | needs a Linux self-hosted runner or box |
| 7 | `aocl-scalapack`: still "under construction"; decide keep, finish, or drop | TODO | needs MPI runtime |
| 8 | Fill API gaps CPUFish providers need (gemm batched, LAPACK/FFT/LibM subsets) | BLOCKED | on CPUFish design (cpufish docs/SURVEY.md) |
