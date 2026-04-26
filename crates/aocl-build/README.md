# aocl-build

Build-script helpers shared across the `aocl-*-sys` crates. **Build-time only** — never appears in a downstream user's runtime dependency tree.

Provides:

- [`locate_aocl_root`] — finds the AOCL install directory via the `AOCL_ROOT` environment variable, then OS-default install paths (`C:\Program Files\AMD\AOCL-Windows`, `/opt/AMD/aocl/aocl-linux-*`).
- [`ensure_libclang_path`] — auto-detects an LLVM install on Windows so `bindgen` doesn't pick up a Swift-bundled (or other unsuitable) `libclang.dll` from `PATH`.
- [`Component`] — declarative description of one AOCL component (header location, lib names per OS / linkage, ILP64 vs. LP64 layout). Call [`Component::build`] from a `*-sys` crate's `build.rs`.

This crate is dual-licensed under MIT or Apache-2.0.
