# aocl-scalapack

Safe Rust wrappers for AOCL-ScaLAPACK — distributed linear algebra over MPI.

**Under construction.** ScaLAPACK ships only Fortran sources and Fortran-style symbols (no public C headers). Hand-written `extern "C"` declarations for the BLACS lifecycle (`Cblacs_*`) and ScaLAPACK Fortran routines (`pdgesv_`, `pdgemm_`, …), plus a safe `Grid` / `BlockCyclic`-descriptor wrapper, are queued as the next milestone after the workspace restructure stabilizes.

Built on top of [`aocl-scalapack-sys`](../aocl-scalapack-sys/).

Dual-licensed under MIT or Apache-2.0.
