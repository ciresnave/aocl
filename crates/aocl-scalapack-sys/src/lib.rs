//! Link shim for AOCL-ScaLAPACK.
//!
//! ScaLAPACK is a Fortran + MPI library and AOCL ships no public C
//! headers for it. This crate currently emits link directives only;
//! hand-written `extern "C"` declarations of the BLACS lifecycle and
//! ScaLAPACK Fortran routines (`pdgesv_`, `pdgemm_`, …) will be added
//! here as their safe-side wrappers in [`aocl-scalapack`] are built out.
//!
//! The MPI runtime is not declared as a Cargo dependency — it must be
//! present on the target system (MS-MPI / MPICH / OpenMPI on Linux).
