//! Safe Rust wrappers for AOCL-ScaLAPACK.
//!
//! This crate is under active construction. AOCL ships no public C
//! headers for ScaLAPACK; the safe API will be built on top of
//! hand-written Fortran-symbol bindings in [`aocl-scalapack-sys`] using
//! the `Grid` (BLACS process grid) + `BlockCyclic` (distributed-matrix
//! descriptor) abstractions.
//!
//! Calling any ScaLAPACK routine requires an MPI runtime
//! (MS-MPI / MPICH / OpenMPI) installed and the host program launched
//! under `mpiexec`.

#![warn(missing_debug_implementations)]
#![cfg_attr(docsrs, feature(doc_cfg))]

pub use aocl_error::{Error, Result};
pub use aocl_types::{Layout, Trans};
