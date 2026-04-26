//! Shared types used by AOCL safe wrappers.
//!
//! Pull-out crate for the matrix-orientation / triangular-storage enums
//! that recur across BLAS, LAPACK, sparse-BLAS, and data-analytics calls.
//! Each safe `aocl-*` crate accepts these types and converts them to the
//! corresponding native FFI representation at its boundary.

#![warn(missing_debug_implementations, missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

/// Storage order of a 2-D matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Layout {
    /// Rows are stored contiguously (C-style).
    RowMajor,
    /// Columns are stored contiguously (Fortran-style).
    ColMajor,
}

/// How to interpret a matrix operand: as-is, transposed, or
/// conjugate-transposed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Trans {
    /// Use the matrix as stored.
    No,
    /// Use the transpose `Aᵀ`.
    T,
    /// Use the conjugate transpose `Aᴴ`. Equivalent to `T` for real matrices.
    C,
}

/// Triangular fill mode for a matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Uplo {
    /// Upper triangle is referenced; lower is implied.
    Upper,
    /// Lower triangle is referenced; upper is implied.
    Lower,
}

/// Diagonal interpretation for a triangular matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Diag {
    /// Diagonal entries are explicit.
    NonUnit,
    /// Diagonal entries are implicitly `1` (unit-triangular).
    Unit,
}

/// On which side of an operand a matrix is applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Side {
    /// `A` is applied on the left of `B` (`op(A) · B`).
    Left,
    /// `A` is applied on the right of `B` (`B · op(A)`).
    Right,
}

/// Module containing a sealed marker trait for use as a bound on public
/// traits whose set of implementing types should not grow outside this
/// project.
///
/// Each safe `aocl-*` crate's `Scalar` trait extends `Sealed` so users
/// cannot add their own scalar types.
pub mod sealed {
    /// Sealed marker. Implemented for `f32` and `f64`. Additional types
    /// (e.g. complex precisions) will be added as they are wrapped.
    pub trait Sealed {}

    impl Sealed for f32 {}
    impl Sealed for f64 {}
}
