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

/// Single-precision complex number, ABI-compatible with the C struct
/// `{ float real; float imag; }` used by BLIS's `scomplex`, LAPACK's
/// `lapack_complex_float`, and FFTW's `fftwf_complex`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complex32 {
    /// Real part.
    pub re: f32,
    /// Imaginary part.
    pub im: f32,
}

impl Complex32 {
    /// Create a complex number `re + im·i`.
    #[inline]
    pub const fn new(re: f32, im: f32) -> Self {
        Self { re, im }
    }

    /// `0 + 0i`.
    pub const ZERO: Self = Self::new(0.0, 0.0);
    /// `1 + 0i`.
    pub const ONE: Self = Self::new(1.0, 0.0);
    /// `0 + 1i`.
    pub const I: Self = Self::new(0.0, 1.0);

    /// Modulus squared `re² + im²` (avoids the square root of `abs`).
    #[inline]
    pub fn norm_sqr(&self) -> f32 {
        self.re * self.re + self.im * self.im
    }

    /// Modulus `√(re² + im²)`.
    #[inline]
    pub fn abs(&self) -> f32 {
        self.norm_sqr().sqrt()
    }

    /// Complex conjugate.
    #[inline]
    pub fn conj(&self) -> Self {
        Self::new(self.re, -self.im)
    }
}

impl From<(f32, f32)> for Complex32 {
    #[inline]
    fn from((re, im): (f32, f32)) -> Self {
        Self::new(re, im)
    }
}

impl From<[f32; 2]> for Complex32 {
    #[inline]
    fn from(a: [f32; 2]) -> Self {
        Self::new(a[0], a[1])
    }
}

impl From<Complex32> for [f32; 2] {
    #[inline]
    fn from(c: Complex32) -> Self {
        [c.re, c.im]
    }
}

impl From<f32> for Complex32 {
    #[inline]
    fn from(re: f32) -> Self {
        Self::new(re, 0.0)
    }
}

/// Double-precision complex number, ABI-compatible with the C struct
/// `{ double real; double imag; }` used by BLIS's `dcomplex`, LAPACK's
/// `lapack_complex_double`, and FFTW's `fftw_complex`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complex64 {
    /// Real part.
    pub re: f64,
    /// Imaginary part.
    pub im: f64,
}

impl Complex64 {
    /// Create a complex number `re + im·i`.
    #[inline]
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    /// `0 + 0i`.
    pub const ZERO: Self = Self::new(0.0, 0.0);
    /// `1 + 0i`.
    pub const ONE: Self = Self::new(1.0, 0.0);
    /// `0 + 1i`.
    pub const I: Self = Self::new(0.0, 1.0);

    /// Modulus squared `re² + im²`.
    #[inline]
    pub fn norm_sqr(&self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    /// Modulus `√(re² + im²)`.
    #[inline]
    pub fn abs(&self) -> f64 {
        self.norm_sqr().sqrt()
    }

    /// Complex conjugate.
    #[inline]
    pub fn conj(&self) -> Self {
        Self::new(self.re, -self.im)
    }
}

impl From<(f64, f64)> for Complex64 {
    #[inline]
    fn from((re, im): (f64, f64)) -> Self {
        Self::new(re, im)
    }
}

impl From<[f64; 2]> for Complex64 {
    #[inline]
    fn from(a: [f64; 2]) -> Self {
        Self::new(a[0], a[1])
    }
}

impl From<Complex64> for [f64; 2] {
    #[inline]
    fn from(c: Complex64) -> Self {
        [c.re, c.im]
    }
}

impl From<f64> for Complex64 {
    #[inline]
    fn from(re: f64) -> Self {
        Self::new(re, 0.0)
    }
}

/// Module containing a sealed marker trait for use as a bound on public
/// traits whose set of implementing types should not grow outside this
/// project.
///
/// Each safe `aocl-*` crate's `Scalar` trait extends `Sealed` so users
/// cannot add their own scalar types.
pub mod sealed {
    use super::{Complex32, Complex64};

    /// Sealed marker. Implemented for `f32`, `f64`, and the complex
    /// precisions `Complex32` / `Complex64`.
    pub trait Sealed {}

    impl Sealed for f32 {}
    impl Sealed for f64 {}
    impl Sealed for Complex32 {}
    impl Sealed for Complex64 {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complex32_layout_matches_array() {
        // Repr-C struct of two f32s must be exactly 8 bytes laid out
        // [re, im], matching [f32; 2] for FFI cast safety.
        assert_eq!(std::mem::size_of::<Complex32>(), 8);
        assert_eq!(
            std::mem::align_of::<Complex32>(),
            std::mem::align_of::<f32>()
        );
        let c = Complex32::new(1.5, -2.0);
        let arr: [f32; 2] = c.into();
        assert_eq!(arr, [1.5, -2.0]);
    }

    #[test]
    fn complex64_layout_matches_array() {
        assert_eq!(std::mem::size_of::<Complex64>(), 16);
        assert_eq!(
            std::mem::align_of::<Complex64>(),
            std::mem::align_of::<f64>()
        );
        let c = Complex64::new(3.0, 4.0);
        assert_eq!(c.abs(), 5.0);
        assert_eq!(c.norm_sqr(), 25.0);
        assert_eq!(c.conj(), Complex64::new(3.0, -4.0));
    }

    #[test]
    fn complex_constants() {
        assert_eq!(Complex32::ZERO, Complex32::new(0.0, 0.0));
        assert_eq!(Complex32::ONE, Complex32::new(1.0, 0.0));
        assert_eq!(Complex32::I, Complex32::new(0.0, 1.0));
        assert_eq!(Complex64::ZERO, Complex64::new(0.0, 0.0));
    }
}
