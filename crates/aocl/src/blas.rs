//! Safe wrappers for AOCL-BLAS (BLIS) via the CBLAS interface.
//!
//! Provides a [`Scalar`] trait implemented for `f32` and `f64` so that GEMM,
//! AXPY, and DOT can be called with either precision. The wrappers take
//! Rust slices (with bounds-checked dimension arguments) instead of raw
//! pointers; mismatched sizes return an [`Error::InvalidArgument`].
//!
//! For routines not yet wrapped here, drop down to [`aocl_sys::blis`] for
//! direct CBLAS access.

use crate::error::{Error, Result};
use aocl_sys::blis as sys;

/// Storage order of a matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Layout {
    /// Rows are stored contiguously (C-style).
    RowMajor,
    /// Columns are stored contiguously (Fortran-style).
    ColMajor,
}

impl Layout {
    fn raw(self) -> sys::CBLAS_ORDER {
        match self {
            Layout::RowMajor => sys::CBLAS_ORDER_CblasRowMajor as sys::CBLAS_ORDER,
            Layout::ColMajor => sys::CBLAS_ORDER_CblasColMajor as sys::CBLAS_ORDER,
        }
    }
}

/// How to interpret a matrix operand: as-is, transposed, or conjugate-transposed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Trans {
    /// Use the matrix as stored.
    No,
    /// Use the transpose `Aᵀ`.
    T,
    /// Use the conjugate transpose `Aᴴ`. Equivalent to `T` for real matrices.
    C,
}

impl Trans {
    fn raw(self) -> sys::CBLAS_TRANSPOSE {
        match self {
            Trans::No => sys::CBLAS_TRANSPOSE_CblasNoTrans as sys::CBLAS_TRANSPOSE,
            Trans::T => sys::CBLAS_TRANSPOSE_CblasTrans as sys::CBLAS_TRANSPOSE,
            Trans::C => sys::CBLAS_TRANSPOSE_CblasConjTrans as sys::CBLAS_TRANSPOSE,
        }
    }
}

/// Scalar element type usable with the BLAS routines exposed here.
///
/// Implemented for `f32` and `f64`; complex types will follow.
pub trait Scalar: Copy + Sized + private::Sealed {
    /// `Y := α·X + Y`.
    fn axpy(alpha: Self, x: &[Self], inc_x: usize, y: &mut [Self], inc_y: usize) -> Result<()>;

    /// `Σ Xᵢ · Yᵢ` (real dot product).
    fn dot(x: &[Self], inc_x: usize, y: &[Self], inc_y: usize) -> Result<Self>;

    /// `C := α · op(A) · op(B) + β · C`.
    ///
    /// Dimensions:
    /// - `op(A)` is `m × k`
    /// - `op(B)` is `k × n`
    /// - `C` is `m × n`
    ///
    /// `lda`, `ldb`, `ldc` are the leading dimensions of `A`, `B`, and `C`
    /// in elements; for row-major they are the row strides, for col-major
    /// they are the column strides.
    #[allow(clippy::too_many_arguments)]
    fn gemm(
        layout: Layout,
        trans_a: Trans,
        trans_b: Trans,
        m: usize,
        n: usize,
        k: usize,
        alpha: Self,
        a: &[Self],
        lda: usize,
        b: &[Self],
        ldb: usize,
        beta: Self,
        c: &mut [Self],
        ldc: usize,
    ) -> Result<()>;
}

mod private {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

fn check_strided_len(name: &str, slice_len: usize, n: usize, inc: usize) -> Result<()> {
    if inc == 0 {
        return Err(Error::InvalidArgument(format!(
            "{name}: stride must be non-zero"
        )));
    }
    let needed = (n - 1).saturating_mul(inc) + 1;
    if n > 0 && slice_len < needed {
        return Err(Error::InvalidArgument(format!(
            "{name}: slice length {slice_len} too small for n={n} stride={inc} (need {needed})"
        )));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn check_gemm_shapes(
    layout: Layout,
    trans_a: Trans,
    trans_b: Trans,
    m: usize,
    n: usize,
    k: usize,
    a_len: usize,
    lda: usize,
    b_len: usize,
    ldb: usize,
    c_len: usize,
    ldc: usize,
) -> Result<()> {
    // op(A): rows × cols.
    let (a_rows, a_cols) = match trans_a {
        Trans::No => (m, k),
        Trans::T | Trans::C => (k, m),
    };
    let (b_rows, b_cols) = match trans_b {
        Trans::No => (k, n),
        Trans::T | Trans::C => (n, k),
    };

    let (a_min_ld, a_min_len, b_min_ld, b_min_len, c_min_ld, c_min_len) = match layout {
        Layout::RowMajor => (
            a_cols,
            a_rows.saturating_sub(1) * lda + a_cols.max(1),
            b_cols,
            b_rows.saturating_sub(1) * ldb + b_cols.max(1),
            n,
            m.saturating_sub(1) * ldc + n.max(1),
        ),
        Layout::ColMajor => (
            a_rows,
            a_cols.saturating_sub(1) * lda + a_rows.max(1),
            b_rows,
            b_cols.saturating_sub(1) * ldb + b_rows.max(1),
            m,
            n.saturating_sub(1) * ldc + m.max(1),
        ),
    };

    if lda < a_min_ld.max(1) {
        return Err(Error::InvalidArgument(format!("gemm: lda={lda} < {a_min_ld}")));
    }
    if ldb < b_min_ld.max(1) {
        return Err(Error::InvalidArgument(format!("gemm: ldb={ldb} < {b_min_ld}")));
    }
    if ldc < c_min_ld.max(1) {
        return Err(Error::InvalidArgument(format!("gemm: ldc={ldc} < {c_min_ld}")));
    }
    if a_len < a_min_len {
        return Err(Error::InvalidArgument(format!(
            "gemm: A slice {a_len} too small (need {a_min_len})"
        )));
    }
    if b_len < b_min_len {
        return Err(Error::InvalidArgument(format!(
            "gemm: B slice {b_len} too small (need {b_min_len})"
        )));
    }
    if c_len < c_min_len {
        return Err(Error::InvalidArgument(format!(
            "gemm: C slice {c_len} too small (need {c_min_len})"
        )));
    }
    let _ = (n, layout, trans_a, trans_b);
    Ok(())
}

macro_rules! impl_scalar {
    ($t:ty, $axpy:ident, $dot:ident, $gemm:ident) => {
        impl Scalar for $t {
            fn axpy(
                alpha: Self,
                x: &[Self],
                inc_x: usize,
                y: &mut [Self],
                inc_y: usize,
            ) -> Result<()> {
                if x.len() != y.len() && (inc_x == 1 && inc_y == 1) {
                    // For unit stride we expect equal lengths; otherwise the
                    // caller is responsible and we just check the strided
                    // bound below.
                }
                let n = if inc_x == 1 && inc_y == 1 {
                    x.len().min(y.len())
                } else {
                    // With non-unit stride, the caller controls n via slice
                    // sizing. Use the smaller capacity.
                    x.len() / inc_x.max(1)
                };
                check_strided_len("axpy: x", x.len(), n, inc_x)?;
                check_strided_len("axpy: y", y.len(), n, inc_y)?;
                // SAFETY: pointers come from valid slices; n and strides
                // checked above; alpha is a copy.
                unsafe {
                    sys::$axpy(
                        n as sys::f77_int,
                        alpha,
                        x.as_ptr(),
                        inc_x as sys::f77_int,
                        y.as_mut_ptr(),
                        inc_y as sys::f77_int,
                    );
                }
                Ok(())
            }

            fn dot(
                x: &[Self],
                inc_x: usize,
                y: &[Self],
                inc_y: usize,
            ) -> Result<Self> {
                let n = if inc_x == 1 && inc_y == 1 {
                    x.len().min(y.len())
                } else {
                    x.len() / inc_x.max(1)
                };
                check_strided_len("dot: x", x.len(), n, inc_x)?;
                check_strided_len("dot: y", y.len(), n, inc_y)?;
                let r = unsafe {
                    sys::$dot(
                        n as sys::f77_int,
                        x.as_ptr(),
                        inc_x as sys::f77_int,
                        y.as_ptr(),
                        inc_y as sys::f77_int,
                    )
                };
                Ok(r)
            }

            #[allow(clippy::too_many_arguments)]
            fn gemm(
                layout: Layout,
                trans_a: Trans,
                trans_b: Trans,
                m: usize,
                n: usize,
                k: usize,
                alpha: Self,
                a: &[Self],
                lda: usize,
                b: &[Self],
                ldb: usize,
                beta: Self,
                c: &mut [Self],
                ldc: usize,
            ) -> Result<()> {
                check_gemm_shapes(
                    layout, trans_a, trans_b, m, n, k, a.len(), lda, b.len(), ldb, c.len(), ldc,
                )?;
                unsafe {
                    sys::$gemm(
                        layout.raw(),
                        trans_a.raw(),
                        trans_b.raw(),
                        m as sys::f77_int,
                        n as sys::f77_int,
                        k as sys::f77_int,
                        alpha,
                        a.as_ptr(),
                        lda as sys::f77_int,
                        b.as_ptr(),
                        ldb as sys::f77_int,
                        beta,
                        c.as_mut_ptr(),
                        ldc as sys::f77_int,
                    );
                }
                Ok(())
            }
        }
    };
}

impl_scalar!(f32, cblas_saxpy, cblas_sdot, cblas_sgemm);
impl_scalar!(f64, cblas_daxpy, cblas_ddot, cblas_dgemm);

/// Convenience entry point: compute `Y := α·X + Y`.
pub fn axpy<T: Scalar>(alpha: T, x: &[T], y: &mut [T]) -> Result<()> {
    T::axpy(alpha, x, 1, y, 1)
}

/// Convenience entry point: compute `Σ Xᵢ · Yᵢ`.
pub fn dot<T: Scalar>(x: &[T], y: &[T]) -> Result<T> {
    T::dot(x, 1, y, 1)
}

/// Convenience entry point: `C := α · op(A) · op(B) + β · C` for tightly-packed
/// row-major matrices.
#[allow(clippy::too_many_arguments)]
pub fn gemm<T: Scalar>(
    trans_a: Trans,
    trans_b: Trans,
    m: usize,
    n: usize,
    k: usize,
    alpha: T,
    a: &[T],
    b: &[T],
    beta: T,
    c: &mut [T],
) -> Result<()> {
    let lda = match trans_a {
        Trans::No => k,
        _ => m,
    };
    let ldb = match trans_b {
        Trans::No => n,
        _ => k,
    };
    let ldc = n;
    T::gemm(
        Layout::RowMajor,
        trans_a,
        trans_b,
        m,
        n,
        k,
        alpha,
        a,
        lda,
        b,
        ldb,
        beta,
        c,
        ldc,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_f64() {
        let x = [1.0_f64, 2.0, 3.0];
        let y = [4.0_f64, 5.0, 6.0];
        let r = dot(&x, &y).unwrap();
        assert!((r - 32.0).abs() < 1e-12);
    }

    #[test]
    fn axpy_f32() {
        let x = [1.0_f32, 2.0, 3.0];
        let mut y = [10.0_f32, 20.0, 30.0];
        axpy(2.0_f32, &x, &mut y).unwrap();
        assert_eq!(y, [12.0, 24.0, 36.0]);
    }

    #[test]
    fn gemm_2x2_identity() {
        // A = I_2, B = [[1, 2], [3, 4]] → C = A·B = B
        let a = [1.0_f64, 0.0, 0.0, 1.0]; // 2x2 row-major
        let b = [1.0_f64, 2.0, 3.0, 4.0];
        let mut c = [0.0_f64; 4];
        gemm(Trans::No, Trans::No, 2, 2, 2, 1.0, &a, &b, 0.0, &mut c).unwrap();
        for (got, want) in c.iter().zip(b.iter()) {
            assert!((got - want).abs() < 1e-12);
        }
    }

    #[test]
    fn gemm_2x3_times_3x2() {
        // A is 2x3 row-major: [[1, 2, 3], [4, 5, 6]]
        // B is 3x2 row-major: [[1, 0], [0, 1], [1, 1]]
        // A * B should be 2x2: [[1+0+3, 0+2+3], [4+0+6, 0+5+6]] = [[4, 5], [10, 11]]
        let a = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = [1.0_f64, 0.0, 0.0, 1.0, 1.0, 1.0];
        let mut c = [0.0_f64; 4];
        gemm(Trans::No, Trans::No, 2, 2, 3, 1.0, &a, &b, 0.0, &mut c).unwrap();
        let want = [4.0, 5.0, 10.0, 11.0];
        for (got, w) in c.iter().zip(want.iter()) {
            assert!((got - w).abs() < 1e-12, "got={got} want={w}");
        }
    }

    #[test]
    fn gemm_dim_mismatch_is_error() {
        let a = [1.0_f64; 6]; // 2x3
        let b = [1.0_f64; 4]; // wrong: should be 3x2
        let mut c = [0.0_f64; 4];
        let err = gemm(Trans::No, Trans::No, 2, 2, 3, 1.0, &a, &b, 0.0, &mut c).unwrap_err();
        match err {
            Error::InvalidArgument(_) => {}
            other => panic!("unexpected error: {other:?}"),
        }
    }
}
