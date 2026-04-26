//! Safe wrappers for AOCL-BLAS (BLIS) via the CBLAS interface.
//!
//! Comprehensive Level-1 (vector–vector) coverage across `f32`, `f64`,
//! `Complex32`, and `Complex64`, plus GEMM from Level 3. Level 2 and the
//! rest of Level 3 are next.
//!
//! The [`Scalar`] trait gathers operations defined for all four
//! precisions; the [`RealScalar`] extension trait gathers operations
//! defined only for the real precisions (`rotg`, `rot`, `dsdot`).
//!
//! All wrappers take Rust slices with bounds-checked dimensions; mismatched
//! sizes return [`Error::InvalidArgument`]. For routines not yet wrapped,
//! drop down to [`aocl-blas-sys`].

#![warn(missing_debug_implementations)]
#![cfg_attr(docsrs, feature(doc_cfg))]

use aocl_blas_sys as sys;
pub use aocl_error::{Error, Result};
pub use aocl_types::{Complex32, Complex64, Layout, Trans};
use aocl_types::sealed::Sealed;

// ===========================================================================
//   Enum → CBLAS conversion helpers
// ===========================================================================

fn layout_raw(l: Layout) -> sys::CBLAS_ORDER {
    match l {
        Layout::RowMajor => sys::CBLAS_ORDER_CblasRowMajor as sys::CBLAS_ORDER,
        Layout::ColMajor => sys::CBLAS_ORDER_CblasColMajor as sys::CBLAS_ORDER,
    }
}

fn trans_raw(t: Trans) -> sys::CBLAS_TRANSPOSE {
    match t {
        Trans::No => sys::CBLAS_TRANSPOSE_CblasNoTrans as sys::CBLAS_TRANSPOSE,
        Trans::T => sys::CBLAS_TRANSPOSE_CblasTrans as sys::CBLAS_TRANSPOSE,
        Trans::C => sys::CBLAS_TRANSPOSE_CblasConjTrans as sys::CBLAS_TRANSPOSE,
    }
}

// ===========================================================================
//   Slice / dimension validation
// ===========================================================================

fn check_strided_len(name: &str, slice_len: usize, n: usize, inc: usize) -> Result<()> {
    if inc == 0 {
        return Err(Error::InvalidArgument(format!(
            "{name}: stride must be non-zero"
        )));
    }
    if n == 0 {
        return Ok(());
    }
    let needed = (n - 1).saturating_mul(inc) + 1;
    if slice_len < needed {
        return Err(Error::InvalidArgument(format!(
            "{name}: slice length {slice_len} too small for n={n} stride={inc} (need {needed})"
        )));
    }
    Ok(())
}

/// Determine `n` from two slices and their strides, picking the smaller
/// number of complete elements available in either.
fn infer_n(x_len: usize, inc_x: usize, y_len: usize, inc_y: usize) -> usize {
    if inc_x == 1 && inc_y == 1 {
        x_len.min(y_len)
    } else {
        let nx = if inc_x == 0 { 0 } else { x_len.div_ceil(inc_x.max(1)) };
        let ny = if inc_y == 0 { 0 } else { y_len.div_ceil(inc_y.max(1)) };
        nx.min(ny)
    }
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
    Ok(())
}

// ===========================================================================
//   Scalar / RealScalar traits
// ===========================================================================

/// Element type usable with the wrapped BLAS routines.
///
/// Implemented for `f32`, `f64`, `Complex32`, and `Complex64`.
pub trait Scalar: Copy + Sized + Sealed {
    /// Underlying real precision. `f32` for `f32` and `Complex32`; `f64`
    /// for `f64` and `Complex64`. Used by `nrm2` / `asum` results.
    type Real: Copy + Sized + Sealed;

    // --- Level-1 ----------------------------------------------------------

    /// `X := α · X`.
    fn scal(alpha: Self, x: &mut [Self], inc: usize) -> Result<()>;

    /// `X := α · X` where the scalar is a real value (useful for complex
    /// `X` to multiply by a real factor without going through `Self`).
    /// For real `Self` this is the same as [`Self::scal`].
    fn scal_real(alpha: Self::Real, x: &mut [Self], inc: usize) -> Result<()>;

    /// `Y := X` (vector copy).
    fn copy(x: &[Self], inc_x: usize, y: &mut [Self], inc_y: usize) -> Result<()>;

    /// Swap two vectors element-wise.
    fn swap(x: &mut [Self], inc_x: usize, y: &mut [Self], inc_y: usize) -> Result<()>;

    /// `Y := α · X + Y`.
    fn axpy(alpha: Self, x: &[Self], inc_x: usize, y: &mut [Self], inc_y: usize) -> Result<()>;

    /// "Natural" inner product: `Σ Xᵢ · Yᵢ` for real types,
    /// `Σ conj(Xᵢ) · Yᵢ` for complex types (the mathematical standard
    /// `⟨x, y⟩`). Use [`Self::dotu`] for the unconjugated complex variant.
    fn dot(x: &[Self], inc_x: usize, y: &[Self], inc_y: usize) -> Result<Self>;

    /// Unconjugated dot product `Σ Xᵢ · Yᵢ`. Identical to [`Self::dot`]
    /// for real types; differs for complex (no conjugate on `X`).
    fn dotu(x: &[Self], inc_x: usize, y: &[Self], inc_y: usize) -> Result<Self>;

    /// `||X||₂ = √(Σ |Xᵢ|²)`. Returns the underlying real precision.
    fn nrm2(x: &[Self], inc: usize) -> Result<Self::Real>;

    /// `Σ |Xᵢ|`. For complex types, AOCL/BLAS define this as
    /// `Σ |Re(Xᵢ)| + |Im(Xᵢ)|` (the "1-norm-ish" L1 sum).
    fn asum(x: &[Self], inc: usize) -> Result<Self::Real>;

    /// 0-based index of the element with the largest absolute value
    /// (`|Re(Xᵢ)| + |Im(Xᵢ)|` for complex). Returns `Err` for empty input.
    fn iamax(x: &[Self], inc: usize) -> Result<usize>;

    /// 0-based index of the element with the smallest absolute value.
    fn iamin(x: &[Self], inc: usize) -> Result<usize>;

    // --- Level-3 (GEMM only for now) -------------------------------------

    /// `C := α · op(A) · op(B) + β · C`.
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

/// Operations defined only for real BLAS precisions (`f32`, `f64`).
pub trait RealScalar: Scalar<Real = Self> {
    /// Compute the parameters `c, s` of a Givens rotation that zeroes out
    /// `b`. The values of `a` and `b` are modified per the BLAS standard.
    fn rotg(a: &mut Self, b: &mut Self, c: &mut Self, s: &mut Self);

    /// Apply a Givens rotation to two vectors:
    /// `[Xᵢ, Yᵢ] := [c·Xᵢ + s·Yᵢ, -s·Xᵢ + c·Yᵢ]`.
    fn rot(x: &mut [Self], inc_x: usize, y: &mut [Self], inc_y: usize, c: Self, s: Self) -> Result<()>;

    /// Mixed-precision dot for `f32` operands accumulating in `f64`. For
    /// `f64` `Self` this returns the same value as [`Scalar::dot`].
    fn dsdot(x: &[Self], inc_x: usize, y: &[Self], inc_y: usize) -> Result<f64>;
}

// ===========================================================================
//   impl Scalar for f32 and f64
// ===========================================================================

macro_rules! impl_real_scalar {
    (
        $t:ty,
        scal = $scal:ident,
        copy = $copy:ident,
        swap = $swap:ident,
        axpy = $axpy:ident,
        dot  = $dot:ident,
        nrm2 = $nrm2:ident,
        asum = $asum:ident,
        iamax = $iamax:ident,
        iamin = $iamin:ident,
        gemm = $gemm:ident,
        rotg = $rotg:ident,
        rot  = $rot:ident,
        dsdot_fn = $dsdot_fn:expr
    ) => {
        impl Scalar for $t {
            type Real = $t;

            fn scal(alpha: Self, x: &mut [Self], inc: usize) -> Result<()> {
                let n = if inc == 0 { 0 } else { x.len() / inc.max(1) };
                check_strided_len("scal: x", x.len(), n, inc)?;
                unsafe {
                    sys::$scal(n as sys::f77_int, alpha, x.as_mut_ptr(), inc as sys::f77_int);
                }
                Ok(())
            }

            fn scal_real(alpha: Self::Real, x: &mut [Self], inc: usize) -> Result<()> {
                Self::scal(alpha, x, inc)
            }

            fn copy(x: &[Self], inc_x: usize, y: &mut [Self], inc_y: usize) -> Result<()> {
                let n = infer_n(x.len(), inc_x, y.len(), inc_y);
                check_strided_len("copy: x", x.len(), n, inc_x)?;
                check_strided_len("copy: y", y.len(), n, inc_y)?;
                unsafe {
                    sys::$copy(n as sys::f77_int, x.as_ptr(), inc_x as sys::f77_int,
                               y.as_mut_ptr(), inc_y as sys::f77_int);
                }
                Ok(())
            }

            fn swap(x: &mut [Self], inc_x: usize, y: &mut [Self], inc_y: usize) -> Result<()> {
                let n = infer_n(x.len(), inc_x, y.len(), inc_y);
                check_strided_len("swap: x", x.len(), n, inc_x)?;
                check_strided_len("swap: y", y.len(), n, inc_y)?;
                unsafe {
                    sys::$swap(n as sys::f77_int, x.as_mut_ptr(), inc_x as sys::f77_int,
                               y.as_mut_ptr(), inc_y as sys::f77_int);
                }
                Ok(())
            }

            fn axpy(alpha: Self, x: &[Self], inc_x: usize, y: &mut [Self], inc_y: usize) -> Result<()> {
                let n = infer_n(x.len(), inc_x, y.len(), inc_y);
                check_strided_len("axpy: x", x.len(), n, inc_x)?;
                check_strided_len("axpy: y", y.len(), n, inc_y)?;
                unsafe {
                    sys::$axpy(n as sys::f77_int, alpha, x.as_ptr(), inc_x as sys::f77_int,
                               y.as_mut_ptr(), inc_y as sys::f77_int);
                }
                Ok(())
            }

            fn dot(x: &[Self], inc_x: usize, y: &[Self], inc_y: usize) -> Result<Self> {
                let n = infer_n(x.len(), inc_x, y.len(), inc_y);
                check_strided_len("dot: x", x.len(), n, inc_x)?;
                check_strided_len("dot: y", y.len(), n, inc_y)?;
                let r = unsafe {
                    sys::$dot(n as sys::f77_int, x.as_ptr(), inc_x as sys::f77_int,
                              y.as_ptr(), inc_y as sys::f77_int)
                };
                Ok(r)
            }

            fn dotu(x: &[Self], inc_x: usize, y: &[Self], inc_y: usize) -> Result<Self> {
                Self::dot(x, inc_x, y, inc_y)
            }

            fn nrm2(x: &[Self], inc: usize) -> Result<Self::Real> {
                let n = if inc == 0 { 0 } else { x.len() / inc.max(1) };
                check_strided_len("nrm2: x", x.len(), n, inc)?;
                let r = unsafe { sys::$nrm2(n as sys::f77_int, x.as_ptr(), inc as sys::f77_int) };
                Ok(r)
            }

            fn asum(x: &[Self], inc: usize) -> Result<Self::Real> {
                let n = if inc == 0 { 0 } else { x.len() / inc.max(1) };
                check_strided_len("asum: x", x.len(), n, inc)?;
                let r = unsafe { sys::$asum(n as sys::f77_int, x.as_ptr(), inc as sys::f77_int) };
                Ok(r)
            }

            fn iamax(x: &[Self], inc: usize) -> Result<usize> {
                let n = if inc == 0 { 0 } else { x.len() / inc.max(1) };
                if n == 0 {
                    return Err(Error::InvalidArgument("iamax: empty input".into()));
                }
                check_strided_len("iamax: x", x.len(), n, inc)?;
                let i = unsafe { sys::$iamax(n as sys::f77_int, x.as_ptr(), inc as sys::f77_int) };
                Ok(i as usize)
            }

            fn iamin(x: &[Self], inc: usize) -> Result<usize> {
                let n = if inc == 0 { 0 } else { x.len() / inc.max(1) };
                if n == 0 {
                    return Err(Error::InvalidArgument("iamin: empty input".into()));
                }
                check_strided_len("iamin: x", x.len(), n, inc)?;
                let i = unsafe { sys::$iamin(n as sys::f77_int, x.as_ptr(), inc as sys::f77_int) };
                Ok(i as usize)
            }

            #[allow(clippy::too_many_arguments)]
            fn gemm(
                layout: Layout, trans_a: Trans, trans_b: Trans,
                m: usize, n: usize, k: usize,
                alpha: Self, a: &[Self], lda: usize,
                b: &[Self], ldb: usize,
                beta: Self, c: &mut [Self], ldc: usize,
            ) -> Result<()> {
                check_gemm_shapes(layout, trans_a, trans_b, m, n, k,
                    a.len(), lda, b.len(), ldb, c.len(), ldc)?;
                unsafe {
                    sys::$gemm(
                        layout_raw(layout), trans_raw(trans_a), trans_raw(trans_b),
                        m as sys::f77_int, n as sys::f77_int, k as sys::f77_int,
                        alpha, a.as_ptr(), lda as sys::f77_int,
                        b.as_ptr(), ldb as sys::f77_int,
                        beta, c.as_mut_ptr(), ldc as sys::f77_int,
                    );
                }
                Ok(())
            }
        }

        impl RealScalar for $t {
            fn rotg(a: &mut Self, b: &mut Self, c: &mut Self, s: &mut Self) {
                unsafe { sys::$rotg(a, b, c, s); }
            }

            fn rot(x: &mut [Self], inc_x: usize, y: &mut [Self], inc_y: usize, c: Self, s: Self) -> Result<()> {
                let n = infer_n(x.len(), inc_x, y.len(), inc_y);
                check_strided_len("rot: x", x.len(), n, inc_x)?;
                check_strided_len("rot: y", y.len(), n, inc_y)?;
                unsafe {
                    sys::$rot(n as sys::f77_int,
                              x.as_mut_ptr(), inc_x as sys::f77_int,
                              y.as_mut_ptr(), inc_y as sys::f77_int,
                              c, s);
                }
                Ok(())
            }

            fn dsdot(x: &[Self], inc_x: usize, y: &[Self], inc_y: usize) -> Result<f64> {
                let n = infer_n(x.len(), inc_x, y.len(), inc_y);
                check_strided_len("dsdot: x", x.len(), n, inc_x)?;
                check_strided_len("dsdot: y", y.len(), n, inc_y)?;
                let r: f64 = ($dsdot_fn)(n as sys::f77_int,
                                         x.as_ptr(), inc_x as sys::f77_int,
                                         y.as_ptr(), inc_y as sys::f77_int);
                Ok(r)
            }
        }
    };
}

impl_real_scalar!(
    f32,
    scal = cblas_sscal, copy = cblas_scopy, swap = cblas_sswap, axpy = cblas_saxpy,
    dot = cblas_sdot, nrm2 = cblas_snrm2, asum = cblas_sasum,
    iamax = cblas_isamax, iamin = cblas_isamin, gemm = cblas_sgemm,
    rotg = cblas_srotg, rot = cblas_srot,
    dsdot_fn = (|n, x, ix, y, iy| unsafe { sys::cblas_dsdot(n, x, ix, y, iy) })
);
impl_real_scalar!(
    f64,
    scal = cblas_dscal, copy = cblas_dcopy, swap = cblas_dswap, axpy = cblas_daxpy,
    dot = cblas_ddot, nrm2 = cblas_dnrm2, asum = cblas_dasum,
    iamax = cblas_idamax, iamin = cblas_idamin, gemm = cblas_dgemm,
    rotg = cblas_drotg, rot = cblas_drot,
    // For f64 dsdot semantically equals dot on f64 inputs; we forward to ddot.
    dsdot_fn = (|n, x, ix, y, iy| unsafe { sys::cblas_ddot(n, x, ix, y, iy) })
);

// ===========================================================================
//   impl Scalar for Complex32 and Complex64
// ===========================================================================

macro_rules! impl_complex_scalar {
    (
        $t:ty, $real:ty,
        scal = $scal:ident, scal_real = $scal_real:ident,
        copy = $copy:ident, swap = $swap:ident,
        axpy = $axpy:ident,
        dotu = $dotu:ident, dotc = $dotc:ident,
        nrm2 = $nrm2:ident, asum = $asum:ident,
        iamax = $iamax:ident, iamin = $iamin:ident,
        gemm = $gemm:ident
    ) => {
        impl Scalar for $t {
            type Real = $real;

            fn scal(alpha: Self, x: &mut [Self], inc: usize) -> Result<()> {
                let n = if inc == 0 { 0 } else { x.len() / inc.max(1) };
                check_strided_len("scal: x", x.len(), n, inc)?;
                unsafe {
                    sys::$scal(
                        n as sys::f77_int,
                        &alpha as *const Self as *const std::os::raw::c_void,
                        x.as_mut_ptr() as *mut std::os::raw::c_void,
                        inc as sys::f77_int,
                    );
                }
                Ok(())
            }

            fn scal_real(alpha: Self::Real, x: &mut [Self], inc: usize) -> Result<()> {
                let n = if inc == 0 { 0 } else { x.len() / inc.max(1) };
                check_strided_len("scal_real: x", x.len(), n, inc)?;
                unsafe {
                    sys::$scal_real(
                        n as sys::f77_int,
                        alpha,
                        x.as_mut_ptr() as *mut std::os::raw::c_void,
                        inc as sys::f77_int,
                    );
                }
                Ok(())
            }

            fn copy(x: &[Self], inc_x: usize, y: &mut [Self], inc_y: usize) -> Result<()> {
                let n = infer_n(x.len(), inc_x, y.len(), inc_y);
                check_strided_len("copy: x", x.len(), n, inc_x)?;
                check_strided_len("copy: y", y.len(), n, inc_y)?;
                unsafe {
                    sys::$copy(
                        n as sys::f77_int,
                        x.as_ptr() as *const std::os::raw::c_void,
                        inc_x as sys::f77_int,
                        y.as_mut_ptr() as *mut std::os::raw::c_void,
                        inc_y as sys::f77_int,
                    );
                }
                Ok(())
            }

            fn swap(x: &mut [Self], inc_x: usize, y: &mut [Self], inc_y: usize) -> Result<()> {
                let n = infer_n(x.len(), inc_x, y.len(), inc_y);
                check_strided_len("swap: x", x.len(), n, inc_x)?;
                check_strided_len("swap: y", y.len(), n, inc_y)?;
                unsafe {
                    sys::$swap(
                        n as sys::f77_int,
                        x.as_mut_ptr() as *mut std::os::raw::c_void,
                        inc_x as sys::f77_int,
                        y.as_mut_ptr() as *mut std::os::raw::c_void,
                        inc_y as sys::f77_int,
                    );
                }
                Ok(())
            }

            fn axpy(alpha: Self, x: &[Self], inc_x: usize, y: &mut [Self], inc_y: usize) -> Result<()> {
                let n = infer_n(x.len(), inc_x, y.len(), inc_y);
                check_strided_len("axpy: x", x.len(), n, inc_x)?;
                check_strided_len("axpy: y", y.len(), n, inc_y)?;
                unsafe {
                    sys::$axpy(
                        n as sys::f77_int,
                        &alpha as *const Self as *const std::os::raw::c_void,
                        x.as_ptr() as *const std::os::raw::c_void,
                        inc_x as sys::f77_int,
                        y.as_mut_ptr() as *mut std::os::raw::c_void,
                        inc_y as sys::f77_int,
                    );
                }
                Ok(())
            }

            fn dot(x: &[Self], inc_x: usize, y: &[Self], inc_y: usize) -> Result<Self> {
                // Mathematical convention: ⟨x, y⟩ = Σ conj(xᵢ)·yᵢ — call dotc.
                let n = infer_n(x.len(), inc_x, y.len(), inc_y);
                check_strided_len("dot: x", x.len(), n, inc_x)?;
                check_strided_len("dot: y", y.len(), n, inc_y)?;
                let mut out = <$t>::ZERO;
                unsafe {
                    sys::$dotc(
                        n as sys::f77_int,
                        x.as_ptr() as *const std::os::raw::c_void,
                        inc_x as sys::f77_int,
                        y.as_ptr() as *const std::os::raw::c_void,
                        inc_y as sys::f77_int,
                        &mut out as *mut Self as *mut std::os::raw::c_void,
                    );
                }
                Ok(out)
            }

            fn dotu(x: &[Self], inc_x: usize, y: &[Self], inc_y: usize) -> Result<Self> {
                let n = infer_n(x.len(), inc_x, y.len(), inc_y);
                check_strided_len("dotu: x", x.len(), n, inc_x)?;
                check_strided_len("dotu: y", y.len(), n, inc_y)?;
                let mut out = <$t>::ZERO;
                unsafe {
                    sys::$dotu(
                        n as sys::f77_int,
                        x.as_ptr() as *const std::os::raw::c_void,
                        inc_x as sys::f77_int,
                        y.as_ptr() as *const std::os::raw::c_void,
                        inc_y as sys::f77_int,
                        &mut out as *mut Self as *mut std::os::raw::c_void,
                    );
                }
                Ok(out)
            }

            fn nrm2(x: &[Self], inc: usize) -> Result<Self::Real> {
                let n = if inc == 0 { 0 } else { x.len() / inc.max(1) };
                check_strided_len("nrm2: x", x.len(), n, inc)?;
                let r = unsafe {
                    sys::$nrm2(
                        n as sys::f77_int,
                        x.as_ptr() as *const std::os::raw::c_void,
                        inc as sys::f77_int,
                    )
                };
                Ok(r)
            }

            fn asum(x: &[Self], inc: usize) -> Result<Self::Real> {
                let n = if inc == 0 { 0 } else { x.len() / inc.max(1) };
                check_strided_len("asum: x", x.len(), n, inc)?;
                let r = unsafe {
                    sys::$asum(
                        n as sys::f77_int,
                        x.as_ptr() as *const std::os::raw::c_void,
                        inc as sys::f77_int,
                    )
                };
                Ok(r)
            }

            fn iamax(x: &[Self], inc: usize) -> Result<usize> {
                let n = if inc == 0 { 0 } else { x.len() / inc.max(1) };
                if n == 0 {
                    return Err(Error::InvalidArgument("iamax: empty input".into()));
                }
                check_strided_len("iamax: x", x.len(), n, inc)?;
                let i = unsafe {
                    sys::$iamax(
                        n as sys::f77_int,
                        x.as_ptr() as *const std::os::raw::c_void,
                        inc as sys::f77_int,
                    )
                };
                Ok(i as usize)
            }

            fn iamin(x: &[Self], inc: usize) -> Result<usize> {
                let n = if inc == 0 { 0 } else { x.len() / inc.max(1) };
                if n == 0 {
                    return Err(Error::InvalidArgument("iamin: empty input".into()));
                }
                check_strided_len("iamin: x", x.len(), n, inc)?;
                let i = unsafe {
                    sys::$iamin(
                        n as sys::f77_int,
                        x.as_ptr() as *const std::os::raw::c_void,
                        inc as sys::f77_int,
                    )
                };
                Ok(i as usize)
            }

            #[allow(clippy::too_many_arguments)]
            fn gemm(
                layout: Layout, trans_a: Trans, trans_b: Trans,
                m: usize, n: usize, k: usize,
                alpha: Self, a: &[Self], lda: usize,
                b: &[Self], ldb: usize,
                beta: Self, c: &mut [Self], ldc: usize,
            ) -> Result<()> {
                check_gemm_shapes(layout, trans_a, trans_b, m, n, k,
                    a.len(), lda, b.len(), ldb, c.len(), ldc)?;
                unsafe {
                    sys::$gemm(
                        layout_raw(layout), trans_raw(trans_a), trans_raw(trans_b),
                        m as sys::f77_int, n as sys::f77_int, k as sys::f77_int,
                        &alpha as *const Self as *const std::os::raw::c_void,
                        a.as_ptr() as *const std::os::raw::c_void, lda as sys::f77_int,
                        b.as_ptr() as *const std::os::raw::c_void, ldb as sys::f77_int,
                        &beta as *const Self as *const std::os::raw::c_void,
                        c.as_mut_ptr() as *mut std::os::raw::c_void, ldc as sys::f77_int,
                    );
                }
                Ok(())
            }
        }
    };
}

impl_complex_scalar!(
    Complex32, f32,
    scal = cblas_cscal, scal_real = cblas_csscal,
    copy = cblas_ccopy, swap = cblas_cswap, axpy = cblas_caxpy,
    dotu = cblas_cdotu_sub, dotc = cblas_cdotc_sub,
    nrm2 = cblas_scnrm2, asum = cblas_scasum,
    iamax = cblas_icamax, iamin = cblas_icamin,
    gemm = cblas_cgemm
);
impl_complex_scalar!(
    Complex64, f64,
    scal = cblas_zscal, scal_real = cblas_zdscal,
    copy = cblas_zcopy, swap = cblas_zswap, axpy = cblas_zaxpy,
    dotu = cblas_zdotu_sub, dotc = cblas_zdotc_sub,
    nrm2 = cblas_dznrm2, asum = cblas_dzasum,
    iamax = cblas_izamax, iamin = cblas_izamin,
    gemm = cblas_zgemm
);

// ===========================================================================
//   Free-function convenience entry points (unit stride; matching sizes)
// ===========================================================================

/// `X := α · X` over the whole slice (unit stride).
pub fn scal<T: Scalar>(alpha: T, x: &mut [T]) -> Result<()> {
    T::scal(alpha, x, 1)
}

/// `Y := X` (vector copy, unit stride; sizes must match).
pub fn copy<T: Scalar>(x: &[T], y: &mut [T]) -> Result<()> {
    if x.len() != y.len() {
        return Err(Error::InvalidArgument(format!(
            "copy: x.len()={}, y.len()={}",
            x.len(),
            y.len()
        )));
    }
    T::copy(x, 1, y, 1)
}

/// Swap two unit-stride vectors (sizes must match).
pub fn swap<T: Scalar>(x: &mut [T], y: &mut [T]) -> Result<()> {
    if x.len() != y.len() {
        return Err(Error::InvalidArgument(format!(
            "swap: x.len()={}, y.len()={}",
            x.len(),
            y.len()
        )));
    }
    T::swap(x, 1, y, 1)
}

/// `Y := α · X + Y` (unit stride; sizes must match).
pub fn axpy<T: Scalar>(alpha: T, x: &[T], y: &mut [T]) -> Result<()> {
    if x.len() != y.len() {
        return Err(Error::InvalidArgument(format!(
            "axpy: x.len()={}, y.len()={}",
            x.len(),
            y.len()
        )));
    }
    T::axpy(alpha, x, 1, y, 1)
}

/// "Natural" inner product. See [`Scalar::dot`].
pub fn dot<T: Scalar>(x: &[T], y: &[T]) -> Result<T> {
    T::dot(x, 1, y, 1)
}

/// Unconjugated dot product. See [`Scalar::dotu`].
pub fn dotu<T: Scalar>(x: &[T], y: &[T]) -> Result<T> {
    T::dotu(x, 1, y, 1)
}

/// 2-norm `||X||₂` over a unit-stride slice.
pub fn nrm2<T: Scalar>(x: &[T]) -> Result<T::Real> {
    T::nrm2(x, 1)
}

/// 1-norm-ish absolute sum (see [`Scalar::asum`]).
pub fn asum<T: Scalar>(x: &[T]) -> Result<T::Real> {
    T::asum(x, 1)
}

/// 0-based index of the largest-magnitude element of `x`. Errors on empty.
pub fn iamax<T: Scalar>(x: &[T]) -> Result<usize> {
    T::iamax(x, 1)
}

/// 0-based index of the smallest-magnitude element of `x`. Errors on empty.
pub fn iamin<T: Scalar>(x: &[T]) -> Result<usize> {
    T::iamin(x, 1)
}

/// `C := α · op(A) · op(B) + β · C` for tightly-packed row-major matrices.
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

/// Compute the Givens rotation parameters `(c, s)` that zero out `b`.
/// On exit, `a` and `b` are modified per the BLAS standard.
pub fn rotg<T: RealScalar + Default>(a: &mut T, b: &mut T) -> (T, T) {
    let mut c = T::default();
    let mut s = T::default();
    T::rotg(a, b, &mut c, &mut s);
    (c, s)
}

/// Apply a Givens rotation to two unit-stride vectors of equal length.
pub fn rot<T: RealScalar>(x: &mut [T], y: &mut [T], c: T, s: T) -> Result<()> {
    if x.len() != y.len() {
        return Err(Error::InvalidArgument(format!(
            "rot: x.len()={}, y.len()={}",
            x.len(),
            y.len()
        )));
    }
    T::rot(x, 1, y, 1, c, s)
}

// ===========================================================================
//   Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, eps: f64) {
        assert!((a - b).abs() < eps, "|{a} - {b}| > {eps}");
    }

    // --- Real precisions --------------------------------------------------

    #[test]
    fn scal_f64() {
        let mut x = [1.0_f64, 2.0, 3.0];
        scal(2.0_f64, &mut x).unwrap();
        assert_eq!(x, [2.0, 4.0, 6.0]);
    }

    #[test]
    fn copy_swap_f32() {
        let x = [1.0_f32, 2.0, 3.0];
        let mut y = [0.0_f32; 3];
        copy(&x, &mut y).unwrap();
        assert_eq!(y, [1.0, 2.0, 3.0]);

        let mut a = [1.0_f32, 2.0];
        let mut b = [3.0_f32, 4.0];
        swap(&mut a, &mut b).unwrap();
        assert_eq!(a, [3.0, 4.0]);
        assert_eq!(b, [1.0, 2.0]);
    }

    #[test]
    fn dot_axpy_f64() {
        let x = [1.0_f64, 2.0, 3.0];
        let y = [4.0_f64, 5.0, 6.0];
        let r = dot(&x, &y).unwrap();
        assert_eq!(r, 32.0);

        let mut z = [10.0_f64, 20.0, 30.0];
        axpy(2.0, &x, &mut z).unwrap();
        assert_eq!(z, [12.0, 24.0, 36.0]);
    }

    #[test]
    fn nrm2_asum_real() {
        let x = [3.0_f64, -4.0];
        approx(nrm2(&x).unwrap(), 5.0, 1e-12);
        approx(asum(&x).unwrap(), 7.0, 1e-12);
    }

    #[test]
    fn iamax_iamin_real() {
        let x = [1.0_f64, -3.0, 2.0, 0.5];
        assert_eq!(iamax(&x).unwrap(), 1);
        assert_eq!(iamin(&x).unwrap(), 3);
    }

    #[test]
    fn iamax_empty_is_error() {
        let x: [f64; 0] = [];
        assert!(matches!(iamax(&x), Err(Error::InvalidArgument(_))));
    }

    #[test]
    fn rotg_zeros_b() {
        let mut a = 3.0_f64;
        let mut b = 4.0_f64;
        let (c, s) = rotg(&mut a, &mut b);
        approx(c * c + s * s, 1.0, 1e-12);
        let mut x = [3.0_f64];
        let mut y = [4.0_f64];
        rot(&mut x, &mut y, c, s).unwrap();
        approx(y[0], 0.0, 1e-12);
    }

    #[test]
    fn dsdot_f32_accumulates_in_f64() {
        let x = [1.0_f32; 100];
        let y = [1.0_f32; 100];
        let r = <f32 as RealScalar>::dsdot(&x, 1, &y, 1).unwrap();
        approx(r, 100.0, 1e-12);
    }

    #[test]
    fn strided_dot_f64() {
        let x = [1.0_f64, 2.0, 3.0, 4.0];
        let y = [10.0_f64, 20.0, 30.0, 40.0];
        let r = <f64 as Scalar>::dot(&x, 2, &y, 2).unwrap();
        approx(r, 100.0, 1e-12);
    }

    // --- Complex precisions -----------------------------------------------

    #[test]
    fn scal_complex64() {
        let mut x = [Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)];
        scal(Complex64::new(2.0, 0.0), &mut x).unwrap();
        assert_eq!(x[0], Complex64::new(2.0, 4.0));
        assert_eq!(x[1], Complex64::new(6.0, 8.0));
    }

    #[test]
    fn scal_real_complex64() {
        let mut x = [Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)];
        <Complex64 as Scalar>::scal_real(2.0, &mut x, 1).unwrap();
        assert_eq!(x[0], Complex64::new(2.0, 4.0));
        assert_eq!(x[1], Complex64::new(6.0, 8.0));
    }

    #[test]
    fn axpy_complex32() {
        let x = [Complex32::new(1.0, 0.0), Complex32::new(0.0, 1.0)];
        let mut y = [Complex32::new(1.0, 1.0), Complex32::new(2.0, 2.0)];
        axpy(Complex32::new(1.0, 0.0), &x, &mut y).unwrap();
        assert_eq!(y[0], Complex32::new(2.0, 1.0));
        assert_eq!(y[1], Complex32::new(2.0, 3.0));
    }

    #[test]
    fn dotc_dotu_differ_for_complex() {
        // x = [1+i, 2+0i], y = [1+0i, 0+1i]
        // dotu = (1+i)(1) + (2+0i)(0+i) = (1+i) + (0+2i) = 1 + 3i
        // dotc = conj(1+i)·1 + conj(2+0i)·(i) = (1-i) + (0+2i) = 1 + i
        let x = [Complex64::new(1.0, 1.0), Complex64::new(2.0, 0.0)];
        let y = [Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)];
        let u = dotu(&x, &y).unwrap();
        let c = dot(&x, &y).unwrap();
        assert_eq!(u, Complex64::new(1.0, 3.0));
        assert_eq!(c, Complex64::new(1.0, 1.0));
    }

    #[test]
    fn nrm2_complex_returns_real() {
        let x = [Complex64::new(3.0, 4.0)];
        let n: f64 = nrm2(&x).unwrap();
        approx(n, 5.0, 1e-12);

        let xs = [Complex32::new(3.0, 4.0)];
        let ns: f32 = nrm2(&xs).unwrap();
        assert!((ns - 5.0_f32).abs() < 1e-6);
    }

    #[test]
    fn asum_complex_re_plus_im() {
        let x = [
            Complex64::new(1.0, -2.0),
            Complex64::new(-3.0, 4.0),
        ];
        let s: f64 = asum(&x).unwrap();
        approx(s, 1.0 + 2.0 + 3.0 + 4.0, 1e-12);
    }

    #[test]
    fn iamax_complex() {
        let x = [
            Complex64::new(1.0, 0.0),
            Complex64::new(3.0, -4.0),
            Complex64::new(2.0, 0.0),
        ];
        assert_eq!(iamax(&x).unwrap(), 1);
    }

    #[test]
    fn copy_complex32() {
        let x = [Complex32::new(1.0, 2.0), Complex32::new(3.0, 4.0)];
        let mut y = [Complex32::ZERO; 2];
        copy(&x, &mut y).unwrap();
        assert_eq!(y, x);
    }

    #[test]
    fn swap_complex64() {
        let mut a = [Complex64::new(1.0, 2.0)];
        let mut b = [Complex64::new(3.0, 4.0)];
        swap(&mut a, &mut b).unwrap();
        assert_eq!(a[0], Complex64::new(3.0, 4.0));
        assert_eq!(b[0], Complex64::new(1.0, 2.0));
    }

    // --- GEMM regression -------------------------------------------------

    #[test]
    fn gemm_2x2_identity() {
        let a = [1.0_f64, 0.0, 0.0, 1.0];
        let b = [1.0_f64, 2.0, 3.0, 4.0];
        let mut c = [0.0_f64; 4];
        gemm(Trans::No, Trans::No, 2, 2, 2, 1.0, &a, &b, 0.0, &mut c).unwrap();
        for (got, want) in c.iter().zip(b.iter()) {
            assert!((got - want).abs() < 1e-12);
        }
    }

    #[test]
    fn gemm_2x3_times_3x2() {
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
    fn gemm_complex64_2x2() {
        let i = Complex64::ONE;
        let z = Complex64::ZERO;
        let a = [i, z, z, i];
        let b = [i, z, z, i];
        let mut c = [z; 4];
        gemm(Trans::No, Trans::No, 2, 2, 2, i, &a, &b, z, &mut c).unwrap();
        assert_eq!(c, [i, z, z, i]);
    }

    #[test]
    fn gemm_dim_mismatch_is_error() {
        let a = [1.0_f64; 6];
        let b = [1.0_f64; 4];
        let mut c = [0.0_f64; 4];
        let err = gemm(Trans::No, Trans::No, 2, 2, 3, 1.0, &a, &b, 0.0, &mut c).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)));
    }
}
