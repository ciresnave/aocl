//! Safe wrappers for AOCL-BLAS (BLIS) via the CBLAS interface.
//!
//! Comprehensive Level-1 (vector-vector) and core Level-2 (matrix-vector)
//! coverage across `f32`, `f64`, `Complex32`, and `Complex64`, plus GEMM
//! from Level 3. Banded / packed Level-2 variants and the rest of Level 3
//! are next.
//!
//! Trait split:
//! - [`Scalar`] — operations defined for all four precisions: every
//!   Level-1 op, plus `gemv` / `trmv` / `trsv`, plus `gemm`.
//! - [`RealScalar`] — operations defined only for `f32` / `f64`:
//!   Givens rotations, the precision-mixing dot variants, plus
//!   symmetric Level-2 ops (`symv`, `syr`, `syr2`, `ger`).
//! - [`ComplexScalar`] — operations defined only for `Complex32` /
//!   `Complex64`: Hermitian Level-2 ops (`hemv`, `her`, `her2`) and the
//!   complex rank-1 update variants (`geru`, `gerc`).
//!
//! All wrappers take Rust slices with bounds-checked dimensions; mismatched
//! sizes return [`Error::InvalidArgument`]. For routines not yet wrapped,
//! drop down to [`aocl-blas-sys`].

#![warn(missing_debug_implementations)]
#![cfg_attr(docsrs, feature(doc_cfg))]

use aocl_blas_sys as sys;
pub use aocl_error::{Error, Result};
pub use aocl_types::{Complex32, Complex64, Diag, Layout, Side, Trans, Uplo};
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

fn uplo_raw(u: Uplo) -> sys::CBLAS_UPLO {
    match u {
        Uplo::Upper => sys::CBLAS_UPLO_CblasUpper as sys::CBLAS_UPLO,
        Uplo::Lower => sys::CBLAS_UPLO_CblasLower as sys::CBLAS_UPLO,
    }
}

fn diag_raw(d: Diag) -> sys::CBLAS_DIAG {
    match d {
        Diag::NonUnit => sys::CBLAS_DIAG_CblasNonUnit as sys::CBLAS_DIAG,
        Diag::Unit => sys::CBLAS_DIAG_CblasUnit as sys::CBLAS_DIAG,
    }
}

// --- Level-2 dimension validators -----------------------------------------

/// Required leading dimension for an `m × n` row- or column-major matrix.
fn min_ld(layout: Layout, rows: usize, cols: usize) -> usize {
    match layout {
        Layout::RowMajor => cols.max(1),
        Layout::ColMajor => rows.max(1),
    }
}

/// Minimum slice length needed to address an `m × n` matrix with the
/// given leading dimension and storage order.
fn min_matrix_len(layout: Layout, rows: usize, cols: usize, ld: usize) -> usize {
    if rows == 0 || cols == 0 {
        return 0;
    }
    let (lead, trail) = match layout {
        Layout::RowMajor => (rows, cols),
        Layout::ColMajor => (cols, rows),
    };
    (lead - 1) * ld + trail
}

fn check_matrix(
    name: &str,
    layout: Layout,
    rows: usize,
    cols: usize,
    ld: usize,
    slice_len: usize,
) -> Result<()> {
    let needed_ld = min_ld(layout, rows, cols);
    if ld < needed_ld {
        return Err(Error::InvalidArgument(format!(
            "{name}: ld={ld} < {needed_ld}"
        )));
    }
    let needed = min_matrix_len(layout, rows, cols, ld);
    if slice_len < needed {
        return Err(Error::InvalidArgument(format!(
            "{name}: matrix slice length {slice_len} too small (need {needed})"
        )));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn check_gemv(
    layout: Layout,
    trans_a: Trans,
    m: usize,
    n: usize,
    a_len: usize,
    lda: usize,
    x_len: usize,
    inc_x: usize,
    y_len: usize,
    inc_y: usize,
) -> Result<()> {
    check_matrix("gemv: A", layout, m, n, lda, a_len)?;
    let (x_n, y_n) = match trans_a {
        Trans::No => (n, m),
        Trans::T | Trans::C => (m, n),
    };
    check_strided_len("gemv: x", x_len, x_n, inc_x)?;
    check_strided_len("gemv: y", y_len, y_n, inc_y)
}

fn check_n_square(
    name: &str,
    layout: Layout,
    n: usize,
    a_len: usize,
    lda: usize,
    x_len: usize,
    inc_x: usize,
) -> Result<()> {
    check_matrix(name, layout, n, n, lda, a_len)?;
    check_strided_len(&format!("{name}: x"), x_len, n, inc_x)
}

#[allow(clippy::too_many_arguments)]
fn check_n_square_xy(
    name: &str,
    layout: Layout,
    n: usize,
    a_len: usize,
    lda: usize,
    x_len: usize,
    inc_x: usize,
    y_len: usize,
    inc_y: usize,
) -> Result<()> {
    check_matrix(name, layout, n, n, lda, a_len)?;
    check_strided_len(&format!("{name}: x"), x_len, n, inc_x)?;
    check_strided_len(&format!("{name}: y"), y_len, n, inc_y)
}

#[allow(clippy::too_many_arguments)]
fn check_ger(
    layout: Layout,
    m: usize,
    n: usize,
    a_len: usize,
    lda: usize,
    x_len: usize,
    inc_x: usize,
    y_len: usize,
    inc_y: usize,
) -> Result<()> {
    check_matrix("ger: A", layout, m, n, lda, a_len)?;
    check_strided_len("ger: x", x_len, m, inc_x)?;
    check_strided_len("ger: y", y_len, n, inc_y)
}

fn side_to_n(side: Side, m: usize, n: usize) -> usize {
    match side {
        Side::Left => m,
        Side::Right => n,
    }
}

fn side_raw(s: Side) -> sys::CBLAS_SIDE {
    match s {
        Side::Left => sys::CBLAS_SIDE_CblasLeft as sys::CBLAS_SIDE,
        Side::Right => sys::CBLAS_SIDE_CblasRight as sys::CBLAS_SIDE,
    }
}

#[allow(clippy::too_many_arguments)]
fn check_symm(
    layout: Layout,
    side: Side,
    m: usize,
    n: usize,
    a_len: usize,
    lda: usize,
    b_len: usize,
    ldb: usize,
    c_len: usize,
    ldc: usize,
) -> Result<()> {
    let a_n = side_to_n(side, m, n);
    check_matrix("symm: A", layout, a_n, a_n, lda, a_len)?;
    check_matrix("symm: B", layout, m, n, ldb, b_len)?;
    check_matrix("symm: C", layout, m, n, ldc, c_len)
}

#[allow(clippy::too_many_arguments)]
fn check_syrk(
    layout: Layout,
    trans: Trans,
    n: usize,
    k: usize,
    a_len: usize,
    lda: usize,
    c_len: usize,
    ldc: usize,
) -> Result<()> {
    let (a_rows, a_cols) = match trans {
        Trans::No => (n, k),
        Trans::T | Trans::C => (k, n),
    };
    check_matrix("syrk: A", layout, a_rows, a_cols, lda, a_len)?;
    check_matrix("syrk: C", layout, n, n, ldc, c_len)
}

#[allow(clippy::too_many_arguments)]
fn check_syr2k(
    layout: Layout,
    trans: Trans,
    n: usize,
    k: usize,
    a_len: usize,
    lda: usize,
    b_len: usize,
    ldb: usize,
    c_len: usize,
    ldc: usize,
) -> Result<()> {
    let (a_rows, a_cols) = match trans {
        Trans::No => (n, k),
        Trans::T | Trans::C => (k, n),
    };
    check_matrix("syr2k: A", layout, a_rows, a_cols, lda, a_len)?;
    check_matrix("syr2k: B", layout, a_rows, a_cols, ldb, b_len)?;
    check_matrix("syr2k: C", layout, n, n, ldc, c_len)
}

#[allow(clippy::too_many_arguments)]
fn check_trxxm(
    name: &str,
    layout: Layout,
    side: Side,
    m: usize,
    n: usize,
    a_len: usize,
    lda: usize,
    b_len: usize,
    ldb: usize,
) -> Result<()> {
    let a_n = side_to_n(side, m, n);
    check_matrix(&format!("{name}: A"), layout, a_n, a_n, lda, a_len)?;
    check_matrix(&format!("{name}: B"), layout, m, n, ldb, b_len)
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

    // --- Level-2 (matrix-vector) -----------------------------------------

    /// `Y := α · op(A) · X + β · Y` where `op(A)` is `m × n`.
    #[allow(clippy::too_many_arguments)]
    fn gemv(
        layout: Layout,
        trans_a: Trans,
        m: usize,
        n: usize,
        alpha: Self,
        a: &[Self],
        lda: usize,
        x: &[Self],
        inc_x: usize,
        beta: Self,
        y: &mut [Self],
        inc_y: usize,
    ) -> Result<()>;

    /// `X := op(A) · X` for triangular `A` (`n × n`).
    #[allow(clippy::too_many_arguments)]
    fn trmv(
        layout: Layout,
        uplo: Uplo,
        trans_a: Trans,
        diag: Diag,
        n: usize,
        a: &[Self],
        lda: usize,
        x: &mut [Self],
        inc_x: usize,
    ) -> Result<()>;

    /// Solve `op(A) · X' = X` for triangular `A` (`n × n`), overwriting `X`
    /// with the solution `X'`.
    #[allow(clippy::too_many_arguments)]
    fn trsv(
        layout: Layout,
        uplo: Uplo,
        trans_a: Trans,
        diag: Diag,
        n: usize,
        a: &[Self],
        lda: usize,
        x: &mut [Self],
        inc_x: usize,
    ) -> Result<()>;

    // --- Level-3 ---------------------------------------------------------

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

    /// `C := α · A · B + β · C` (Side::Left) or `C := α · B · A + β · C`
    /// (Side::Right) where `A` is symmetric (`m × m` for Left, `n × n`
    /// for Right). Only the `uplo` triangle of `A` is referenced.
    #[allow(clippy::too_many_arguments)]
    fn symm(
        layout: Layout,
        side: Side,
        uplo: Uplo,
        m: usize,
        n: usize,
        alpha: Self,
        a: &[Self],
        lda: usize,
        b: &[Self],
        ldb: usize,
        beta: Self,
        c: &mut [Self],
        ldc: usize,
    ) -> Result<()>;

    /// Symmetric rank-k update: `C := α · op(A) · op(A)ᵀ + β · C`.
    /// `op(A)` is `n × k`. Only the `uplo` triangle of `C` is updated.
    #[allow(clippy::too_many_arguments)]
    fn syrk(
        layout: Layout,
        uplo: Uplo,
        trans: Trans,
        n: usize,
        k: usize,
        alpha: Self,
        a: &[Self],
        lda: usize,
        beta: Self,
        c: &mut [Self],
        ldc: usize,
    ) -> Result<()>;

    /// Symmetric rank-2k update:
    /// `C := α · op(A) · op(B)ᵀ + α · op(B) · op(A)ᵀ + β · C`.
    #[allow(clippy::too_many_arguments)]
    fn syr2k(
        layout: Layout,
        uplo: Uplo,
        trans: Trans,
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

    /// `B := α · op(A) · B` (Side::Left) or `B := α · B · op(A)`
    /// (Side::Right) where `A` is triangular.
    #[allow(clippy::too_many_arguments)]
    fn trmm(
        layout: Layout,
        side: Side,
        uplo: Uplo,
        trans_a: Trans,
        diag: Diag,
        m: usize,
        n: usize,
        alpha: Self,
        a: &[Self],
        lda: usize,
        b: &mut [Self],
        ldb: usize,
    ) -> Result<()>;

    /// Solve `op(A) · X = α · B` (Side::Left) or `X · op(A) = α · B`
    /// (Side::Right) for triangular `A`, overwriting `B` with `X`.
    #[allow(clippy::too_many_arguments)]
    fn trsm(
        layout: Layout,
        side: Side,
        uplo: Uplo,
        trans_a: Trans,
        diag: Diag,
        m: usize,
        n: usize,
        alpha: Self,
        a: &[Self],
        lda: usize,
        b: &mut [Self],
        ldb: usize,
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

    /// `Y := α · A · X + β · Y` for symmetric `A` (`n × n`).
    /// Only the triangle indicated by `uplo` is referenced.
    #[allow(clippy::too_many_arguments)]
    fn symv(
        layout: Layout,
        uplo: Uplo,
        n: usize,
        alpha: Self,
        a: &[Self],
        lda: usize,
        x: &[Self],
        inc_x: usize,
        beta: Self,
        y: &mut [Self],
        inc_y: usize,
    ) -> Result<()>;

    /// Symmetric rank-1 update: `A := α · X · Xᵀ + A` for symmetric `A`.
    /// Only the triangle indicated by `uplo` is referenced and updated.
    #[allow(clippy::too_many_arguments)]
    fn syr(
        layout: Layout,
        uplo: Uplo,
        n: usize,
        alpha: Self,
        x: &[Self],
        inc_x: usize,
        a: &mut [Self],
        lda: usize,
    ) -> Result<()>;

    /// Symmetric rank-2 update: `A := α · X · Yᵀ + α · Y · Xᵀ + A`.
    #[allow(clippy::too_many_arguments)]
    fn syr2(
        layout: Layout,
        uplo: Uplo,
        n: usize,
        alpha: Self,
        x: &[Self],
        inc_x: usize,
        y: &[Self],
        inc_y: usize,
        a: &mut [Self],
        lda: usize,
    ) -> Result<()>;

    /// General rank-1 update: `A := α · X · Yᵀ + A` for `m × n` `A`.
    #[allow(clippy::too_many_arguments)]
    fn ger(
        layout: Layout,
        m: usize,
        n: usize,
        alpha: Self,
        x: &[Self],
        inc_x: usize,
        y: &[Self],
        inc_y: usize,
        a: &mut [Self],
        lda: usize,
    ) -> Result<()>;
}

/// Operations defined only for complex BLAS precisions (`Complex32`,
/// `Complex64`).
pub trait ComplexScalar: Scalar {
    /// `Y := α · A · X + β · Y` for Hermitian `A` (`n × n`).
    /// Only the triangle indicated by `uplo` is referenced.
    #[allow(clippy::too_many_arguments)]
    fn hemv(
        layout: Layout,
        uplo: Uplo,
        n: usize,
        alpha: Self,
        a: &[Self],
        lda: usize,
        x: &[Self],
        inc_x: usize,
        beta: Self,
        y: &mut [Self],
        inc_y: usize,
    ) -> Result<()>;

    /// Hermitian rank-1 update: `A := α · X · Xᴴ + A`. Note that AOCL/BLAS
    /// require the scalar `α` to be **real** here (the diagonal of `A`
    /// is enforced real). Pass `Self::Real`, which is `f32` for
    /// `Complex32` and `f64` for `Complex64`.
    #[allow(clippy::too_many_arguments)]
    fn her(
        layout: Layout,
        uplo: Uplo,
        n: usize,
        alpha: Self::Real,
        x: &[Self],
        inc_x: usize,
        a: &mut [Self],
        lda: usize,
    ) -> Result<()>;

    /// Hermitian rank-2 update: `A := α · X · Yᴴ + conj(α) · Y · Xᴴ + A`.
    #[allow(clippy::too_many_arguments)]
    fn her2(
        layout: Layout,
        uplo: Uplo,
        n: usize,
        alpha: Self,
        x: &[Self],
        inc_x: usize,
        y: &[Self],
        inc_y: usize,
        a: &mut [Self],
        lda: usize,
    ) -> Result<()>;

    /// Unconjugated rank-1 update: `A := α · X · Yᵀ + A` for `m × n` `A`.
    #[allow(clippy::too_many_arguments)]
    fn geru(
        layout: Layout,
        m: usize,
        n: usize,
        alpha: Self,
        x: &[Self],
        inc_x: usize,
        y: &[Self],
        inc_y: usize,
        a: &mut [Self],
        lda: usize,
    ) -> Result<()>;

    /// Conjugated rank-1 update: `A := α · X · Yᴴ + A` for `m × n` `A`.
    #[allow(clippy::too_many_arguments)]
    fn gerc(
        layout: Layout,
        m: usize,
        n: usize,
        alpha: Self,
        x: &[Self],
        inc_x: usize,
        y: &[Self],
        inc_y: usize,
        a: &mut [Self],
        lda: usize,
    ) -> Result<()>;

    // --- Level-3 Hermitian -----------------------------------------------

    /// `C := α · A · B + β · C` (Side::Left) or `C := α · B · A + β · C`
    /// (Side::Right) where `A` is Hermitian.
    #[allow(clippy::too_many_arguments)]
    fn hemm(
        layout: Layout,
        side: Side,
        uplo: Uplo,
        m: usize,
        n: usize,
        alpha: Self,
        a: &[Self],
        lda: usize,
        b: &[Self],
        ldb: usize,
        beta: Self,
        c: &mut [Self],
        ldc: usize,
    ) -> Result<()>;

    /// Hermitian rank-k update: `C := α · op(A) · op(A)ᴴ + β · C`.
    /// Both `α` and `β` are real (the diagonal of `C` is enforced real).
    /// For Hermitian, `Trans::T` is illegal; use `Trans::No` or
    /// `Trans::C`.
    #[allow(clippy::too_many_arguments)]
    fn herk(
        layout: Layout,
        uplo: Uplo,
        trans: Trans,
        n: usize,
        k: usize,
        alpha: Self::Real,
        a: &[Self],
        lda: usize,
        beta: Self::Real,
        c: &mut [Self],
        ldc: usize,
    ) -> Result<()>;

    /// Hermitian rank-2k update:
    /// `C := α · op(A) · op(B)ᴴ + conj(α) · op(B) · op(A)ᴴ + β · C`,
    /// where `β` is real.
    #[allow(clippy::too_many_arguments)]
    fn her2k(
        layout: Layout,
        uplo: Uplo,
        trans: Trans,
        n: usize,
        k: usize,
        alpha: Self,
        a: &[Self],
        lda: usize,
        b: &[Self],
        ldb: usize,
        beta: Self::Real,
        c: &mut [Self],
        ldc: usize,
    ) -> Result<()>;
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
        dsdot_fn = $dsdot_fn:expr,
        gemv = $gemv:ident,
        trmv = $trmv:ident,
        trsv = $trsv:ident,
        symv = $symv:ident,
        syr  = $syr:ident,
        syr2 = $syr2:ident,
        ger  = $ger:ident,
        symm = $symm:ident,
        syrk = $syrk:ident,
        syr2k = $syr2k:ident,
        trmm = $trmm:ident,
        trsm = $trsm:ident
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
            fn gemv(
                layout: Layout, trans_a: Trans,
                m: usize, n: usize,
                alpha: Self, a: &[Self], lda: usize,
                x: &[Self], inc_x: usize,
                beta: Self, y: &mut [Self], inc_y: usize,
            ) -> Result<()> {
                check_gemv(layout, trans_a, m, n, a.len(), lda, x.len(), inc_x, y.len(), inc_y)?;
                unsafe {
                    sys::$gemv(
                        layout_raw(layout), trans_raw(trans_a),
                        m as sys::f77_int, n as sys::f77_int,
                        alpha, a.as_ptr(), lda as sys::f77_int,
                        x.as_ptr(), inc_x as sys::f77_int,
                        beta, y.as_mut_ptr(), inc_y as sys::f77_int,
                    );
                }
                Ok(())
            }

            #[allow(clippy::too_many_arguments)]
            fn trmv(
                layout: Layout, uplo: Uplo, trans_a: Trans, diag: Diag,
                n: usize, a: &[Self], lda: usize,
                x: &mut [Self], inc_x: usize,
            ) -> Result<()> {
                check_n_square("trmv: A", layout, n, a.len(), lda, x.len(), inc_x)?;
                unsafe {
                    sys::$trmv(
                        layout_raw(layout), uplo_raw(uplo), trans_raw(trans_a), diag_raw(diag),
                        n as sys::f77_int, a.as_ptr(), lda as sys::f77_int,
                        x.as_mut_ptr(), inc_x as sys::f77_int,
                    );
                }
                Ok(())
            }

            #[allow(clippy::too_many_arguments)]
            fn trsv(
                layout: Layout, uplo: Uplo, trans_a: Trans, diag: Diag,
                n: usize, a: &[Self], lda: usize,
                x: &mut [Self], inc_x: usize,
            ) -> Result<()> {
                check_n_square("trsv: A", layout, n, a.len(), lda, x.len(), inc_x)?;
                unsafe {
                    sys::$trsv(
                        layout_raw(layout), uplo_raw(uplo), trans_raw(trans_a), diag_raw(diag),
                        n as sys::f77_int, a.as_ptr(), lda as sys::f77_int,
                        x.as_mut_ptr(), inc_x as sys::f77_int,
                    );
                }
                Ok(())
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

            #[allow(clippy::too_many_arguments)]
            fn symm(
                layout: Layout, side: Side, uplo: Uplo,
                m: usize, n: usize,
                alpha: Self, a: &[Self], lda: usize,
                b: &[Self], ldb: usize,
                beta: Self, c: &mut [Self], ldc: usize,
            ) -> Result<()> {
                check_symm(layout, side, m, n, a.len(), lda, b.len(), ldb, c.len(), ldc)?;
                unsafe {
                    sys::$symm(
                        layout_raw(layout), side_raw(side), uplo_raw(uplo),
                        m as sys::f77_int, n as sys::f77_int,
                        alpha, a.as_ptr(), lda as sys::f77_int,
                        b.as_ptr(), ldb as sys::f77_int,
                        beta, c.as_mut_ptr(), ldc as sys::f77_int,
                    );
                }
                Ok(())
            }

            #[allow(clippy::too_many_arguments)]
            fn syrk(
                layout: Layout, uplo: Uplo, trans: Trans,
                n: usize, k: usize,
                alpha: Self, a: &[Self], lda: usize,
                beta: Self, c: &mut [Self], ldc: usize,
            ) -> Result<()> {
                check_syrk(layout, trans, n, k, a.len(), lda, c.len(), ldc)?;
                unsafe {
                    sys::$syrk(
                        layout_raw(layout), uplo_raw(uplo), trans_raw(trans),
                        n as sys::f77_int, k as sys::f77_int,
                        alpha, a.as_ptr(), lda as sys::f77_int,
                        beta, c.as_mut_ptr(), ldc as sys::f77_int,
                    );
                }
                Ok(())
            }

            #[allow(clippy::too_many_arguments)]
            fn syr2k(
                layout: Layout, uplo: Uplo, trans: Trans,
                n: usize, k: usize,
                alpha: Self, a: &[Self], lda: usize,
                b: &[Self], ldb: usize,
                beta: Self, c: &mut [Self], ldc: usize,
            ) -> Result<()> {
                check_syr2k(layout, trans, n, k, a.len(), lda, b.len(), ldb, c.len(), ldc)?;
                unsafe {
                    sys::$syr2k(
                        layout_raw(layout), uplo_raw(uplo), trans_raw(trans),
                        n as sys::f77_int, k as sys::f77_int,
                        alpha, a.as_ptr(), lda as sys::f77_int,
                        b.as_ptr(), ldb as sys::f77_int,
                        beta, c.as_mut_ptr(), ldc as sys::f77_int,
                    );
                }
                Ok(())
            }

            #[allow(clippy::too_many_arguments)]
            fn trmm(
                layout: Layout, side: Side, uplo: Uplo,
                trans_a: Trans, diag: Diag,
                m: usize, n: usize,
                alpha: Self, a: &[Self], lda: usize,
                b: &mut [Self], ldb: usize,
            ) -> Result<()> {
                check_trxxm("trmm", layout, side, m, n, a.len(), lda, b.len(), ldb)?;
                unsafe {
                    sys::$trmm(
                        layout_raw(layout), side_raw(side), uplo_raw(uplo),
                        trans_raw(trans_a), diag_raw(diag),
                        m as sys::f77_int, n as sys::f77_int,
                        alpha, a.as_ptr(), lda as sys::f77_int,
                        b.as_mut_ptr(), ldb as sys::f77_int,
                    );
                }
                Ok(())
            }

            #[allow(clippy::too_many_arguments)]
            fn trsm(
                layout: Layout, side: Side, uplo: Uplo,
                trans_a: Trans, diag: Diag,
                m: usize, n: usize,
                alpha: Self, a: &[Self], lda: usize,
                b: &mut [Self], ldb: usize,
            ) -> Result<()> {
                check_trxxm("trsm", layout, side, m, n, a.len(), lda, b.len(), ldb)?;
                unsafe {
                    sys::$trsm(
                        layout_raw(layout), side_raw(side), uplo_raw(uplo),
                        trans_raw(trans_a), diag_raw(diag),
                        m as sys::f77_int, n as sys::f77_int,
                        alpha, a.as_ptr(), lda as sys::f77_int,
                        b.as_mut_ptr(), ldb as sys::f77_int,
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

            #[allow(clippy::too_many_arguments)]
            fn symv(
                layout: Layout, uplo: Uplo, n: usize,
                alpha: Self, a: &[Self], lda: usize,
                x: &[Self], inc_x: usize,
                beta: Self, y: &mut [Self], inc_y: usize,
            ) -> Result<()> {
                check_n_square_xy("symv: A", layout, n, a.len(), lda, x.len(), inc_x, y.len(), inc_y)?;
                unsafe {
                    sys::$symv(
                        layout_raw(layout), uplo_raw(uplo),
                        n as sys::f77_int, alpha, a.as_ptr(), lda as sys::f77_int,
                        x.as_ptr(), inc_x as sys::f77_int,
                        beta, y.as_mut_ptr(), inc_y as sys::f77_int,
                    );
                }
                Ok(())
            }

            #[allow(clippy::too_many_arguments)]
            fn syr(
                layout: Layout, uplo: Uplo, n: usize,
                alpha: Self, x: &[Self], inc_x: usize,
                a: &mut [Self], lda: usize,
            ) -> Result<()> {
                check_n_square("syr: A", layout, n, a.len(), lda, x.len(), inc_x)?;
                unsafe {
                    sys::$syr(
                        layout_raw(layout), uplo_raw(uplo),
                        n as sys::f77_int, alpha, x.as_ptr(), inc_x as sys::f77_int,
                        a.as_mut_ptr(), lda as sys::f77_int,
                    );
                }
                Ok(())
            }

            #[allow(clippy::too_many_arguments)]
            fn syr2(
                layout: Layout, uplo: Uplo, n: usize,
                alpha: Self, x: &[Self], inc_x: usize,
                y: &[Self], inc_y: usize,
                a: &mut [Self], lda: usize,
            ) -> Result<()> {
                check_n_square_xy("syr2: A", layout, n, a.len(), lda, x.len(), inc_x, y.len(), inc_y)?;
                unsafe {
                    sys::$syr2(
                        layout_raw(layout), uplo_raw(uplo),
                        n as sys::f77_int, alpha,
                        x.as_ptr(), inc_x as sys::f77_int,
                        y.as_ptr(), inc_y as sys::f77_int,
                        a.as_mut_ptr(), lda as sys::f77_int,
                    );
                }
                Ok(())
            }

            #[allow(clippy::too_many_arguments)]
            fn ger(
                layout: Layout, m: usize, n: usize,
                alpha: Self, x: &[Self], inc_x: usize,
                y: &[Self], inc_y: usize,
                a: &mut [Self], lda: usize,
            ) -> Result<()> {
                check_ger(layout, m, n, a.len(), lda, x.len(), inc_x, y.len(), inc_y)?;
                unsafe {
                    sys::$ger(
                        layout_raw(layout),
                        m as sys::f77_int, n as sys::f77_int,
                        alpha,
                        x.as_ptr(), inc_x as sys::f77_int,
                        y.as_ptr(), inc_y as sys::f77_int,
                        a.as_mut_ptr(), lda as sys::f77_int,
                    );
                }
                Ok(())
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
    dsdot_fn = (|n, x, ix, y, iy| unsafe { sys::cblas_dsdot(n, x, ix, y, iy) }),
    gemv = cblas_sgemv, trmv = cblas_strmv, trsv = cblas_strsv,
    symv = cblas_ssymv, syr = cblas_ssyr, syr2 = cblas_ssyr2, ger = cblas_sger,
    symm = cblas_ssymm, syrk = cblas_ssyrk, syr2k = cblas_ssyr2k,
    trmm = cblas_strmm, trsm = cblas_strsm
);
impl_real_scalar!(
    f64,
    scal = cblas_dscal, copy = cblas_dcopy, swap = cblas_dswap, axpy = cblas_daxpy,
    dot = cblas_ddot, nrm2 = cblas_dnrm2, asum = cblas_dasum,
    iamax = cblas_idamax, iamin = cblas_idamin, gemm = cblas_dgemm,
    rotg = cblas_drotg, rot = cblas_drot,
    // For f64 dsdot semantically equals dot on f64 inputs; we forward to ddot.
    dsdot_fn = (|n, x, ix, y, iy| unsafe { sys::cblas_ddot(n, x, ix, y, iy) }),
    gemv = cblas_dgemv, trmv = cblas_dtrmv, trsv = cblas_dtrsv,
    symv = cblas_dsymv, syr = cblas_dsyr, syr2 = cblas_dsyr2, ger = cblas_dger,
    symm = cblas_dsymm, syrk = cblas_dsyrk, syr2k = cblas_dsyr2k,
    trmm = cblas_dtrmm, trsm = cblas_dtrsm
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
        gemm = $gemm:ident,
        gemv = $gemv:ident, trmv = $trmv:ident, trsv = $trsv:ident,
        hemv = $hemv:ident, her = $her:ident, her2 = $her2:ident,
        geru = $geru:ident, gerc = $gerc:ident,
        symm = $symm:ident, syrk = $syrk:ident, syr2k = $syr2k:ident,
        trmm = $trmm:ident, trsm = $trsm:ident,
        hemm = $hemm:ident, herk = $herk:ident, her2k = $her2k:ident
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
            fn gemv(
                layout: Layout, trans_a: Trans,
                m: usize, n: usize,
                alpha: Self, a: &[Self], lda: usize,
                x: &[Self], inc_x: usize,
                beta: Self, y: &mut [Self], inc_y: usize,
            ) -> Result<()> {
                check_gemv(layout, trans_a, m, n, a.len(), lda, x.len(), inc_x, y.len(), inc_y)?;
                unsafe {
                    sys::$gemv(
                        layout_raw(layout), trans_raw(trans_a),
                        m as sys::f77_int, n as sys::f77_int,
                        &alpha as *const Self as *const std::os::raw::c_void,
                        a.as_ptr() as *const std::os::raw::c_void, lda as sys::f77_int,
                        x.as_ptr() as *const std::os::raw::c_void, inc_x as sys::f77_int,
                        &beta as *const Self as *const std::os::raw::c_void,
                        y.as_mut_ptr() as *mut std::os::raw::c_void, inc_y as sys::f77_int,
                    );
                }
                Ok(())
            }

            #[allow(clippy::too_many_arguments)]
            fn trmv(
                layout: Layout, uplo: Uplo, trans_a: Trans, diag: Diag,
                n: usize, a: &[Self], lda: usize,
                x: &mut [Self], inc_x: usize,
            ) -> Result<()> {
                check_n_square("trmv: A", layout, n, a.len(), lda, x.len(), inc_x)?;
                unsafe {
                    sys::$trmv(
                        layout_raw(layout), uplo_raw(uplo), trans_raw(trans_a), diag_raw(diag),
                        n as sys::f77_int,
                        a.as_ptr() as *const std::os::raw::c_void, lda as sys::f77_int,
                        x.as_mut_ptr() as *mut std::os::raw::c_void, inc_x as sys::f77_int,
                    );
                }
                Ok(())
            }

            #[allow(clippy::too_many_arguments)]
            fn trsv(
                layout: Layout, uplo: Uplo, trans_a: Trans, diag: Diag,
                n: usize, a: &[Self], lda: usize,
                x: &mut [Self], inc_x: usize,
            ) -> Result<()> {
                check_n_square("trsv: A", layout, n, a.len(), lda, x.len(), inc_x)?;
                unsafe {
                    sys::$trsv(
                        layout_raw(layout), uplo_raw(uplo), trans_raw(trans_a), diag_raw(diag),
                        n as sys::f77_int,
                        a.as_ptr() as *const std::os::raw::c_void, lda as sys::f77_int,
                        x.as_mut_ptr() as *mut std::os::raw::c_void, inc_x as sys::f77_int,
                    );
                }
                Ok(())
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

            #[allow(clippy::too_many_arguments)]
            fn symm(
                layout: Layout, side: Side, uplo: Uplo,
                m: usize, n: usize,
                alpha: Self, a: &[Self], lda: usize,
                b: &[Self], ldb: usize,
                beta: Self, c: &mut [Self], ldc: usize,
            ) -> Result<()> {
                check_symm(layout, side, m, n, a.len(), lda, b.len(), ldb, c.len(), ldc)?;
                unsafe {
                    sys::$symm(
                        layout_raw(layout), side_raw(side), uplo_raw(uplo),
                        m as sys::f77_int, n as sys::f77_int,
                        &alpha as *const Self as *const std::os::raw::c_void,
                        a.as_ptr() as *const std::os::raw::c_void, lda as sys::f77_int,
                        b.as_ptr() as *const std::os::raw::c_void, ldb as sys::f77_int,
                        &beta as *const Self as *const std::os::raw::c_void,
                        c.as_mut_ptr() as *mut std::os::raw::c_void, ldc as sys::f77_int,
                    );
                }
                Ok(())
            }

            #[allow(clippy::too_many_arguments)]
            fn syrk(
                layout: Layout, uplo: Uplo, trans: Trans,
                n: usize, k: usize,
                alpha: Self, a: &[Self], lda: usize,
                beta: Self, c: &mut [Self], ldc: usize,
            ) -> Result<()> {
                check_syrk(layout, trans, n, k, a.len(), lda, c.len(), ldc)?;
                unsafe {
                    sys::$syrk(
                        layout_raw(layout), uplo_raw(uplo), trans_raw(trans),
                        n as sys::f77_int, k as sys::f77_int,
                        &alpha as *const Self as *const std::os::raw::c_void,
                        a.as_ptr() as *const std::os::raw::c_void, lda as sys::f77_int,
                        &beta as *const Self as *const std::os::raw::c_void,
                        c.as_mut_ptr() as *mut std::os::raw::c_void, ldc as sys::f77_int,
                    );
                }
                Ok(())
            }

            #[allow(clippy::too_many_arguments)]
            fn syr2k(
                layout: Layout, uplo: Uplo, trans: Trans,
                n: usize, k: usize,
                alpha: Self, a: &[Self], lda: usize,
                b: &[Self], ldb: usize,
                beta: Self, c: &mut [Self], ldc: usize,
            ) -> Result<()> {
                check_syr2k(layout, trans, n, k, a.len(), lda, b.len(), ldb, c.len(), ldc)?;
                unsafe {
                    sys::$syr2k(
                        layout_raw(layout), uplo_raw(uplo), trans_raw(trans),
                        n as sys::f77_int, k as sys::f77_int,
                        &alpha as *const Self as *const std::os::raw::c_void,
                        a.as_ptr() as *const std::os::raw::c_void, lda as sys::f77_int,
                        b.as_ptr() as *const std::os::raw::c_void, ldb as sys::f77_int,
                        &beta as *const Self as *const std::os::raw::c_void,
                        c.as_mut_ptr() as *mut std::os::raw::c_void, ldc as sys::f77_int,
                    );
                }
                Ok(())
            }

            #[allow(clippy::too_many_arguments)]
            fn trmm(
                layout: Layout, side: Side, uplo: Uplo,
                trans_a: Trans, diag: Diag,
                m: usize, n: usize,
                alpha: Self, a: &[Self], lda: usize,
                b: &mut [Self], ldb: usize,
            ) -> Result<()> {
                check_trxxm("trmm", layout, side, m, n, a.len(), lda, b.len(), ldb)?;
                unsafe {
                    sys::$trmm(
                        layout_raw(layout), side_raw(side), uplo_raw(uplo),
                        trans_raw(trans_a), diag_raw(diag),
                        m as sys::f77_int, n as sys::f77_int,
                        &alpha as *const Self as *const std::os::raw::c_void,
                        a.as_ptr() as *const std::os::raw::c_void, lda as sys::f77_int,
                        b.as_mut_ptr() as *mut std::os::raw::c_void, ldb as sys::f77_int,
                    );
                }
                Ok(())
            }

            #[allow(clippy::too_many_arguments)]
            fn trsm(
                layout: Layout, side: Side, uplo: Uplo,
                trans_a: Trans, diag: Diag,
                m: usize, n: usize,
                alpha: Self, a: &[Self], lda: usize,
                b: &mut [Self], ldb: usize,
            ) -> Result<()> {
                check_trxxm("trsm", layout, side, m, n, a.len(), lda, b.len(), ldb)?;
                unsafe {
                    sys::$trsm(
                        layout_raw(layout), side_raw(side), uplo_raw(uplo),
                        trans_raw(trans_a), diag_raw(diag),
                        m as sys::f77_int, n as sys::f77_int,
                        &alpha as *const Self as *const std::os::raw::c_void,
                        a.as_ptr() as *const std::os::raw::c_void, lda as sys::f77_int,
                        b.as_mut_ptr() as *mut std::os::raw::c_void, ldb as sys::f77_int,
                    );
                }
                Ok(())
            }
        }

        impl ComplexScalar for $t {
            #[allow(clippy::too_many_arguments)]
            fn hemv(
                layout: Layout, uplo: Uplo, n: usize,
                alpha: Self, a: &[Self], lda: usize,
                x: &[Self], inc_x: usize,
                beta: Self, y: &mut [Self], inc_y: usize,
            ) -> Result<()> {
                check_n_square_xy("hemv: A", layout, n, a.len(), lda, x.len(), inc_x, y.len(), inc_y)?;
                unsafe {
                    sys::$hemv(
                        layout_raw(layout), uplo_raw(uplo),
                        n as sys::f77_int,
                        &alpha as *const Self as *const std::os::raw::c_void,
                        a.as_ptr() as *const std::os::raw::c_void, lda as sys::f77_int,
                        x.as_ptr() as *const std::os::raw::c_void, inc_x as sys::f77_int,
                        &beta as *const Self as *const std::os::raw::c_void,
                        y.as_mut_ptr() as *mut std::os::raw::c_void, inc_y as sys::f77_int,
                    );
                }
                Ok(())
            }

            #[allow(clippy::too_many_arguments)]
            fn her(
                layout: Layout, uplo: Uplo, n: usize,
                alpha: Self::Real, x: &[Self], inc_x: usize,
                a: &mut [Self], lda: usize,
            ) -> Result<()> {
                check_n_square("her: A", layout, n, a.len(), lda, x.len(), inc_x)?;
                unsafe {
                    sys::$her(
                        layout_raw(layout), uplo_raw(uplo),
                        n as sys::f77_int, alpha,
                        x.as_ptr() as *const std::os::raw::c_void, inc_x as sys::f77_int,
                        a.as_mut_ptr() as *mut std::os::raw::c_void, lda as sys::f77_int,
                    );
                }
                Ok(())
            }

            #[allow(clippy::too_many_arguments)]
            fn her2(
                layout: Layout, uplo: Uplo, n: usize,
                alpha: Self, x: &[Self], inc_x: usize,
                y: &[Self], inc_y: usize,
                a: &mut [Self], lda: usize,
            ) -> Result<()> {
                check_n_square_xy("her2: A", layout, n, a.len(), lda, x.len(), inc_x, y.len(), inc_y)?;
                unsafe {
                    sys::$her2(
                        layout_raw(layout), uplo_raw(uplo),
                        n as sys::f77_int,
                        &alpha as *const Self as *const std::os::raw::c_void,
                        x.as_ptr() as *const std::os::raw::c_void, inc_x as sys::f77_int,
                        y.as_ptr() as *const std::os::raw::c_void, inc_y as sys::f77_int,
                        a.as_mut_ptr() as *mut std::os::raw::c_void, lda as sys::f77_int,
                    );
                }
                Ok(())
            }

            #[allow(clippy::too_many_arguments)]
            fn geru(
                layout: Layout, m: usize, n: usize,
                alpha: Self, x: &[Self], inc_x: usize,
                y: &[Self], inc_y: usize,
                a: &mut [Self], lda: usize,
            ) -> Result<()> {
                check_ger(layout, m, n, a.len(), lda, x.len(), inc_x, y.len(), inc_y)?;
                unsafe {
                    sys::$geru(
                        layout_raw(layout),
                        m as sys::f77_int, n as sys::f77_int,
                        &alpha as *const Self as *const std::os::raw::c_void,
                        x.as_ptr() as *const std::os::raw::c_void, inc_x as sys::f77_int,
                        y.as_ptr() as *const std::os::raw::c_void, inc_y as sys::f77_int,
                        a.as_mut_ptr() as *mut std::os::raw::c_void, lda as sys::f77_int,
                    );
                }
                Ok(())
            }

            #[allow(clippy::too_many_arguments)]
            fn gerc(
                layout: Layout, m: usize, n: usize,
                alpha: Self, x: &[Self], inc_x: usize,
                y: &[Self], inc_y: usize,
                a: &mut [Self], lda: usize,
            ) -> Result<()> {
                check_ger(layout, m, n, a.len(), lda, x.len(), inc_x, y.len(), inc_y)?;
                unsafe {
                    sys::$gerc(
                        layout_raw(layout),
                        m as sys::f77_int, n as sys::f77_int,
                        &alpha as *const Self as *const std::os::raw::c_void,
                        x.as_ptr() as *const std::os::raw::c_void, inc_x as sys::f77_int,
                        y.as_ptr() as *const std::os::raw::c_void, inc_y as sys::f77_int,
                        a.as_mut_ptr() as *mut std::os::raw::c_void, lda as sys::f77_int,
                    );
                }
                Ok(())
            }

            #[allow(clippy::too_many_arguments)]
            fn hemm(
                layout: Layout, side: Side, uplo: Uplo,
                m: usize, n: usize,
                alpha: Self, a: &[Self], lda: usize,
                b: &[Self], ldb: usize,
                beta: Self, c: &mut [Self], ldc: usize,
            ) -> Result<()> {
                check_symm(layout, side, m, n, a.len(), lda, b.len(), ldb, c.len(), ldc)?;
                unsafe {
                    sys::$hemm(
                        layout_raw(layout), side_raw(side), uplo_raw(uplo),
                        m as sys::f77_int, n as sys::f77_int,
                        &alpha as *const Self as *const std::os::raw::c_void,
                        a.as_ptr() as *const std::os::raw::c_void, lda as sys::f77_int,
                        b.as_ptr() as *const std::os::raw::c_void, ldb as sys::f77_int,
                        &beta as *const Self as *const std::os::raw::c_void,
                        c.as_mut_ptr() as *mut std::os::raw::c_void, ldc as sys::f77_int,
                    );
                }
                Ok(())
            }

            #[allow(clippy::too_many_arguments)]
            fn herk(
                layout: Layout, uplo: Uplo, trans: Trans,
                n: usize, k: usize,
                alpha: Self::Real, a: &[Self], lda: usize,
                beta: Self::Real, c: &mut [Self], ldc: usize,
            ) -> Result<()> {
                if matches!(trans, Trans::T) {
                    return Err(Error::InvalidArgument(
                        "herk: Trans::T is not allowed for Hermitian rank-k; use Trans::No or Trans::C".into()
                    ));
                }
                check_syrk(layout, trans, n, k, a.len(), lda, c.len(), ldc)?;
                unsafe {
                    sys::$herk(
                        layout_raw(layout), uplo_raw(uplo), trans_raw(trans),
                        n as sys::f77_int, k as sys::f77_int,
                        alpha,
                        a.as_ptr() as *const std::os::raw::c_void, lda as sys::f77_int,
                        beta,
                        c.as_mut_ptr() as *mut std::os::raw::c_void, ldc as sys::f77_int,
                    );
                }
                Ok(())
            }

            #[allow(clippy::too_many_arguments)]
            fn her2k(
                layout: Layout, uplo: Uplo, trans: Trans,
                n: usize, k: usize,
                alpha: Self, a: &[Self], lda: usize,
                b: &[Self], ldb: usize,
                beta: Self::Real, c: &mut [Self], ldc: usize,
            ) -> Result<()> {
                if matches!(trans, Trans::T) {
                    return Err(Error::InvalidArgument(
                        "her2k: Trans::T is not allowed for Hermitian rank-2k; use Trans::No or Trans::C".into()
                    ));
                }
                check_syr2k(layout, trans, n, k, a.len(), lda, b.len(), ldb, c.len(), ldc)?;
                unsafe {
                    sys::$her2k(
                        layout_raw(layout), uplo_raw(uplo), trans_raw(trans),
                        n as sys::f77_int, k as sys::f77_int,
                        &alpha as *const Self as *const std::os::raw::c_void,
                        a.as_ptr() as *const std::os::raw::c_void, lda as sys::f77_int,
                        b.as_ptr() as *const std::os::raw::c_void, ldb as sys::f77_int,
                        beta,
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
    gemm = cblas_cgemm,
    gemv = cblas_cgemv, trmv = cblas_ctrmv, trsv = cblas_ctrsv,
    hemv = cblas_chemv, her = cblas_cher, her2 = cblas_cher2,
    geru = cblas_cgeru, gerc = cblas_cgerc,
    symm = cblas_csymm, syrk = cblas_csyrk, syr2k = cblas_csyr2k,
    trmm = cblas_ctrmm, trsm = cblas_ctrsm,
    hemm = cblas_chemm, herk = cblas_cherk, her2k = cblas_cher2k
);
impl_complex_scalar!(
    Complex64, f64,
    scal = cblas_zscal, scal_real = cblas_zdscal,
    copy = cblas_zcopy, swap = cblas_zswap, axpy = cblas_zaxpy,
    dotu = cblas_zdotu_sub, dotc = cblas_zdotc_sub,
    nrm2 = cblas_dznrm2, asum = cblas_dzasum,
    iamax = cblas_izamax, iamin = cblas_izamin,
    gemm = cblas_zgemm,
    gemv = cblas_zgemv, trmv = cblas_ztrmv, trsv = cblas_ztrsv,
    hemv = cblas_zhemv, her = cblas_zher, her2 = cblas_zher2,
    geru = cblas_zgeru, gerc = cblas_zgerc,
    symm = cblas_zsymm, syrk = cblas_zsyrk, syr2k = cblas_zsyr2k,
    trmm = cblas_ztrmm, trsm = cblas_ztrsm,
    hemm = cblas_zhemm, herk = cblas_zherk, her2k = cblas_zher2k
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
//   Level-2 free-function convenience entry points
//   (tightly-packed row-major matrices; unit stride vectors)
// ===========================================================================

/// `Y := α · op(A) · X + β · Y` where `A` is `m × n` row-major
/// tightly-packed. `X` length must match `n` (no trans) or `m` (trans);
/// `Y` length must match the other.
#[allow(clippy::too_many_arguments)]
pub fn gemv<T: Scalar>(
    trans_a: Trans,
    m: usize,
    n: usize,
    alpha: T,
    a: &[T],
    x: &[T],
    beta: T,
    y: &mut [T],
) -> Result<()> {
    T::gemv(Layout::RowMajor, trans_a, m, n, alpha, a, n, x, 1, beta, y, 1)
}

/// `X := op(A) · X` for triangular row-major `A` (`n × n`).
pub fn trmv<T: Scalar>(
    uplo: Uplo,
    trans_a: Trans,
    diag: Diag,
    n: usize,
    a: &[T],
    x: &mut [T],
) -> Result<()> {
    T::trmv(Layout::RowMajor, uplo, trans_a, diag, n, a, n, x, 1)
}

/// Solve `op(A) · X' = X` for triangular row-major `A` (`n × n`).
pub fn trsv<T: Scalar>(
    uplo: Uplo,
    trans_a: Trans,
    diag: Diag,
    n: usize,
    a: &[T],
    x: &mut [T],
) -> Result<()> {
    T::trsv(Layout::RowMajor, uplo, trans_a, diag, n, a, n, x, 1)
}

/// `Y := α · A · X + β · Y` for symmetric row-major `A` (`n × n`).
#[allow(clippy::too_many_arguments)]
pub fn symv<T: RealScalar>(
    uplo: Uplo,
    n: usize,
    alpha: T,
    a: &[T],
    x: &[T],
    beta: T,
    y: &mut [T],
) -> Result<()> {
    T::symv(Layout::RowMajor, uplo, n, alpha, a, n, x, 1, beta, y, 1)
}

/// Symmetric rank-1 update `A := α · X · Xᵀ + A`.
pub fn syr<T: RealScalar>(
    uplo: Uplo,
    n: usize,
    alpha: T,
    x: &[T],
    a: &mut [T],
) -> Result<()> {
    T::syr(Layout::RowMajor, uplo, n, alpha, x, 1, a, n)
}

/// Symmetric rank-2 update `A := α · X · Yᵀ + α · Y · Xᵀ + A`.
pub fn syr2<T: RealScalar>(
    uplo: Uplo,
    n: usize,
    alpha: T,
    x: &[T],
    y: &[T],
    a: &mut [T],
) -> Result<()> {
    T::syr2(Layout::RowMajor, uplo, n, alpha, x, 1, y, 1, a, n)
}

/// General rank-1 update `A := α · X · Yᵀ + A` for `m × n` row-major `A`.
pub fn ger<T: RealScalar>(
    m: usize,
    n: usize,
    alpha: T,
    x: &[T],
    y: &[T],
    a: &mut [T],
) -> Result<()> {
    T::ger(Layout::RowMajor, m, n, alpha, x, 1, y, 1, a, n)
}

/// `Y := α · A · X + β · Y` for Hermitian row-major `A` (`n × n`).
#[allow(clippy::too_many_arguments)]
pub fn hemv<T: ComplexScalar>(
    uplo: Uplo,
    n: usize,
    alpha: T,
    a: &[T],
    x: &[T],
    beta: T,
    y: &mut [T],
) -> Result<()> {
    T::hemv(Layout::RowMajor, uplo, n, alpha, a, n, x, 1, beta, y, 1)
}

/// Hermitian rank-1 update `A := α · X · Xᴴ + A` (real scalar).
pub fn her<T: ComplexScalar>(
    uplo: Uplo,
    n: usize,
    alpha: T::Real,
    x: &[T],
    a: &mut [T],
) -> Result<()> {
    T::her(Layout::RowMajor, uplo, n, alpha, x, 1, a, n)
}

/// Hermitian rank-2 update `A := α · X · Yᴴ + conj(α) · Y · Xᴴ + A`.
pub fn her2<T: ComplexScalar>(
    uplo: Uplo,
    n: usize,
    alpha: T,
    x: &[T],
    y: &[T],
    a: &mut [T],
) -> Result<()> {
    T::her2(Layout::RowMajor, uplo, n, alpha, x, 1, y, 1, a, n)
}

/// Unconjugated rank-1 update `A := α · X · Yᵀ + A` for complex.
pub fn geru<T: ComplexScalar>(
    m: usize,
    n: usize,
    alpha: T,
    x: &[T],
    y: &[T],
    a: &mut [T],
) -> Result<()> {
    T::geru(Layout::RowMajor, m, n, alpha, x, 1, y, 1, a, n)
}

/// Conjugated rank-1 update `A := α · X · Yᴴ + A` for complex.
pub fn gerc<T: ComplexScalar>(
    m: usize,
    n: usize,
    alpha: T,
    x: &[T],
    y: &[T],
    a: &mut [T],
) -> Result<()> {
    T::gerc(Layout::RowMajor, m, n, alpha, x, 1, y, 1, a, n)
}

// ===========================================================================
//   Level-3 free-function convenience entry points
// ===========================================================================

/// `C := α · A · B + β · C` (Side::Left) or `C := α · B · A + β · C`
/// (Side::Right) where `A` is symmetric, row-major tightly-packed.
#[allow(clippy::too_many_arguments)]
pub fn symm<T: Scalar>(
    side: Side,
    uplo: Uplo,
    m: usize,
    n: usize,
    alpha: T,
    a: &[T],
    b: &[T],
    beta: T,
    c: &mut [T],
) -> Result<()> {
    let a_n = side_to_n(side, m, n);
    T::symm(Layout::RowMajor, side, uplo, m, n, alpha, a, a_n, b, n, beta, c, n)
}

/// Symmetric rank-k update `C := α · op(A) · op(A)ᵀ + β · C`,
/// row-major tightly-packed.
#[allow(clippy::too_many_arguments)]
pub fn syrk<T: Scalar>(
    uplo: Uplo,
    trans: Trans,
    n: usize,
    k: usize,
    alpha: T,
    a: &[T],
    beta: T,
    c: &mut [T],
) -> Result<()> {
    let lda = match trans {
        Trans::No => k,
        _ => n,
    };
    T::syrk(Layout::RowMajor, uplo, trans, n, k, alpha, a, lda, beta, c, n)
}

/// Symmetric rank-2k update.
#[allow(clippy::too_many_arguments)]
pub fn syr2k<T: Scalar>(
    uplo: Uplo,
    trans: Trans,
    n: usize,
    k: usize,
    alpha: T,
    a: &[T],
    b: &[T],
    beta: T,
    c: &mut [T],
) -> Result<()> {
    let lda = match trans {
        Trans::No => k,
        _ => n,
    };
    let ldb = lda;
    T::syr2k(Layout::RowMajor, uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, n)
}

/// Triangular matrix-matrix multiply `B := α · op(A) · B` (Side::Left)
/// or `B := α · B · op(A)` (Side::Right). Row-major tightly-packed.
#[allow(clippy::too_many_arguments)]
pub fn trmm<T: Scalar>(
    side: Side,
    uplo: Uplo,
    trans_a: Trans,
    diag: Diag,
    m: usize,
    n: usize,
    alpha: T,
    a: &[T],
    b: &mut [T],
) -> Result<()> {
    let a_n = side_to_n(side, m, n);
    T::trmm(Layout::RowMajor, side, uplo, trans_a, diag, m, n, alpha, a, a_n, b, n)
}

/// Triangular solve with multiple right-hand sides.
#[allow(clippy::too_many_arguments)]
pub fn trsm<T: Scalar>(
    side: Side,
    uplo: Uplo,
    trans_a: Trans,
    diag: Diag,
    m: usize,
    n: usize,
    alpha: T,
    a: &[T],
    b: &mut [T],
) -> Result<()> {
    let a_n = side_to_n(side, m, n);
    T::trsm(Layout::RowMajor, side, uplo, trans_a, diag, m, n, alpha, a, a_n, b, n)
}

/// Hermitian matrix-matrix multiply (complex precisions).
#[allow(clippy::too_many_arguments)]
pub fn hemm<T: ComplexScalar>(
    side: Side,
    uplo: Uplo,
    m: usize,
    n: usize,
    alpha: T,
    a: &[T],
    b: &[T],
    beta: T,
    c: &mut [T],
) -> Result<()> {
    let a_n = side_to_n(side, m, n);
    T::hemm(Layout::RowMajor, side, uplo, m, n, alpha, a, a_n, b, n, beta, c, n)
}

/// Hermitian rank-k update `C := α · op(A) · op(A)ᴴ + β · C` with
/// real `α` and `β`. `trans` must be `Trans::No` or `Trans::C`.
#[allow(clippy::too_many_arguments)]
pub fn herk<T: ComplexScalar>(
    uplo: Uplo,
    trans: Trans,
    n: usize,
    k: usize,
    alpha: T::Real,
    a: &[T],
    beta: T::Real,
    c: &mut [T],
) -> Result<()> {
    let lda = match trans {
        Trans::No => k,
        _ => n,
    };
    T::herk(Layout::RowMajor, uplo, trans, n, k, alpha, a, lda, beta, c, n)
}

/// Hermitian rank-2k update.
#[allow(clippy::too_many_arguments)]
pub fn her2k<T: ComplexScalar>(
    uplo: Uplo,
    trans: Trans,
    n: usize,
    k: usize,
    alpha: T,
    a: &[T],
    b: &[T],
    beta: T::Real,
    c: &mut [T],
) -> Result<()> {
    let lda = match trans {
        Trans::No => k,
        _ => n,
    };
    let ldb = lda;
    T::her2k(Layout::RowMajor, uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, n)
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

    // --- Level 2: gemv ----------------------------------------------------

    #[test]
    fn gemv_f64_no_trans() {
        // A = [[1,2,3],[4,5,6]] (2×3), x=[1,1,1], y=A·x = [6, 15]
        let a = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = [1.0_f64; 3];
        let mut y = [0.0_f64; 2];
        gemv(Trans::No, 2, 3, 1.0, &a, &x, 0.0, &mut y).unwrap();
        assert_eq!(y, [6.0, 15.0]);
    }

    #[test]
    fn gemv_f64_transpose() {
        // A = [[1,2,3],[4,5,6]] (2×3), x=[1,1] → Aᵀ·x = [5, 7, 9]
        let a = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = [1.0_f64; 2];
        let mut y = [0.0_f64; 3];
        gemv(Trans::T, 2, 3, 1.0, &a, &x, 0.0, &mut y).unwrap();
        assert_eq!(y, [5.0, 7.0, 9.0]);
    }

    #[test]
    fn gemv_complex64() {
        // A = [[1+i, 0],[0, 1+i]], x = [1, 1] → A·x = [1+i, 1+i]
        let i = Complex64::new(1.0, 1.0);
        let z = Complex64::ZERO;
        let one = Complex64::ONE;
        let a = [i, z, z, i];
        let x = [one, one];
        let mut y = [z; 2];
        gemv(Trans::No, 2, 2, one, &a, &x, z, &mut y).unwrap();
        assert_eq!(y, [i, i]);
    }

    #[test]
    fn gemv_dim_mismatch() {
        let a = [1.0_f64; 6];
        let x = [1.0_f64; 2]; // wrong: needs 3 for no-trans 2×3
        let mut y = [0.0_f64; 2];
        let err = gemv(Trans::No, 2, 3, 1.0, &a, &x, 0.0, &mut y).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)));
    }

    // --- Level 2: trmv / trsv --------------------------------------------

    #[test]
    fn trmv_upper_unit() {
        // U = [[1, 2, 3],[0, 1, 4],[0, 0, 1]] (unit-triangular row-major,
        // diagonals implicit). x = [1, 2, 3] → U·x = [1+4+9, 2+12, 3] = [14, 14, 3]
        // BLAS uses the upper triangle including diagonal; with Diag::Unit
        // it ignores the actual diagonal entries.
        let a = [
            0.0_f64, 2.0, 3.0,
            0.0,     0.0, 4.0,
            0.0,     0.0, 0.0,
        ];
        let mut x = [1.0_f64, 2.0, 3.0];
        trmv(Uplo::Upper, Trans::No, Diag::Unit, 3, &a, &mut x).unwrap();
        assert_eq!(x, [14.0, 14.0, 3.0]);
    }

    #[test]
    fn trsv_then_trmv_round_trip() {
        // For non-unit upper triangular U, trsv solves Ux=b, trmv computes U·y.
        // Solve U·z = b then verify U·z ≈ b.
        let u = [
            2.0_f64, 1.0, 1.0,
            0.0,     3.0, 2.0,
            0.0,     0.0, 4.0,
        ];
        let b = [11.0_f64, 13.0, 8.0];
        let mut z = b;
        trsv(Uplo::Upper, Trans::No, Diag::NonUnit, 3, &u, &mut z).unwrap();

        let mut bp = z;
        trmv(Uplo::Upper, Trans::No, Diag::NonUnit, 3, &u, &mut bp).unwrap();
        for (got, orig) in bp.iter().zip(b.iter()) {
            assert!((got - orig).abs() < 1e-10);
        }
    }

    // --- Level 2: symv / syr / syr2 / ger -------------------------------

    #[test]
    fn symv_f64() {
        // A = [[1, 2],[2, 3]] symmetric, x = [1, 1] → A·x = [3, 5]
        // Storing only upper triangle works because Uplo::Upper is asked.
        let a = [1.0_f64, 2.0, 0.0, 3.0]; // upper-triangle: [1, 2; -, 3]
        let x = [1.0_f64, 1.0];
        let mut y = [0.0_f64; 2];
        symv(Uplo::Upper, 2, 1.0, &a, &x, 0.0, &mut y).unwrap();
        assert_eq!(y, [3.0, 5.0]);
    }

    #[test]
    fn syr_rank1_update() {
        // A = 0; x = [1, 2]; α=1 → A := x·xᵀ = [[1, 2],[2, 4]]
        // Storing upper triangle: A[0,0]=1, A[0,1]=2, A[1,1]=4.
        let mut a = [0.0_f64; 4];
        let x = [1.0_f64, 2.0];
        syr(Uplo::Upper, 2, 1.0, &x, &mut a).unwrap();
        assert_eq!(a[0], 1.0); // (0,0)
        assert_eq!(a[1], 2.0); // (0,1)
        assert_eq!(a[3], 4.0); // (1,1)
    }

    #[test]
    fn syr2_symmetric_rank2_update() {
        // A=0, x=[1,0], y=[0,1], α=1 → A = x·yᵀ + y·xᵀ = [[0,1],[1,0]]
        let mut a = [0.0_f64; 4];
        let x = [1.0_f64, 0.0];
        let y = [0.0_f64, 1.0];
        syr2(Uplo::Upper, 2, 1.0, &x, &y, &mut a).unwrap();
        assert_eq!(a[1], 1.0); // upper triangle (0,1)
    }

    #[test]
    fn ger_rank1_general() {
        // A = 0 (2×3), x=[1,2], y=[3,4,5], α=1 → A = x·yᵀ
        // = [[3,4,5],[6,8,10]]
        let mut a = [0.0_f64; 6];
        let x = [1.0_f64, 2.0];
        let y = [3.0_f64, 4.0, 5.0];
        ger(2, 3, 1.0, &x, &y, &mut a).unwrap();
        assert_eq!(a, [3.0, 4.0, 5.0, 6.0, 8.0, 10.0]);
    }

    // --- Level 2: hemv / her / her2 / geru / gerc ----------------------

    #[test]
    fn hemv_complex64() {
        // A = [[2, 1+i], [1-i, 3]] is Hermitian. x = [1, 1] → A·x = [3+i, 4-i]
        // Stored upper-triangle row-major (zero out the unused triangle):
        let two = Complex64::new(2.0, 0.0);
        let three = Complex64::new(3.0, 0.0);
        let a01 = Complex64::new(1.0, 1.0);
        let z = Complex64::ZERO;
        let a = [two, a01, z, three];
        let x = [Complex64::ONE, Complex64::ONE];
        let mut y = [z; 2];
        hemv(Uplo::Upper, 2, Complex64::ONE, &a, &x, z, &mut y).unwrap();
        assert_eq!(y[0], Complex64::new(3.0, 1.0));
        assert_eq!(y[1], Complex64::new(4.0, -1.0));
    }

    #[test]
    fn her_rank1_complex() {
        // A=0; x = [1, i]; α=1 → A := x·xᴴ
        // x·xᴴ = [[1,  -i],[i, 1]] (Hermitian)
        // Upper triangle (row-major): A[0,0]=1, A[0,1]=-i, A[1,1]=1.
        let mut a = [Complex64::ZERO; 4];
        let x = [Complex64::ONE, Complex64::I];
        her(Uplo::Upper, 2, 1.0, &x, &mut a).unwrap();
        assert_eq!(a[0], Complex64::ONE);
        assert_eq!(a[1], Complex64::new(0.0, -1.0));
        assert_eq!(a[3], Complex64::ONE);
    }

    #[test]
    fn geru_unconjugated() {
        // A=0, x=[1+i, 0], y=[1, 0], α=1, m=n=2
        // A = x·yᵀ = [[1+i, 0],[0, 0]]
        let mut a = [Complex64::ZERO; 4];
        let x = [Complex64::new(1.0, 1.0), Complex64::ZERO];
        let y = [Complex64::ONE, Complex64::ZERO];
        geru(2, 2, Complex64::ONE, &x, &y, &mut a).unwrap();
        assert_eq!(a[0], Complex64::new(1.0, 1.0));
        assert_eq!(a[1], Complex64::ZERO);
    }

    #[test]
    fn gerc_conjugated() {
        // A=0, x=[1+i], y=[1+i], m=n=1, α=1
        // gerc: A = x·yᴴ = (1+i)·conj(1+i) = (1+i)(1-i) = 1 + 1 = 2
        let mut a = [Complex64::ZERO; 1];
        let x = [Complex64::new(1.0, 1.0)];
        let y = [Complex64::new(1.0, 1.0)];
        gerc(1, 1, Complex64::ONE, &x, &y, &mut a).unwrap();
        assert_eq!(a[0], Complex64::new(2.0, 0.0));
    }

    // --- Level 3: symm / syrk / syr2k / trmm / trsm ---------------------

    #[test]
    fn symm_left_f64() {
        // A symmetric 2×2 = [[1,2],[2,3]], B = [[1, 0],[0, 1]] = I_2
        // C = A·B = A.
        let a = [1.0_f64, 2.0, 0.0, 3.0]; // upper triangle
        let b = [1.0_f64, 0.0, 0.0, 1.0];
        let mut c = [0.0_f64; 4];
        symm(Side::Left, Uplo::Upper, 2, 2, 1.0, &a, &b, 0.0, &mut c).unwrap();
        // C should be symmetric reconstruction of A.
        assert_eq!(c[0], 1.0);
        assert_eq!(c[1], 2.0);
        assert_eq!(c[2], 2.0);
        assert_eq!(c[3], 3.0);
    }

    #[test]
    fn syrk_rank_k_update() {
        // A = [[1, 2],[3, 4]] (2×2), C = 0
        // syrk: C := A · Aᵀ. A·Aᵀ = [[1+4, 3+8],[3+8, 9+16]] = [[5, 11],[11, 25]]
        // Only upper triangle of C is updated.
        let a = [1.0_f64, 2.0, 3.0, 4.0];
        let mut c = [0.0_f64; 4];
        syrk(Uplo::Upper, Trans::No, 2, 2, 1.0, &a, 0.0, &mut c).unwrap();
        assert_eq!(c[0], 5.0);  // (0,0)
        assert_eq!(c[1], 11.0); // (0,1)
        assert_eq!(c[3], 25.0); // (1,1)
    }

    #[test]
    fn syr2k_rank_2k_update() {
        // A = [[1, 0]], B = [[0, 1]], n=1, k=2, α=1, β=0
        // syr2k: C = A·Bᵀ + B·Aᵀ = (1·0 + 0·1) + (0·1 + 1·0) = 0
        let a = [1.0_f64, 0.0];
        let b = [0.0_f64, 1.0];
        let mut c = [99.0_f64];
        syr2k(Uplo::Upper, Trans::No, 1, 2, 1.0, &a, &b, 0.0, &mut c).unwrap();
        assert_eq!(c[0], 0.0);
    }

    #[test]
    fn trmm_left_lower_unit() {
        // L = [[1, 0],[2, 1]] (lower-unit triangular row-major), B = [[3],[4]]
        // trmm Side::Left: B := L·B = [[3],[2*3+4]] = [[3],[10]]
        let l = [0.0_f64, 0.0, 2.0, 0.0]; // diag implicit
        let mut b = [3.0_f64, 4.0];
        trmm(Side::Left, Uplo::Lower, Trans::No, Diag::Unit, 2, 1, 1.0, &l, &mut b).unwrap();
        assert_eq!(b, [3.0, 10.0]);
    }

    #[test]
    fn trsm_solves_unit_lower() {
        // L · X = B, L = [[1, 0],[2, 1]], B = [[3],[10]] → X = [[3],[4]]
        let l = [0.0_f64, 0.0, 2.0, 0.0];
        let mut b = [3.0_f64, 10.0];
        trsm(Side::Left, Uplo::Lower, Trans::No, Diag::Unit, 2, 1, 1.0, &l, &mut b).unwrap();
        assert_eq!(b, [3.0, 4.0]);
    }

    #[test]
    fn hemm_complex64() {
        // A = [[2, 0],[0, 2]] Hermitian (real entries), B = I_2 = [[1,0],[0,1]]
        // C = α·A·B = 2·I_2.
        let a = [
            Complex64::new(2.0, 0.0), Complex64::ZERO,
            Complex64::ZERO, Complex64::new(2.0, 0.0),
        ];
        let b = [Complex64::ONE, Complex64::ZERO, Complex64::ZERO, Complex64::ONE];
        let mut c = [Complex64::ZERO; 4];
        hemm(Side::Left, Uplo::Upper, 2, 2, Complex64::ONE, &a, &b, Complex64::ZERO, &mut c).unwrap();
        assert_eq!(c[0], Complex64::new(2.0, 0.0));
        assert_eq!(c[3], Complex64::new(2.0, 0.0));
    }

    #[test]
    fn herk_complex64() {
        // A = [[1, i]], n=1, k=2, real α=1, β=0
        // herk Trans::No: C := A·Aᴴ = 1·1 + i·conj(i) = 1 + 1 = 2
        let a = [Complex64::ONE, Complex64::I];
        let mut c = [Complex64::ZERO; 1];
        herk(Uplo::Upper, Trans::No, 1, 2, 1.0_f64, &a, 0.0_f64, &mut c).unwrap();
        // c[0] should be real-valued 2.
        assert!((c[0].re - 2.0).abs() < 1e-12, "got {:?}", c[0]);
        assert!(c[0].im.abs() < 1e-12);
    }

    #[test]
    fn herk_rejects_trans_t() {
        let a = [Complex64::ONE];
        let mut c = [Complex64::ZERO; 1];
        let err = herk(Uplo::Upper, Trans::T, 1, 1, 1.0, &a, 0.0, &mut c).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)));
    }

    #[test]
    fn her2k_complex64() {
        // A = [[1+0i]], B = [[1+0i]], n=1, k=1, α=1, β=0
        // her2k: C = α·A·Bᴴ + conj(α)·B·Aᴴ = 1·1 + 1·1 = 2
        let a = [Complex64::ONE];
        let b = [Complex64::ONE];
        let mut c = [Complex64::ZERO; 1];
        her2k(Uplo::Upper, Trans::No, 1, 1, Complex64::ONE, &a, &b, 0.0, &mut c).unwrap();
        assert!((c[0].re - 2.0).abs() < 1e-12);
    }
}
