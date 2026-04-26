//! Safe wrappers for AOCL-LAPACK (libFLAME) via the LAPACKE C interface.
//!
//! Comprehensive linear-solver coverage across `f32`, `f64`, `Complex32`,
//! and `Complex64`:
//! - General LU: `gesv`, `getrf`, `getrs`, `getri`.
//! - Cholesky (positive-definite): `posv`, `potrf`, `potrs`, `potri`.
//! - Symmetric indefinite (Bunch-Kaufman): `sysv`, `sytrf`, `sytrs`.
//! - Hermitian indefinite (complex only): `hesv`, `hetrf`, `hetrs`.
//! - General banded: `gbsv`, `gbtrf`, `gbtrs`.
//! - Tridiagonal: `gtsv`, `ptsv`.
//! - Least squares: `gels`.
//!
//! Plus factorizations:
//! - QR: `geqrf`. LQ: `gelqf`.

#![warn(missing_debug_implementations)]
#![cfg_attr(docsrs, feature(doc_cfg))]

use aocl_lapack_sys as sys;
pub use aocl_error::{Error, Result};
pub use aocl_types::{Complex32, Complex64, Layout, Trans, Uplo};
use aocl_types::sealed::Sealed;

use std::os::raw::c_char;

const ROW_MAJOR: i32 = 101;
const COL_MAJOR: i32 = 102;

fn layout_raw(l: Layout) -> i32 {
    match l {
        Layout::RowMajor => ROW_MAJOR,
        Layout::ColMajor => COL_MAJOR,
    }
}

fn uplo_char(u: Uplo) -> c_char {
    match u {
        Uplo::Upper => b'U' as c_char,
        Uplo::Lower => b'L' as c_char,
    }
}

fn trans_char(t: Trans) -> c_char {
    match t {
        Trans::No => b'N' as c_char,
        Trans::T => b'T' as c_char,
        Trans::C => b'C' as c_char,
    }
}

fn map_info(component: &'static str, info: i32) -> Result<()> {
    use std::cmp::Ordering;
    match info.cmp(&0) {
        Ordering::Equal => Ok(()),
        Ordering::Less => Err(Error::Status {
            component,
            code: info as i64,
            message: format!("argument {} had an illegal value", -info),
        }),
        Ordering::Greater => Err(Error::Status {
            component,
            code: info as i64,
            message: format!("matrix singular or factor failed at index {info}"),
        }),
    }
}

fn check_matrix(
    name: &str,
    layout: Layout,
    rows: usize,
    cols: usize,
    ld: usize,
    slice_len: usize,
) -> Result<()> {
    let (lead, trail) = match layout {
        Layout::RowMajor => (rows, cols),
        Layout::ColMajor => (cols, rows),
    };
    if ld < trail.max(1) {
        return Err(Error::InvalidArgument(format!(
            "{name}: leading dimension {ld} < {trail}"
        )));
    }
    if rows == 0 || cols == 0 {
        return Ok(());
    }
    let needed = (lead - 1) * ld + trail;
    if slice_len < needed {
        return Err(Error::InvalidArgument(format!(
            "{name}: slice length {slice_len} too small (need {needed})"
        )));
    }
    Ok(())
}

// =========================================================================
//   Trait definitions
// =========================================================================

/// Element type usable with the wrapped LAPACK routines.
pub trait Scalar: Copy + Sized + Sealed {
    /// Underlying real precision (f32 for f32 / Complex32, f64 for
    /// f64 / Complex64). Used by ptsv's real-diagonal `d` array.
    type Real: Copy + Sized + Sealed;

    // --- General LU ------------------------------------------------------

    /// Solve `A · X = B` for general `n × n` `A` (LU + back-substitution).
    #[allow(clippy::too_many_arguments)]
    fn gesv(layout: Layout, n: usize, nrhs: usize,
            a: &mut [Self], lda: usize, ipiv: &mut [i32],
            b: &mut [Self], ldb: usize) -> Result<()>;

    /// LU factorization with partial pivoting.
    fn getrf(layout: Layout, m: usize, n: usize,
             a: &mut [Self], lda: usize, ipiv: &mut [i32]) -> Result<()>;

    /// Back-substitute using a prior LU factorization.
    #[allow(clippy::too_many_arguments)]
    fn getrs(layout: Layout, trans: Trans, n: usize, nrhs: usize,
             a: &[Self], lda: usize, ipiv: &[i32],
             b: &mut [Self], ldb: usize) -> Result<()>;

    /// Inverse from a prior LU factorization.
    fn getri(layout: Layout, n: usize,
             a: &mut [Self], lda: usize, ipiv: &[i32]) -> Result<()>;

    // --- Cholesky --------------------------------------------------------

    /// Solve `A · X = B` for symmetric/Hermitian positive-definite `A`.
    #[allow(clippy::too_many_arguments)]
    fn posv(layout: Layout, uplo: Uplo, n: usize, nrhs: usize,
            a: &mut [Self], lda: usize,
            b: &mut [Self], ldb: usize) -> Result<()>;

    /// Cholesky factorization.
    fn potrf(layout: Layout, uplo: Uplo, n: usize,
             a: &mut [Self], lda: usize) -> Result<()>;

    /// Back-substitute using a prior Cholesky factorization.
    #[allow(clippy::too_many_arguments)]
    fn potrs(layout: Layout, uplo: Uplo, n: usize, nrhs: usize,
             a: &[Self], lda: usize,
             b: &mut [Self], ldb: usize) -> Result<()>;

    /// Inverse from a prior Cholesky factorization.
    fn potri(layout: Layout, uplo: Uplo, n: usize,
             a: &mut [Self], lda: usize) -> Result<()>;

    // --- Symmetric indefinite (Bunch-Kaufman) ----------------------------

    /// Solve `A · X = B` for symmetric indefinite `A` (real or complex
    /// **symmetric**, NOT Hermitian — for complex Hermitian use
    /// [`ComplexScalar::hesv`]).
    #[allow(clippy::too_many_arguments)]
    fn sysv(layout: Layout, uplo: Uplo, n: usize, nrhs: usize,
            a: &mut [Self], lda: usize, ipiv: &mut [i32],
            b: &mut [Self], ldb: usize) -> Result<()>;

    /// Bunch-Kaufman factorization.
    fn sytrf(layout: Layout, uplo: Uplo, n: usize,
             a: &mut [Self], lda: usize, ipiv: &mut [i32]) -> Result<()>;

    /// Back-substitute using a prior Bunch-Kaufman factorization.
    #[allow(clippy::too_many_arguments)]
    fn sytrs(layout: Layout, uplo: Uplo, n: usize, nrhs: usize,
             a: &[Self], lda: usize, ipiv: &[i32],
             b: &mut [Self], ldb: usize) -> Result<()>;

    // --- Banded ----------------------------------------------------------

    /// Solve `A · X = B` for general banded `A`.
    #[allow(clippy::too_many_arguments)]
    fn gbsv(layout: Layout, n: usize, kl: usize, ku: usize, nrhs: usize,
            ab: &mut [Self], ldab: usize, ipiv: &mut [i32],
            b: &mut [Self], ldb: usize) -> Result<()>;

    /// Banded LU factorization.
    #[allow(clippy::too_many_arguments)]
    fn gbtrf(layout: Layout, m: usize, n: usize, kl: usize, ku: usize,
             ab: &mut [Self], ldab: usize, ipiv: &mut [i32]) -> Result<()>;

    /// Back-substitute using a prior banded LU factorization.
    #[allow(clippy::too_many_arguments)]
    fn gbtrs(layout: Layout, trans: Trans, n: usize,
             kl: usize, ku: usize, nrhs: usize,
             ab: &[Self], ldab: usize, ipiv: &[i32],
             b: &mut [Self], ldb: usize) -> Result<()>;

    // --- Tridiagonal -----------------------------------------------------

    /// Solve `A · X = B` for general tridiagonal `A`.
    #[allow(clippy::too_many_arguments)]
    fn gtsv(layout: Layout, n: usize, nrhs: usize,
            dl: &mut [Self], d: &mut [Self], du: &mut [Self],
            b: &mut [Self], ldb: usize) -> Result<()>;

    /// Solve `A · X = B` for symmetric/Hermitian positive-definite
    /// tridiagonal `A`. The main diagonal `d` is **real**; `e` is the
    /// (sub-/super-)diagonal whose precision matches `Self`.
    #[allow(clippy::too_many_arguments)]
    fn ptsv(layout: Layout, n: usize, nrhs: usize,
            d: &mut [Self::Real], e: &mut [Self],
            b: &mut [Self], ldb: usize) -> Result<()>;

    // --- Least squares ---------------------------------------------------

    /// Solve `min || A·X − B ||₂` (or transposed equivalents).
    #[allow(clippy::too_many_arguments)]
    fn gels(layout: Layout, trans: Trans, m: usize, n: usize, nrhs: usize,
            a: &mut [Self], lda: usize,
            b: &mut [Self], ldb: usize) -> Result<()>;

    // --- QR / LQ ---------------------------------------------------------

    /// QR factorization.
    fn geqrf(layout: Layout, m: usize, n: usize,
             a: &mut [Self], lda: usize, tau: &mut [Self]) -> Result<()>;

    /// LQ factorization.
    fn gelqf(layout: Layout, m: usize, n: usize,
             a: &mut [Self], lda: usize, tau: &mut [Self]) -> Result<()>;
}

/// Operations defined only for complex precisions.
pub trait ComplexScalar: Scalar {
    /// Hermitian indefinite solve.
    #[allow(clippy::too_many_arguments)]
    fn hesv(layout: Layout, uplo: Uplo, n: usize, nrhs: usize,
            a: &mut [Self], lda: usize, ipiv: &mut [i32],
            b: &mut [Self], ldb: usize) -> Result<()>;

    /// Hermitian Bunch-Kaufman factorization.
    fn hetrf(layout: Layout, uplo: Uplo, n: usize,
             a: &mut [Self], lda: usize, ipiv: &mut [i32]) -> Result<()>;

    /// Back-substitute using a Hermitian Bunch-Kaufman factorization.
    #[allow(clippy::too_many_arguments)]
    fn hetrs(layout: Layout, uplo: Uplo, n: usize, nrhs: usize,
             a: &[Self], lda: usize, ipiv: &[i32],
             b: &mut [Self], ldb: usize) -> Result<()>;
}

// =========================================================================
//   Real impl (f32, f64)
// =========================================================================

macro_rules! impl_real {
    (
        $t:ty,
        gesv = $gesv:ident, getrf = $getrf:ident, getrs = $getrs:ident, getri = $getri:ident,
        posv = $posv:ident, potrf = $potrf:ident, potrs = $potrs:ident, potri = $potri:ident,
        sysv = $sysv:ident, sytrf = $sytrf:ident, sytrs = $sytrs:ident,
        gbsv = $gbsv:ident, gbtrf = $gbtrf:ident, gbtrs = $gbtrs:ident,
        gtsv = $gtsv:ident, ptsv = $ptsv:ident,
        gels = $gels:ident, geqrf = $geqrf:ident, gelqf = $gelqf:ident
    ) => {
        impl Scalar for $t {
            type Real = $t;

            fn gesv(layout: Layout, n: usize, nrhs: usize,
                    a: &mut [Self], lda: usize, ipiv: &mut [i32],
                    b: &mut [Self], ldb: usize) -> Result<()> {
                check_matrix("gesv: A", layout, n, n, lda, a.len())?;
                check_matrix("gesv: B", layout, n, nrhs, ldb, b.len())?;
                if ipiv.len() < n {
                    return Err(Error::InvalidArgument(format!(
                        "gesv: ipiv length {} < n={n}", ipiv.len()
                    )));
                }
                let info = unsafe {
                    sys::$gesv(layout_raw(layout), n as i32, nrhs as i32,
                               a.as_mut_ptr(), lda as i32, ipiv.as_mut_ptr(),
                               b.as_mut_ptr(), ldb as i32)
                };
                map_info("lapack", info)
            }

            fn getrf(layout: Layout, m: usize, n: usize,
                     a: &mut [Self], lda: usize, ipiv: &mut [i32]) -> Result<()> {
                check_matrix("getrf: A", layout, m, n, lda, a.len())?;
                if ipiv.len() < m.min(n) {
                    return Err(Error::InvalidArgument(format!(
                        "getrf: ipiv length {} < min(m,n)={}", ipiv.len(), m.min(n)
                    )));
                }
                let info = unsafe {
                    sys::$getrf(layout_raw(layout), m as i32, n as i32,
                                a.as_mut_ptr(), lda as i32, ipiv.as_mut_ptr())
                };
                map_info("lapack", info)
            }

            fn getrs(layout: Layout, trans: Trans, n: usize, nrhs: usize,
                     a: &[Self], lda: usize, ipiv: &[i32],
                     b: &mut [Self], ldb: usize) -> Result<()> {
                check_matrix("getrs: A", layout, n, n, lda, a.len())?;
                check_matrix("getrs: B", layout, n, nrhs, ldb, b.len())?;
                let info = unsafe {
                    sys::$getrs(layout_raw(layout), trans_char(trans),
                                n as i32, nrhs as i32,
                                a.as_ptr(), lda as i32, ipiv.as_ptr(),
                                b.as_mut_ptr(), ldb as i32)
                };
                map_info("lapack", info)
            }

            fn getri(layout: Layout, n: usize,
                     a: &mut [Self], lda: usize, ipiv: &[i32]) -> Result<()> {
                check_matrix("getri: A", layout, n, n, lda, a.len())?;
                let info = unsafe {
                    sys::$getri(layout_raw(layout), n as i32,
                                a.as_mut_ptr(), lda as i32, ipiv.as_ptr())
                };
                map_info("lapack", info)
            }

            fn posv(layout: Layout, uplo: Uplo, n: usize, nrhs: usize,
                    a: &mut [Self], lda: usize,
                    b: &mut [Self], ldb: usize) -> Result<()> {
                check_matrix("posv: A", layout, n, n, lda, a.len())?;
                check_matrix("posv: B", layout, n, nrhs, ldb, b.len())?;
                let info = unsafe {
                    sys::$posv(layout_raw(layout), uplo_char(uplo),
                               n as i32, nrhs as i32,
                               a.as_mut_ptr(), lda as i32,
                               b.as_mut_ptr(), ldb as i32)
                };
                map_info("lapack", info)
            }

            fn potrf(layout: Layout, uplo: Uplo, n: usize,
                     a: &mut [Self], lda: usize) -> Result<()> {
                check_matrix("potrf: A", layout, n, n, lda, a.len())?;
                let info = unsafe {
                    sys::$potrf(layout_raw(layout), uplo_char(uplo),
                                n as i32, a.as_mut_ptr(), lda as i32)
                };
                map_info("lapack", info)
            }

            fn potrs(layout: Layout, uplo: Uplo, n: usize, nrhs: usize,
                     a: &[Self], lda: usize,
                     b: &mut [Self], ldb: usize) -> Result<()> {
                check_matrix("potrs: A", layout, n, n, lda, a.len())?;
                check_matrix("potrs: B", layout, n, nrhs, ldb, b.len())?;
                let info = unsafe {
                    sys::$potrs(layout_raw(layout), uplo_char(uplo),
                                n as i32, nrhs as i32,
                                a.as_ptr(), lda as i32,
                                b.as_mut_ptr(), ldb as i32)
                };
                map_info("lapack", info)
            }

            fn potri(layout: Layout, uplo: Uplo, n: usize,
                     a: &mut [Self], lda: usize) -> Result<()> {
                check_matrix("potri: A", layout, n, n, lda, a.len())?;
                let info = unsafe {
                    sys::$potri(layout_raw(layout), uplo_char(uplo),
                                n as i32, a.as_mut_ptr(), lda as i32)
                };
                map_info("lapack", info)
            }

            fn sysv(layout: Layout, uplo: Uplo, n: usize, nrhs: usize,
                    a: &mut [Self], lda: usize, ipiv: &mut [i32],
                    b: &mut [Self], ldb: usize) -> Result<()> {
                check_matrix("sysv: A", layout, n, n, lda, a.len())?;
                check_matrix("sysv: B", layout, n, nrhs, ldb, b.len())?;
                let info = unsafe {
                    sys::$sysv(layout_raw(layout), uplo_char(uplo),
                               n as i32, nrhs as i32,
                               a.as_mut_ptr(), lda as i32, ipiv.as_mut_ptr(),
                               b.as_mut_ptr(), ldb as i32)
                };
                map_info("lapack", info)
            }

            fn sytrf(layout: Layout, uplo: Uplo, n: usize,
                     a: &mut [Self], lda: usize, ipiv: &mut [i32]) -> Result<()> {
                let info = unsafe {
                    sys::$sytrf(layout_raw(layout), uplo_char(uplo),
                                n as i32, a.as_mut_ptr(), lda as i32, ipiv.as_mut_ptr())
                };
                map_info("lapack", info)
            }

            fn sytrs(layout: Layout, uplo: Uplo, n: usize, nrhs: usize,
                     a: &[Self], lda: usize, ipiv: &[i32],
                     b: &mut [Self], ldb: usize) -> Result<()> {
                let info = unsafe {
                    sys::$sytrs(layout_raw(layout), uplo_char(uplo),
                                n as i32, nrhs as i32,
                                a.as_ptr(), lda as i32, ipiv.as_ptr(),
                                b.as_mut_ptr(), ldb as i32)
                };
                map_info("lapack", info)
            }

            fn gbsv(layout: Layout, n: usize, kl: usize, ku: usize, nrhs: usize,
                    ab: &mut [Self], ldab: usize, ipiv: &mut [i32],
                    b: &mut [Self], ldb: usize) -> Result<()> {
                let info = unsafe {
                    sys::$gbsv(layout_raw(layout), n as i32,
                               kl as i32, ku as i32, nrhs as i32,
                               ab.as_mut_ptr(), ldab as i32, ipiv.as_mut_ptr(),
                               b.as_mut_ptr(), ldb as i32)
                };
                map_info("lapack", info)
            }

            fn gbtrf(layout: Layout, m: usize, n: usize, kl: usize, ku: usize,
                     ab: &mut [Self], ldab: usize, ipiv: &mut [i32]) -> Result<()> {
                let info = unsafe {
                    sys::$gbtrf(layout_raw(layout), m as i32, n as i32,
                                kl as i32, ku as i32,
                                ab.as_mut_ptr(), ldab as i32, ipiv.as_mut_ptr())
                };
                map_info("lapack", info)
            }

            fn gbtrs(layout: Layout, trans: Trans, n: usize,
                     kl: usize, ku: usize, nrhs: usize,
                     ab: &[Self], ldab: usize, ipiv: &[i32],
                     b: &mut [Self], ldb: usize) -> Result<()> {
                let info = unsafe {
                    sys::$gbtrs(layout_raw(layout), trans_char(trans),
                                n as i32, kl as i32, ku as i32, nrhs as i32,
                                ab.as_ptr(), ldab as i32, ipiv.as_ptr(),
                                b.as_mut_ptr(), ldb as i32)
                };
                map_info("lapack", info)
            }

            fn gtsv(layout: Layout, n: usize, nrhs: usize,
                    dl: &mut [Self], d: &mut [Self], du: &mut [Self],
                    b: &mut [Self], ldb: usize) -> Result<()> {
                if n > 0 {
                    if d.len() < n {
                        return Err(Error::InvalidArgument(format!(
                            "gtsv: d length {} < n={n}", d.len()
                        )));
                    }
                    if dl.len() < n - 1 || du.len() < n - 1 {
                        return Err(Error::InvalidArgument(format!(
                            "gtsv: dl/du length must be at least n-1 = {}", n - 1
                        )));
                    }
                }
                let info = unsafe {
                    sys::$gtsv(layout_raw(layout), n as i32, nrhs as i32,
                               dl.as_mut_ptr(), d.as_mut_ptr(), du.as_mut_ptr(),
                               b.as_mut_ptr(), ldb as i32)
                };
                map_info("lapack", info)
            }

            fn ptsv(layout: Layout, n: usize, nrhs: usize,
                    d: &mut [Self::Real], e: &mut [Self],
                    b: &mut [Self], ldb: usize) -> Result<()> {
                let info = unsafe {
                    sys::$ptsv(layout_raw(layout), n as i32, nrhs as i32,
                               d.as_mut_ptr(), e.as_mut_ptr(),
                               b.as_mut_ptr(), ldb as i32)
                };
                map_info("lapack", info)
            }

            fn gels(layout: Layout, trans: Trans, m: usize, n: usize, nrhs: usize,
                    a: &mut [Self], lda: usize,
                    b: &mut [Self], ldb: usize) -> Result<()> {
                let info = unsafe {
                    sys::$gels(layout_raw(layout), trans_char(trans),
                               m as i32, n as i32, nrhs as i32,
                               a.as_mut_ptr(), lda as i32,
                               b.as_mut_ptr(), ldb as i32)
                };
                map_info("lapack", info)
            }

            fn geqrf(layout: Layout, m: usize, n: usize,
                     a: &mut [Self], lda: usize, tau: &mut [Self]) -> Result<()> {
                check_matrix("geqrf: A", layout, m, n, lda, a.len())?;
                if tau.len() < m.min(n) {
                    return Err(Error::InvalidArgument(format!(
                        "geqrf: tau length {} < min(m,n)={}", tau.len(), m.min(n)
                    )));
                }
                let info = unsafe {
                    sys::$geqrf(layout_raw(layout), m as i32, n as i32,
                                a.as_mut_ptr(), lda as i32, tau.as_mut_ptr())
                };
                map_info("lapack", info)
            }

            fn gelqf(layout: Layout, m: usize, n: usize,
                     a: &mut [Self], lda: usize, tau: &mut [Self]) -> Result<()> {
                check_matrix("gelqf: A", layout, m, n, lda, a.len())?;
                if tau.len() < m.min(n) {
                    return Err(Error::InvalidArgument(format!(
                        "gelqf: tau length {} < min(m,n)={}", tau.len(), m.min(n)
                    )));
                }
                let info = unsafe {
                    sys::$gelqf(layout_raw(layout), m as i32, n as i32,
                                a.as_mut_ptr(), lda as i32, tau.as_mut_ptr())
                };
                map_info("lapack", info)
            }
        }
    };
}

impl_real!(
    f32,
    gesv = LAPACKE_sgesv, getrf = LAPACKE_sgetrf, getrs = LAPACKE_sgetrs, getri = LAPACKE_sgetri,
    posv = LAPACKE_sposv, potrf = LAPACKE_spotrf, potrs = LAPACKE_spotrs, potri = LAPACKE_spotri,
    sysv = LAPACKE_ssysv, sytrf = LAPACKE_ssytrf, sytrs = LAPACKE_ssytrs,
    gbsv = LAPACKE_sgbsv, gbtrf = LAPACKE_sgbtrf, gbtrs = LAPACKE_sgbtrs,
    gtsv = LAPACKE_sgtsv, ptsv = LAPACKE_sptsv,
    gels = LAPACKE_sgels, geqrf = LAPACKE_sgeqrf, gelqf = LAPACKE_sgelqf
);
impl_real!(
    f64,
    gesv = LAPACKE_dgesv, getrf = LAPACKE_dgetrf, getrs = LAPACKE_dgetrs, getri = LAPACKE_dgetri,
    posv = LAPACKE_dposv, potrf = LAPACKE_dpotrf, potrs = LAPACKE_dpotrs, potri = LAPACKE_dpotri,
    sysv = LAPACKE_dsysv, sytrf = LAPACKE_dsytrf, sytrs = LAPACKE_dsytrs,
    gbsv = LAPACKE_dgbsv, gbtrf = LAPACKE_dgbtrf, gbtrs = LAPACKE_dgbtrs,
    gtsv = LAPACKE_dgtsv, ptsv = LAPACKE_dptsv,
    gels = LAPACKE_dgels, geqrf = LAPACKE_dgeqrf, gelqf = LAPACKE_dgelqf
);

// =========================================================================
//   Complex impl (Complex32, Complex64) — repr(C) compatible with
//   __BindgenComplex<f32/f64> at the LAPACKE FFI boundary.
// =========================================================================

type CC32 = sys::__BindgenComplex<f32>;
type CC64 = sys::__BindgenComplex<f64>;

macro_rules! impl_complex {
    (
        $t:ty, $real:ty, $cc:ty,
        gesv = $gesv:ident, getrf = $getrf:ident, getrs = $getrs:ident, getri = $getri:ident,
        posv = $posv:ident, potrf = $potrf:ident, potrs = $potrs:ident, potri = $potri:ident,
        sysv = $sysv:ident, sytrf = $sytrf:ident, sytrs = $sytrs:ident,
        hesv = $hesv:ident, hetrf = $hetrf:ident, hetrs = $hetrs:ident,
        gbsv = $gbsv:ident, gbtrf = $gbtrf:ident, gbtrs = $gbtrs:ident,
        gtsv = $gtsv:ident, ptsv = $ptsv:ident,
        gels = $gels:ident, geqrf = $geqrf:ident, gelqf = $gelqf:ident
    ) => {
        impl Scalar for $t {
            type Real = $real;

            fn gesv(layout: Layout, n: usize, nrhs: usize,
                    a: &mut [Self], lda: usize, ipiv: &mut [i32],
                    b: &mut [Self], ldb: usize) -> Result<()> {
                check_matrix("gesv: A", layout, n, n, lda, a.len())?;
                check_matrix("gesv: B", layout, n, nrhs, ldb, b.len())?;
                if ipiv.len() < n {
                    return Err(Error::InvalidArgument(format!(
                        "gesv: ipiv length {} < n={n}", ipiv.len()
                    )));
                }
                let info = unsafe {
                    sys::$gesv(layout_raw(layout), n as i32, nrhs as i32,
                               a.as_mut_ptr() as *mut $cc, lda as i32, ipiv.as_mut_ptr(),
                               b.as_mut_ptr() as *mut $cc, ldb as i32)
                };
                map_info("lapack", info)
            }

            fn getrf(layout: Layout, m: usize, n: usize,
                     a: &mut [Self], lda: usize, ipiv: &mut [i32]) -> Result<()> {
                check_matrix("getrf: A", layout, m, n, lda, a.len())?;
                let info = unsafe {
                    sys::$getrf(layout_raw(layout), m as i32, n as i32,
                                a.as_mut_ptr() as *mut $cc, lda as i32, ipiv.as_mut_ptr())
                };
                map_info("lapack", info)
            }

            fn getrs(layout: Layout, trans: Trans, n: usize, nrhs: usize,
                     a: &[Self], lda: usize, ipiv: &[i32],
                     b: &mut [Self], ldb: usize) -> Result<()> {
                let info = unsafe {
                    sys::$getrs(layout_raw(layout), trans_char(trans),
                                n as i32, nrhs as i32,
                                a.as_ptr() as *const $cc, lda as i32, ipiv.as_ptr(),
                                b.as_mut_ptr() as *mut $cc, ldb as i32)
                };
                map_info("lapack", info)
            }

            fn getri(layout: Layout, n: usize,
                     a: &mut [Self], lda: usize, ipiv: &[i32]) -> Result<()> {
                let info = unsafe {
                    sys::$getri(layout_raw(layout), n as i32,
                                a.as_mut_ptr() as *mut $cc, lda as i32, ipiv.as_ptr())
                };
                map_info("lapack", info)
            }

            fn posv(layout: Layout, uplo: Uplo, n: usize, nrhs: usize,
                    a: &mut [Self], lda: usize,
                    b: &mut [Self], ldb: usize) -> Result<()> {
                let info = unsafe {
                    sys::$posv(layout_raw(layout), uplo_char(uplo),
                               n as i32, nrhs as i32,
                               a.as_mut_ptr() as *mut $cc, lda as i32,
                               b.as_mut_ptr() as *mut $cc, ldb as i32)
                };
                map_info("lapack", info)
            }

            fn potrf(layout: Layout, uplo: Uplo, n: usize,
                     a: &mut [Self], lda: usize) -> Result<()> {
                let info = unsafe {
                    sys::$potrf(layout_raw(layout), uplo_char(uplo),
                                n as i32, a.as_mut_ptr() as *mut $cc, lda as i32)
                };
                map_info("lapack", info)
            }

            fn potrs(layout: Layout, uplo: Uplo, n: usize, nrhs: usize,
                     a: &[Self], lda: usize,
                     b: &mut [Self], ldb: usize) -> Result<()> {
                let info = unsafe {
                    sys::$potrs(layout_raw(layout), uplo_char(uplo),
                                n as i32, nrhs as i32,
                                a.as_ptr() as *const $cc, lda as i32,
                                b.as_mut_ptr() as *mut $cc, ldb as i32)
                };
                map_info("lapack", info)
            }

            fn potri(layout: Layout, uplo: Uplo, n: usize,
                     a: &mut [Self], lda: usize) -> Result<()> {
                let info = unsafe {
                    sys::$potri(layout_raw(layout), uplo_char(uplo),
                                n as i32, a.as_mut_ptr() as *mut $cc, lda as i32)
                };
                map_info("lapack", info)
            }

            fn sysv(layout: Layout, uplo: Uplo, n: usize, nrhs: usize,
                    a: &mut [Self], lda: usize, ipiv: &mut [i32],
                    b: &mut [Self], ldb: usize) -> Result<()> {
                let info = unsafe {
                    sys::$sysv(layout_raw(layout), uplo_char(uplo),
                               n as i32, nrhs as i32,
                               a.as_mut_ptr() as *mut $cc, lda as i32, ipiv.as_mut_ptr(),
                               b.as_mut_ptr() as *mut $cc, ldb as i32)
                };
                map_info("lapack", info)
            }

            fn sytrf(layout: Layout, uplo: Uplo, n: usize,
                     a: &mut [Self], lda: usize, ipiv: &mut [i32]) -> Result<()> {
                let info = unsafe {
                    sys::$sytrf(layout_raw(layout), uplo_char(uplo),
                                n as i32, a.as_mut_ptr() as *mut $cc, lda as i32,
                                ipiv.as_mut_ptr())
                };
                map_info("lapack", info)
            }

            fn sytrs(layout: Layout, uplo: Uplo, n: usize, nrhs: usize,
                     a: &[Self], lda: usize, ipiv: &[i32],
                     b: &mut [Self], ldb: usize) -> Result<()> {
                let info = unsafe {
                    sys::$sytrs(layout_raw(layout), uplo_char(uplo),
                                n as i32, nrhs as i32,
                                a.as_ptr() as *const $cc, lda as i32, ipiv.as_ptr(),
                                b.as_mut_ptr() as *mut $cc, ldb as i32)
                };
                map_info("lapack", info)
            }

            fn gbsv(layout: Layout, n: usize, kl: usize, ku: usize, nrhs: usize,
                    ab: &mut [Self], ldab: usize, ipiv: &mut [i32],
                    b: &mut [Self], ldb: usize) -> Result<()> {
                let info = unsafe {
                    sys::$gbsv(layout_raw(layout), n as i32,
                               kl as i32, ku as i32, nrhs as i32,
                               ab.as_mut_ptr() as *mut $cc, ldab as i32, ipiv.as_mut_ptr(),
                               b.as_mut_ptr() as *mut $cc, ldb as i32)
                };
                map_info("lapack", info)
            }

            fn gbtrf(layout: Layout, m: usize, n: usize, kl: usize, ku: usize,
                     ab: &mut [Self], ldab: usize, ipiv: &mut [i32]) -> Result<()> {
                let info = unsafe {
                    sys::$gbtrf(layout_raw(layout), m as i32, n as i32,
                                kl as i32, ku as i32,
                                ab.as_mut_ptr() as *mut $cc, ldab as i32, ipiv.as_mut_ptr())
                };
                map_info("lapack", info)
            }

            fn gbtrs(layout: Layout, trans: Trans, n: usize,
                     kl: usize, ku: usize, nrhs: usize,
                     ab: &[Self], ldab: usize, ipiv: &[i32],
                     b: &mut [Self], ldb: usize) -> Result<()> {
                let info = unsafe {
                    sys::$gbtrs(layout_raw(layout), trans_char(trans),
                                n as i32, kl as i32, ku as i32, nrhs as i32,
                                ab.as_ptr() as *const $cc, ldab as i32, ipiv.as_ptr(),
                                b.as_mut_ptr() as *mut $cc, ldb as i32)
                };
                map_info("lapack", info)
            }

            fn gtsv(layout: Layout, n: usize, nrhs: usize,
                    dl: &mut [Self], d: &mut [Self], du: &mut [Self],
                    b: &mut [Self], ldb: usize) -> Result<()> {
                let info = unsafe {
                    sys::$gtsv(layout_raw(layout), n as i32, nrhs as i32,
                               dl.as_mut_ptr() as *mut $cc,
                               d.as_mut_ptr() as *mut $cc,
                               du.as_mut_ptr() as *mut $cc,
                               b.as_mut_ptr() as *mut $cc, ldb as i32)
                };
                map_info("lapack", info)
            }

            fn ptsv(layout: Layout, n: usize, nrhs: usize,
                    d: &mut [Self::Real], e: &mut [Self],
                    b: &mut [Self], ldb: usize) -> Result<()> {
                let info = unsafe {
                    sys::$ptsv(layout_raw(layout), n as i32, nrhs as i32,
                               d.as_mut_ptr(),
                               e.as_mut_ptr() as *mut $cc,
                               b.as_mut_ptr() as *mut $cc, ldb as i32)
                };
                map_info("lapack", info)
            }

            fn gels(layout: Layout, trans: Trans, m: usize, n: usize, nrhs: usize,
                    a: &mut [Self], lda: usize,
                    b: &mut [Self], ldb: usize) -> Result<()> {
                let info = unsafe {
                    sys::$gels(layout_raw(layout), trans_char(trans),
                               m as i32, n as i32, nrhs as i32,
                               a.as_mut_ptr() as *mut $cc, lda as i32,
                               b.as_mut_ptr() as *mut $cc, ldb as i32)
                };
                map_info("lapack", info)
            }

            fn geqrf(layout: Layout, m: usize, n: usize,
                     a: &mut [Self], lda: usize, tau: &mut [Self]) -> Result<()> {
                let info = unsafe {
                    sys::$geqrf(layout_raw(layout), m as i32, n as i32,
                                a.as_mut_ptr() as *mut $cc, lda as i32,
                                tau.as_mut_ptr() as *mut $cc)
                };
                map_info("lapack", info)
            }

            fn gelqf(layout: Layout, m: usize, n: usize,
                     a: &mut [Self], lda: usize, tau: &mut [Self]) -> Result<()> {
                let info = unsafe {
                    sys::$gelqf(layout_raw(layout), m as i32, n as i32,
                                a.as_mut_ptr() as *mut $cc, lda as i32,
                                tau.as_mut_ptr() as *mut $cc)
                };
                map_info("lapack", info)
            }
        }

        impl ComplexScalar for $t {
            fn hesv(layout: Layout, uplo: Uplo, n: usize, nrhs: usize,
                    a: &mut [Self], lda: usize, ipiv: &mut [i32],
                    b: &mut [Self], ldb: usize) -> Result<()> {
                let info = unsafe {
                    sys::$hesv(layout_raw(layout), uplo_char(uplo),
                               n as i32, nrhs as i32,
                               a.as_mut_ptr() as *mut $cc, lda as i32, ipiv.as_mut_ptr(),
                               b.as_mut_ptr() as *mut $cc, ldb as i32)
                };
                map_info("lapack", info)
            }

            fn hetrf(layout: Layout, uplo: Uplo, n: usize,
                     a: &mut [Self], lda: usize, ipiv: &mut [i32]) -> Result<()> {
                let info = unsafe {
                    sys::$hetrf(layout_raw(layout), uplo_char(uplo),
                                n as i32, a.as_mut_ptr() as *mut $cc, lda as i32,
                                ipiv.as_mut_ptr())
                };
                map_info("lapack", info)
            }

            fn hetrs(layout: Layout, uplo: Uplo, n: usize, nrhs: usize,
                     a: &[Self], lda: usize, ipiv: &[i32],
                     b: &mut [Self], ldb: usize) -> Result<()> {
                let info = unsafe {
                    sys::$hetrs(layout_raw(layout), uplo_char(uplo),
                                n as i32, nrhs as i32,
                                a.as_ptr() as *const $cc, lda as i32, ipiv.as_ptr(),
                                b.as_mut_ptr() as *mut $cc, ldb as i32)
                };
                map_info("lapack", info)
            }
        }
    };
}

impl_complex!(
    Complex32, f32, CC32,
    gesv = LAPACKE_cgesv, getrf = LAPACKE_cgetrf, getrs = LAPACKE_cgetrs, getri = LAPACKE_cgetri,
    posv = LAPACKE_cposv, potrf = LAPACKE_cpotrf, potrs = LAPACKE_cpotrs, potri = LAPACKE_cpotri,
    sysv = LAPACKE_csysv, sytrf = LAPACKE_csytrf, sytrs = LAPACKE_csytrs,
    hesv = LAPACKE_chesv, hetrf = LAPACKE_chetrf, hetrs = LAPACKE_chetrs,
    gbsv = LAPACKE_cgbsv, gbtrf = LAPACKE_cgbtrf, gbtrs = LAPACKE_cgbtrs,
    gtsv = LAPACKE_cgtsv, ptsv = LAPACKE_cptsv,
    gels = LAPACKE_cgels, geqrf = LAPACKE_cgeqrf, gelqf = LAPACKE_cgelqf
);
impl_complex!(
    Complex64, f64, CC64,
    gesv = LAPACKE_zgesv, getrf = LAPACKE_zgetrf, getrs = LAPACKE_zgetrs, getri = LAPACKE_zgetri,
    posv = LAPACKE_zposv, potrf = LAPACKE_zpotrf, potrs = LAPACKE_zpotrs, potri = LAPACKE_zpotri,
    sysv = LAPACKE_zsysv, sytrf = LAPACKE_zsytrf, sytrs = LAPACKE_zsytrs,
    hesv = LAPACKE_zhesv, hetrf = LAPACKE_zhetrf, hetrs = LAPACKE_zhetrs,
    gbsv = LAPACKE_zgbsv, gbtrf = LAPACKE_zgbtrf, gbtrs = LAPACKE_zgbtrs,
    gtsv = LAPACKE_zgtsv, ptsv = LAPACKE_zptsv,
    gels = LAPACKE_zgels, geqrf = LAPACKE_zgeqrf, gelqf = LAPACKE_zgelqf
);

// =========================================================================
//   Free-function convenience entry points (tightly-packed row-major)
// =========================================================================

pub fn gesv<T: Scalar>(n: usize, a: &mut [T], ipiv: &mut [i32], b: &mut [T]) -> Result<()> {
    if n == 0 { return Ok(()); }
    let nrhs = b.len() / n;
    T::gesv(Layout::RowMajor, n, nrhs, a, n, ipiv, b, nrhs.max(1))
}

pub fn getrf<T: Scalar>(m: usize, n: usize, a: &mut [T], ipiv: &mut [i32]) -> Result<()> {
    T::getrf(Layout::RowMajor, m, n, a, n, ipiv)
}

pub fn getrs<T: Scalar>(trans: Trans, n: usize, a: &[T], ipiv: &[i32], b: &mut [T]) -> Result<()> {
    if n == 0 { return Ok(()); }
    let nrhs = b.len() / n;
    T::getrs(Layout::RowMajor, trans, n, nrhs, a, n, ipiv, b, nrhs.max(1))
}

pub fn getri<T: Scalar>(n: usize, a: &mut [T], ipiv: &[i32]) -> Result<()> {
    T::getri(Layout::RowMajor, n, a, n, ipiv)
}

pub fn posv<T: Scalar>(uplo: Uplo, n: usize, a: &mut [T], b: &mut [T]) -> Result<()> {
    if n == 0 { return Ok(()); }
    let nrhs = b.len() / n;
    T::posv(Layout::RowMajor, uplo, n, nrhs, a, n, b, nrhs.max(1))
}

pub fn potrf<T: Scalar>(uplo: Uplo, n: usize, a: &mut [T]) -> Result<()> {
    T::potrf(Layout::RowMajor, uplo, n, a, n)
}

pub fn potrs<T: Scalar>(uplo: Uplo, n: usize, a: &[T], b: &mut [T]) -> Result<()> {
    if n == 0 { return Ok(()); }
    let nrhs = b.len() / n;
    T::potrs(Layout::RowMajor, uplo, n, nrhs, a, n, b, nrhs.max(1))
}

pub fn sysv<T: Scalar>(uplo: Uplo, n: usize, a: &mut [T], ipiv: &mut [i32], b: &mut [T]) -> Result<()> {
    if n == 0 { return Ok(()); }
    let nrhs = b.len() / n;
    T::sysv(Layout::RowMajor, uplo, n, nrhs, a, n, ipiv, b, nrhs.max(1))
}

pub fn hesv<T: ComplexScalar>(uplo: Uplo, n: usize, a: &mut [T], ipiv: &mut [i32], b: &mut [T]) -> Result<()> {
    if n == 0 { return Ok(()); }
    let nrhs = b.len() / n;
    T::hesv(Layout::RowMajor, uplo, n, nrhs, a, n, ipiv, b, nrhs.max(1))
}

pub fn gtsv<T: Scalar>(n: usize, dl: &mut [T], d: &mut [T], du: &mut [T], b: &mut [T]) -> Result<()> {
    if n == 0 { return Ok(()); }
    let nrhs = b.len() / n;
    T::gtsv(Layout::RowMajor, n, nrhs, dl, d, du, b, nrhs.max(1))
}

pub fn ptsv<T: Scalar>(n: usize, d: &mut [T::Real], e: &mut [T], b: &mut [T]) -> Result<()> {
    if n == 0 { return Ok(()); }
    let nrhs = b.len() / n;
    T::ptsv(Layout::RowMajor, n, nrhs, d, e, b, nrhs.max(1))
}

pub fn gels<T: Scalar>(trans: Trans, m: usize, n: usize, a: &mut [T], b: &mut [T]) -> Result<()> {
    let max_mn = m.max(n);
    let nrhs = if max_mn == 0 { 0 } else { b.len() / max_mn };
    T::gels(Layout::RowMajor, trans, m, n, nrhs, a, n, b, nrhs.max(1))
}

pub fn geqrf<T: Scalar>(m: usize, n: usize, a: &mut [T], tau: &mut [T]) -> Result<()> {
    T::geqrf(Layout::RowMajor, m, n, a, n, tau)
}

pub fn gelqf<T: Scalar>(m: usize, n: usize, a: &mut [T], tau: &mut [T]) -> Result<()> {
    T::gelqf(Layout::RowMajor, m, n, a, n, tau)
}

// =========================================================================
//   Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, eps: f64) {
        assert!((a - b).abs() < eps, "|{a} - {b}| > {eps}");
    }

    #[test]
    fn gesv_2x2_f64() {
        let mut a = [2.0_f64, 1.0, 1.0, 3.0];
        let mut b = [1.0_f64, 2.0];
        let mut ipiv = [0_i32; 2];
        gesv(2, &mut a, &mut ipiv, &mut b).unwrap();
        approx(b[0], 0.2, 1e-12);
        approx(b[1], 0.6, 1e-12);
    }

    #[test]
    fn gesv_singular_returns_error() {
        let mut a = [1.0_f64, 2.0, 2.0, 4.0];
        let mut b = [3.0_f64, 6.0];
        let mut ipiv = [0_i32; 2];
        let err = gesv(2, &mut a, &mut ipiv, &mut b).unwrap_err();
        match err {
            Error::Status { code, .. } => assert!(code > 0),
            _ => panic!("expected Status error"),
        }
    }

    #[test]
    fn getrf_then_getrs() {
        // A = [[4,3],[6,3]], x_true = [1,1], so b = A·x_true = [7, 9]
        let mut a = [4.0_f64, 3.0, 6.0, 3.0];
        let mut ipiv = [0_i32; 2];
        getrf(2, 2, &mut a, &mut ipiv).unwrap();
        let mut b = [7.0_f64, 9.0];
        getrs::<f64>(Trans::No, 2, &a, &ipiv, &mut b).unwrap();
        approx(b[0], 1.0, 1e-9);
        approx(b[1], 1.0, 1e-9);
    }

    #[test]
    fn getri_inverse() {
        let mut a = [2.0_f64, 0.0, 0.0, 4.0];
        let mut ipiv = [0_i32; 2];
        getrf(2, 2, &mut a, &mut ipiv).unwrap();
        getri::<f64>(2, &mut a, &ipiv).unwrap();
        approx(a[0], 0.5, 1e-12);
        approx(a[3], 0.25, 1e-12);
    }

    #[test]
    fn posv_cholesky_solve() {
        let mut a = [4.0_f64, 1.0, 0.0, 3.0];
        let mut b = [1.0_f64, 2.0];
        posv(Uplo::Upper, 2, &mut a, &mut b).unwrap();
        let det = 4.0 * 3.0 - 1.0;
        approx(b[0], (3.0 - 2.0) / det, 1e-12);
        approx(b[1], (8.0 - 1.0) / det, 1e-12);
    }

    #[test]
    fn potrf_then_potrs() {
        let mut a = [4.0_f64, 1.0, 0.0, 3.0];
        potrf(Uplo::Upper, 2, &mut a).unwrap();
        let mut b = [1.0_f64, 2.0];
        potrs::<f64>(Uplo::Upper, 2, &a, &mut b).unwrap();
        let det = 4.0 * 3.0 - 1.0;
        approx(b[0], (3.0 - 2.0) / det, 1e-12);
    }

    #[test]
    fn sysv_solve() {
        // A symmetric = [[1,2],[2,3]] (upper triangle), b = [3, 5] → x = [1, 1]
        let mut a = [1.0_f64, 2.0, 0.0, 3.0];
        let mut b = [3.0_f64, 5.0];
        let mut ipiv = [0_i32; 2];
        sysv(Uplo::Upper, 2, &mut a, &mut ipiv, &mut b).unwrap();
        approx(b[0], 1.0, 1e-12);
        approx(b[1], 1.0, 1e-12);
    }

    #[test]
    fn gtsv_solve() {
        let mut dl = [1.0_f64, 1.0];
        let mut d = [2.0_f64, 2.0, 2.0];
        let mut du = [1.0_f64, 1.0];
        let mut b = [3.0_f64, 4.0, 3.0];
        gtsv(3, &mut dl, &mut d, &mut du, &mut b).unwrap();
        for v in &b { approx(*v, 1.0, 1e-12); }
    }

    #[test]
    fn ptsv_solve_real() {
        let mut d = [2.0_f64, 2.0, 2.0];
        let mut e = [1.0_f64, 1.0];
        let mut b = [3.0_f64, 4.0, 3.0];
        ptsv::<f64>(3, &mut d, &mut e, &mut b).unwrap();
        for v in &b { approx(*v, 1.0, 1e-12); }
    }

    #[test]
    fn gels_overdetermined() {
        let mut a = [1.0_f64, 0.0, 0.0, 1.0, 0.0, 0.0];
        let mut b = [1.0_f64, 2.0, 3.0];
        gels::<f64>(Trans::No, 3, 2, &mut a, &mut b).unwrap();
        approx(b[0], 1.0, 1e-12);
        approx(b[1], 2.0, 1e-12);
    }

    #[test]
    fn geqrf_runs() {
        let mut a = [1.0_f64, 2.0, 3.0, 4.0];
        let mut tau = [0.0_f64; 2];
        geqrf::<f64>(2, 2, &mut a, &mut tau).unwrap();
        assert!(tau[0] >= 1.0 && tau[0] <= 2.0, "tau[0]={}", tau[0]);
    }

    #[test]
    fn gesv_complex64_identity() {
        let mut a = [Complex64::ONE, Complex64::ZERO, Complex64::ZERO, Complex64::ONE];
        let mut b = [Complex64::new(3.0, 1.0), Complex64::new(4.0, -1.0)];
        let mut ipiv = [0_i32; 2];
        gesv(2, &mut a, &mut ipiv, &mut b).unwrap();
        assert_eq!(b[0], Complex64::new(3.0, 1.0));
        assert_eq!(b[1], Complex64::new(4.0, -1.0));
    }

    #[test]
    fn potrf_complex64_then_potrs() {
        let two = Complex64::new(2.0, 0.0);
        let three = Complex64::new(3.0, 0.0);
        let z = Complex64::ZERO;
        let mut a = [two, z, z, three];
        potrf(Uplo::Upper, 2, &mut a).unwrap();
        let mut b = [two, three];
        potrs::<Complex64>(Uplo::Upper, 2, &a, &mut b).unwrap();
        assert!((b[0].re - 1.0).abs() < 1e-12);
        assert!((b[1].re - 1.0).abs() < 1e-12);
    }

    #[test]
    fn dim_mismatch_is_error() {
        let mut a = [1.0_f64; 3];
        let mut ipiv = [0_i32; 2];
        let mut b = [1.0_f64, 2.0];
        let err = gesv(2, &mut a, &mut ipiv, &mut b).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)));
    }
}
