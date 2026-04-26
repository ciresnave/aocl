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

    // --- Singular Value Decomposition -----------------------------------

    /// SVD via the QR-iteration algorithm. `s` receives `min(m, n)`
    /// singular values in descending order. The job parameters specify
    /// how much of `U` and `V^H` to compute.
    #[allow(clippy::too_many_arguments)]
    fn gesvd(
        layout: Layout,
        jobu: SvdJob,
        jobvt: SvdJob,
        m: usize,
        n: usize,
        a: &mut [Self],
        lda: usize,
        s: &mut [Self::Real],
        u: &mut [Self],
        ldu: usize,
        vt: &mut [Self],
        ldvt: usize,
        superb: &mut [Self::Real],
    ) -> Result<()>;

    /// SVD via the divide-and-conquer algorithm (typically faster than
    /// gesvd for large matrices).
    #[allow(clippy::too_many_arguments)]
    fn gesdd(
        layout: Layout,
        jobz: SvdJob,
        m: usize,
        n: usize,
        a: &mut [Self],
        lda: usize,
        s: &mut [Self::Real],
        u: &mut [Self],
        ldu: usize,
        vt: &mut [Self],
        ldvt: usize,
    ) -> Result<()>;

    // --- Advanced least squares -----------------------------------------

    /// Least squares using SVD (handles rank-deficient `A`). On exit
    /// `rank` is the effective numerical rank.
    #[allow(clippy::too_many_arguments)]
    fn gelsd(
        layout: Layout,
        m: usize,
        n: usize,
        nrhs: usize,
        a: &mut [Self],
        lda: usize,
        b: &mut [Self],
        ldb: usize,
        s: &mut [Self::Real],
        rcond: Self::Real,
        rank: &mut i32,
    ) -> Result<()>;

    /// Least squares using QR with column pivoting.
    #[allow(clippy::too_many_arguments)]
    fn gelsy(
        layout: Layout,
        m: usize,
        n: usize,
        nrhs: usize,
        a: &mut [Self],
        lda: usize,
        b: &mut [Self],
        ldb: usize,
        jpvt: &mut [i32],
        rcond: Self::Real,
        rank: &mut i32,
    ) -> Result<()>;
}

/// Whether eigendecomposition / SVD routines compute vectors as well
/// as the values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Compute {
    /// Compute eigenvalues / singular values only.
    ValuesOnly,
    /// Compute both values and the corresponding vectors.
    ValuesAndVectors,
}

impl Compute {
    fn job_char(self) -> c_char {
        match self {
            Compute::ValuesOnly => b'N' as c_char,
            Compute::ValuesAndVectors => b'V' as c_char,
        }
    }
}

/// Variant for `gesvd` / `gesdd` controlling how much of `U` and `V^H`
/// is computed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SvdJob {
    /// Compute the full square `U` (m × m) or `V^H` (n × n) matrices.
    All,
    /// Compute only the leading `min(m, n)` columns of `U` / rows of `V^H`.
    Singular,
    /// Compute the corresponding factor in-place inside `A`.
    Overwrite,
    /// Do not compute that factor.
    None,
}

impl SvdJob {
    fn job_char(self) -> c_char {
        match self {
            SvdJob::All => b'A' as c_char,
            SvdJob::Singular => b'S' as c_char,
            SvdJob::Overwrite => b'O' as c_char,
            SvdJob::None => b'N' as c_char,
        }
    }
}

/// Operations defined only for real precisions.
pub trait RealScalar: Scalar<Real = Self> {
    /// Symmetric eigendecomposition: compute eigenvalues (and optionally
    /// eigenvectors) of an `n × n` symmetric matrix.
    /// On exit, `a` contains the eigenvectors when `compute = ValuesAndVectors`.
    /// `w` receives the eigenvalues in ascending order.
    #[allow(clippy::too_many_arguments)]
    fn syev(
        layout: Layout,
        compute: Compute,
        uplo: Uplo,
        n: usize,
        a: &mut [Self],
        lda: usize,
        w: &mut [Self],
    ) -> Result<()>;

    /// Non-symmetric eigendecomposition for an `n × n` real matrix.
    /// Eigenvalues come back as `wr + i·wi` (real and imaginary parts);
    /// complex conjugate pairs share consecutive entries.
    /// `vl` and `vr` receive left/right eigenvectors when their respective
    /// `compute_*` flags request them; pass empty slices when not needed.
    #[allow(clippy::too_many_arguments)]
    fn geev(
        layout: Layout,
        compute_left: Compute,
        compute_right: Compute,
        n: usize,
        a: &mut [Self],
        lda: usize,
        wr: &mut [Self],
        wi: &mut [Self],
        vl: &mut [Self],
        ldvl: usize,
        vr: &mut [Self],
        ldvr: usize,
    ) -> Result<()>;
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

    /// Hermitian eigendecomposition: compute eigenvalues (real) and
    /// optionally eigenvectors of an `n × n` Hermitian matrix.
    #[allow(clippy::too_many_arguments)]
    fn heev(
        layout: Layout,
        compute: Compute,
        uplo: Uplo,
        n: usize,
        a: &mut [Self],
        lda: usize,
        w: &mut [Self::Real],
    ) -> Result<()>;

    /// Non-symmetric complex eigendecomposition.
    /// `w` is the array of complex eigenvalues.
    #[allow(clippy::too_many_arguments)]
    fn geev(
        layout: Layout,
        compute_left: Compute,
        compute_right: Compute,
        n: usize,
        a: &mut [Self],
        lda: usize,
        w: &mut [Self],
        vl: &mut [Self],
        ldvl: usize,
        vr: &mut [Self],
        ldvr: usize,
    ) -> Result<()>;
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
        gels = $gels:ident, geqrf = $geqrf:ident, gelqf = $gelqf:ident,
        syev = $syev:ident, geev = $geev:ident,
        gesvd = $gesvd:ident, gesdd = $gesdd:ident,
        gelsd = $gelsd:ident, gelsy = $gelsy:ident
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

            #[allow(clippy::too_many_arguments)]
            fn gesvd(
                layout: Layout,
                jobu: SvdJob, jobvt: SvdJob,
                m: usize, n: usize,
                a: &mut [Self], lda: usize,
                s: &mut [Self::Real],
                u: &mut [Self], ldu: usize,
                vt: &mut [Self], ldvt: usize,
                superb: &mut [Self::Real],
            ) -> Result<()> {
                let info = unsafe {
                    sys::$gesvd(
                        layout_raw(layout),
                        jobu.job_char(), jobvt.job_char(),
                        m as i32, n as i32,
                        a.as_mut_ptr(), lda as i32,
                        s.as_mut_ptr(),
                        u.as_mut_ptr(), ldu as i32,
                        vt.as_mut_ptr(), ldvt as i32,
                        superb.as_mut_ptr(),
                    )
                };
                map_info("lapack", info)
            }

            #[allow(clippy::too_many_arguments)]
            fn gesdd(
                layout: Layout,
                jobz: SvdJob,
                m: usize, n: usize,
                a: &mut [Self], lda: usize,
                s: &mut [Self::Real],
                u: &mut [Self], ldu: usize,
                vt: &mut [Self], ldvt: usize,
            ) -> Result<()> {
                let info = unsafe {
                    sys::$gesdd(
                        layout_raw(layout),
                        jobz.job_char(),
                        m as i32, n as i32,
                        a.as_mut_ptr(), lda as i32,
                        s.as_mut_ptr(),
                        u.as_mut_ptr(), ldu as i32,
                        vt.as_mut_ptr(), ldvt as i32,
                    )
                };
                map_info("lapack", info)
            }

            #[allow(clippy::too_many_arguments)]
            fn gelsd(
                layout: Layout,
                m: usize, n: usize, nrhs: usize,
                a: &mut [Self], lda: usize,
                b: &mut [Self], ldb: usize,
                s: &mut [Self::Real],
                rcond: Self::Real,
                rank: &mut i32,
            ) -> Result<()> {
                let info = unsafe {
                    sys::$gelsd(
                        layout_raw(layout),
                        m as i32, n as i32, nrhs as i32,
                        a.as_mut_ptr(), lda as i32,
                        b.as_mut_ptr(), ldb as i32,
                        s.as_mut_ptr(), rcond, rank,
                    )
                };
                map_info("lapack", info)
            }

            #[allow(clippy::too_many_arguments)]
            fn gelsy(
                layout: Layout,
                m: usize, n: usize, nrhs: usize,
                a: &mut [Self], lda: usize,
                b: &mut [Self], ldb: usize,
                jpvt: &mut [i32],
                rcond: Self::Real,
                rank: &mut i32,
            ) -> Result<()> {
                let info = unsafe {
                    sys::$gelsy(
                        layout_raw(layout),
                        m as i32, n as i32, nrhs as i32,
                        a.as_mut_ptr(), lda as i32,
                        b.as_mut_ptr(), ldb as i32,
                        jpvt.as_mut_ptr(), rcond, rank,
                    )
                };
                map_info("lapack", info)
            }
        }

        impl RealScalar for $t {
            #[allow(clippy::too_many_arguments)]
            fn syev(
                layout: Layout, compute: Compute, uplo: Uplo, n: usize,
                a: &mut [Self], lda: usize, w: &mut [Self],
            ) -> Result<()> {
                let info = unsafe {
                    sys::$syev(
                        layout_raw(layout), compute.job_char(), uplo_char(uplo),
                        n as i32, a.as_mut_ptr(), lda as i32, w.as_mut_ptr(),
                    )
                };
                map_info("lapack", info)
            }

            #[allow(clippy::too_many_arguments)]
            fn geev(
                layout: Layout,
                compute_left: Compute, compute_right: Compute,
                n: usize,
                a: &mut [Self], lda: usize,
                wr: &mut [Self], wi: &mut [Self],
                vl: &mut [Self], ldvl: usize,
                vr: &mut [Self], ldvr: usize,
            ) -> Result<()> {
                let info = unsafe {
                    sys::$geev(
                        layout_raw(layout),
                        compute_left.job_char(), compute_right.job_char(),
                        n as i32,
                        a.as_mut_ptr(), lda as i32,
                        wr.as_mut_ptr(), wi.as_mut_ptr(),
                        vl.as_mut_ptr(), ldvl as i32,
                        vr.as_mut_ptr(), ldvr as i32,
                    )
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
    gels = LAPACKE_sgels, geqrf = LAPACKE_sgeqrf, gelqf = LAPACKE_sgelqf,
    syev = LAPACKE_ssyev, geev = LAPACKE_sgeev,
    gesvd = LAPACKE_sgesvd, gesdd = LAPACKE_sgesdd,
    gelsd = LAPACKE_sgelsd, gelsy = LAPACKE_sgelsy
);
impl_real!(
    f64,
    gesv = LAPACKE_dgesv, getrf = LAPACKE_dgetrf, getrs = LAPACKE_dgetrs, getri = LAPACKE_dgetri,
    posv = LAPACKE_dposv, potrf = LAPACKE_dpotrf, potrs = LAPACKE_dpotrs, potri = LAPACKE_dpotri,
    sysv = LAPACKE_dsysv, sytrf = LAPACKE_dsytrf, sytrs = LAPACKE_dsytrs,
    gbsv = LAPACKE_dgbsv, gbtrf = LAPACKE_dgbtrf, gbtrs = LAPACKE_dgbtrs,
    gtsv = LAPACKE_dgtsv, ptsv = LAPACKE_dptsv,
    gels = LAPACKE_dgels, geqrf = LAPACKE_dgeqrf, gelqf = LAPACKE_dgelqf,
    syev = LAPACKE_dsyev, geev = LAPACKE_dgeev,
    gesvd = LAPACKE_dgesvd, gesdd = LAPACKE_dgesdd,
    gelsd = LAPACKE_dgelsd, gelsy = LAPACKE_dgelsy
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
        gels = $gels:ident, geqrf = $geqrf:ident, gelqf = $gelqf:ident,
        heev = $heev:ident, geev = $geev:ident,
        gesvd = $gesvd:ident, gesdd = $gesdd:ident,
        gelsd = $gelsd:ident, gelsy = $gelsy:ident
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

            #[allow(clippy::too_many_arguments)]
            fn gesvd(
                layout: Layout,
                jobu: SvdJob, jobvt: SvdJob,
                m: usize, n: usize,
                a: &mut [Self], lda: usize,
                s: &mut [Self::Real],
                u: &mut [Self], ldu: usize,
                vt: &mut [Self], ldvt: usize,
                superb: &mut [Self::Real],
            ) -> Result<()> {
                let info = unsafe {
                    sys::$gesvd(
                        layout_raw(layout),
                        jobu.job_char(), jobvt.job_char(),
                        m as i32, n as i32,
                        a.as_mut_ptr() as *mut $cc, lda as i32,
                        s.as_mut_ptr(),
                        u.as_mut_ptr() as *mut $cc, ldu as i32,
                        vt.as_mut_ptr() as *mut $cc, ldvt as i32,
                        superb.as_mut_ptr(),
                    )
                };
                map_info("lapack", info)
            }

            #[allow(clippy::too_many_arguments)]
            fn gesdd(
                layout: Layout,
                jobz: SvdJob,
                m: usize, n: usize,
                a: &mut [Self], lda: usize,
                s: &mut [Self::Real],
                u: &mut [Self], ldu: usize,
                vt: &mut [Self], ldvt: usize,
            ) -> Result<()> {
                let info = unsafe {
                    sys::$gesdd(
                        layout_raw(layout),
                        jobz.job_char(),
                        m as i32, n as i32,
                        a.as_mut_ptr() as *mut $cc, lda as i32,
                        s.as_mut_ptr(),
                        u.as_mut_ptr() as *mut $cc, ldu as i32,
                        vt.as_mut_ptr() as *mut $cc, ldvt as i32,
                    )
                };
                map_info("lapack", info)
            }

            #[allow(clippy::too_many_arguments)]
            fn gelsd(
                layout: Layout,
                m: usize, n: usize, nrhs: usize,
                a: &mut [Self], lda: usize,
                b: &mut [Self], ldb: usize,
                s: &mut [Self::Real],
                rcond: Self::Real,
                rank: &mut i32,
            ) -> Result<()> {
                let info = unsafe {
                    sys::$gelsd(
                        layout_raw(layout),
                        m as i32, n as i32, nrhs as i32,
                        a.as_mut_ptr() as *mut $cc, lda as i32,
                        b.as_mut_ptr() as *mut $cc, ldb as i32,
                        s.as_mut_ptr(), rcond, rank,
                    )
                };
                map_info("lapack", info)
            }

            #[allow(clippy::too_many_arguments)]
            fn gelsy(
                layout: Layout,
                m: usize, n: usize, nrhs: usize,
                a: &mut [Self], lda: usize,
                b: &mut [Self], ldb: usize,
                jpvt: &mut [i32],
                rcond: Self::Real,
                rank: &mut i32,
            ) -> Result<()> {
                let info = unsafe {
                    sys::$gelsy(
                        layout_raw(layout),
                        m as i32, n as i32, nrhs as i32,
                        a.as_mut_ptr() as *mut $cc, lda as i32,
                        b.as_mut_ptr() as *mut $cc, ldb as i32,
                        jpvt.as_mut_ptr(), rcond, rank,
                    )
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

            #[allow(clippy::too_many_arguments)]
            fn heev(
                layout: Layout, compute: Compute, uplo: Uplo, n: usize,
                a: &mut [Self], lda: usize, w: &mut [Self::Real],
            ) -> Result<()> {
                let info = unsafe {
                    sys::$heev(
                        layout_raw(layout), compute.job_char(), uplo_char(uplo),
                        n as i32, a.as_mut_ptr() as *mut $cc, lda as i32,
                        w.as_mut_ptr(),
                    )
                };
                map_info("lapack", info)
            }

            #[allow(clippy::too_many_arguments)]
            fn geev(
                layout: Layout,
                compute_left: Compute, compute_right: Compute,
                n: usize,
                a: &mut [Self], lda: usize,
                w: &mut [Self],
                vl: &mut [Self], ldvl: usize,
                vr: &mut [Self], ldvr: usize,
            ) -> Result<()> {
                let info = unsafe {
                    sys::$geev(
                        layout_raw(layout),
                        compute_left.job_char(), compute_right.job_char(),
                        n as i32,
                        a.as_mut_ptr() as *mut $cc, lda as i32,
                        w.as_mut_ptr() as *mut $cc,
                        vl.as_mut_ptr() as *mut $cc, ldvl as i32,
                        vr.as_mut_ptr() as *mut $cc, ldvr as i32,
                    )
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
    gels = LAPACKE_cgels, geqrf = LAPACKE_cgeqrf, gelqf = LAPACKE_cgelqf,
    heev = LAPACKE_cheev, geev = LAPACKE_cgeev,
    gesvd = LAPACKE_cgesvd, gesdd = LAPACKE_cgesdd,
    gelsd = LAPACKE_cgelsd, gelsy = LAPACKE_cgelsy
);
impl_complex!(
    Complex64, f64, CC64,
    gesv = LAPACKE_zgesv, getrf = LAPACKE_zgetrf, getrs = LAPACKE_zgetrs, getri = LAPACKE_zgetri,
    posv = LAPACKE_zposv, potrf = LAPACKE_zpotrf, potrs = LAPACKE_zpotrs, potri = LAPACKE_zpotri,
    sysv = LAPACKE_zsysv, sytrf = LAPACKE_zsytrf, sytrs = LAPACKE_zsytrs,
    hesv = LAPACKE_zhesv, hetrf = LAPACKE_zhetrf, hetrs = LAPACKE_zhetrs,
    gbsv = LAPACKE_zgbsv, gbtrf = LAPACKE_zgbtrf, gbtrs = LAPACKE_zgbtrs,
    gtsv = LAPACKE_zgtsv, ptsv = LAPACKE_zptsv,
    gels = LAPACKE_zgels, geqrf = LAPACKE_zgeqrf, gelqf = LAPACKE_zgelqf,
    heev = LAPACKE_zheev, geev = LAPACKE_zgeev,
    gesvd = LAPACKE_zgesvd, gesdd = LAPACKE_zgesdd,
    gelsd = LAPACKE_zgelsd, gelsy = LAPACKE_zgelsy
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

/// Symmetric eigendecomposition (real precisions). On exit, `a` holds
/// the eigenvectors when `compute = ValuesAndVectors`; `w` holds
/// eigenvalues in ascending order.
pub fn syev<T: RealScalar>(
    compute: Compute,
    uplo: Uplo,
    n: usize,
    a: &mut [T],
    w: &mut [T],
) -> Result<()> {
    T::syev(Layout::RowMajor, compute, uplo, n, a, n, w)
}

/// Hermitian eigendecomposition (complex precisions). `w` is real.
pub fn heev<T: ComplexScalar>(
    compute: Compute,
    uplo: Uplo,
    n: usize,
    a: &mut [T],
    w: &mut [T::Real],
) -> Result<()> {
    T::heev(Layout::RowMajor, compute, uplo, n, a, n, w)
}

/// SVD via the QR-iteration algorithm. Returns the count of converged
/// singular values; pass `superb` of size `min(m, n) - 1`.
#[allow(clippy::too_many_arguments)]
pub fn gesvd<T: Scalar>(
    jobu: SvdJob,
    jobvt: SvdJob,
    m: usize,
    n: usize,
    a: &mut [T],
    s: &mut [T::Real],
    u: &mut [T],
    ldu: usize,
    vt: &mut [T],
    ldvt: usize,
    superb: &mut [T::Real],
) -> Result<()> {
    T::gesvd(Layout::RowMajor, jobu, jobvt, m, n, a, n, s, u, ldu, vt, ldvt, superb)
}

/// SVD via divide-and-conquer.
#[allow(clippy::too_many_arguments)]
pub fn gesdd<T: Scalar>(
    jobz: SvdJob,
    m: usize,
    n: usize,
    a: &mut [T],
    s: &mut [T::Real],
    u: &mut [T],
    ldu: usize,
    vt: &mut [T],
    ldvt: usize,
) -> Result<()> {
    T::gesdd(Layout::RowMajor, jobz, m, n, a, n, s, u, ldu, vt, ldvt)
}

/// SVD-based least squares (handles rank-deficient `A`).
#[allow(clippy::too_many_arguments)]
pub fn gelsd<T: Scalar>(
    m: usize,
    n: usize,
    a: &mut [T],
    b: &mut [T],
    s: &mut [T::Real],
    rcond: T::Real,
) -> Result<i32> {
    let max_mn = m.max(n);
    let nrhs = if max_mn == 0 { 0 } else { b.len() / max_mn };
    let mut rank: i32 = 0;
    T::gelsd(Layout::RowMajor, m, n, nrhs, a, n, b, nrhs.max(1), s, rcond, &mut rank)?;
    Ok(rank)
}

/// QR-with-column-pivoting least squares.
#[allow(clippy::too_many_arguments)]
pub fn gelsy<T: Scalar>(
    m: usize,
    n: usize,
    a: &mut [T],
    b: &mut [T],
    jpvt: &mut [i32],
    rcond: T::Real,
) -> Result<i32> {
    let max_mn = m.max(n);
    let nrhs = if max_mn == 0 { 0 } else { b.len() / max_mn };
    let mut rank: i32 = 0;
    T::gelsy(Layout::RowMajor, m, n, nrhs, a, n, b, nrhs.max(1), jpvt, rcond, &mut rank)?;
    Ok(rank)
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

    // --- Eigenvalues / SVD / advanced least squares -----------------------

    #[test]
    fn syev_diagonal_eigenvalues() {
        // A = diag(3, 1, 2) symmetric; eigenvalues sorted ascending: 1, 2, 3.
        let mut a = [
            3.0_f64, 0.0, 0.0,
            0.0,     1.0, 0.0,
            0.0,     0.0, 2.0,
        ];
        let mut w = [0.0_f64; 3];
        syev(Compute::ValuesOnly, Uplo::Upper, 3, &mut a, &mut w).unwrap();
        approx(w[0], 1.0, 1e-10);
        approx(w[1], 2.0, 1e-10);
        approx(w[2], 3.0, 1e-10);
    }

    #[test]
    fn heev_real_eigenvalues_for_hermitian() {
        // 2×2 Hermitian: [[2, i], [-i, 2]]. Eigenvalues = 1, 3.
        let mut a = [
            Complex64::new(2.0, 0.0), Complex64::new(0.0, 1.0),
            Complex64::ZERO,          Complex64::new(2.0, 0.0),
        ];
        let mut w = [0.0_f64; 2];
        heev(Compute::ValuesOnly, Uplo::Upper, 2, &mut a, &mut w).unwrap();
        approx(w[0], 1.0, 1e-10);
        approx(w[1], 3.0, 1e-10);
    }

    #[test]
    fn gesvd_diagonal_singular_values() {
        // A = diag(2, 3) singular values are [3, 2] (descending).
        let mut a = [2.0_f64, 0.0, 0.0, 3.0];
        let mut s = [0.0_f64; 2];
        let mut u = [0.0_f64; 4];
        let mut vt = [0.0_f64; 4];
        let mut superb = [0.0_f64; 2];
        gesvd::<f64>(
            SvdJob::All, SvdJob::All, 2, 2, &mut a, &mut s,
            &mut u, 2, &mut vt, 2, &mut superb,
        ).unwrap();
        approx(s[0], 3.0, 1e-10);
        approx(s[1], 2.0, 1e-10);
    }

    #[test]
    fn gesdd_diagonal() {
        let mut a = [4.0_f64, 0.0, 0.0, 1.0];
        let mut s = [0.0_f64; 2];
        let mut u = [0.0_f64; 4];
        let mut vt = [0.0_f64; 4];
        gesdd::<f64>(SvdJob::All, 2, 2, &mut a, &mut s, &mut u, 2, &mut vt, 2).unwrap();
        approx(s[0], 4.0, 1e-10);
        approx(s[1], 1.0, 1e-10);
    }

    #[test]
    fn gelsd_full_rank() {
        // m=3, n=2 overdetermined least squares. A = [[1,0],[0,1],[0,0]],
        // b = [1, 2, 3] → solution [1, 2], singular values [1, 1], rank 2.
        let mut a = [1.0_f64, 0.0, 0.0, 1.0, 0.0, 0.0];
        let mut b = [1.0_f64, 2.0, 3.0];
        let mut s = [0.0_f64; 2];
        let rank = gelsd::<f64>(3, 2, &mut a, &mut b, &mut s, -1.0).unwrap();
        assert_eq!(rank, 2);
        approx(b[0], 1.0, 1e-10);
        approx(b[1], 2.0, 1e-10);
    }

    #[test]
    fn gelsy_full_rank() {
        let mut a = [1.0_f64, 0.0, 0.0, 1.0, 0.0, 0.0];
        let mut b = [1.0_f64, 2.0, 3.0];
        let mut jpvt = [0_i32; 2];
        let rank = gelsy::<f64>(3, 2, &mut a, &mut b, &mut jpvt, -1.0).unwrap();
        assert_eq!(rank, 2);
        approx(b[0], 1.0, 1e-10);
        approx(b[1], 2.0, 1e-10);
    }
}
