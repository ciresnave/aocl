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

pub use aocl_error::{Error, Result};
use aocl_lapack_sys as sys;
use aocl_types::sealed::Sealed;
pub use aocl_types::{Complex32, Complex64, Layout, Trans, Uplo};

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
    fn gesv(
        layout: Layout,
        n: usize,
        nrhs: usize,
        a: &mut [Self],
        lda: usize,
        ipiv: &mut [i32],
        b: &mut [Self],
        ldb: usize,
    ) -> Result<()>;

    /// LU factorization with partial pivoting.
    fn getrf(
        layout: Layout,
        m: usize,
        n: usize,
        a: &mut [Self],
        lda: usize,
        ipiv: &mut [i32],
    ) -> Result<()>;

    /// Back-substitute using a prior LU factorization.
    #[allow(clippy::too_many_arguments)]
    fn getrs(
        layout: Layout,
        trans: Trans,
        n: usize,
        nrhs: usize,
        a: &[Self],
        lda: usize,
        ipiv: &[i32],
        b: &mut [Self],
        ldb: usize,
    ) -> Result<()>;

    /// Inverse from a prior LU factorization.
    fn getri(layout: Layout, n: usize, a: &mut [Self], lda: usize, ipiv: &[i32]) -> Result<()>;

    // --- Cholesky --------------------------------------------------------

    /// Solve `A · X = B` for symmetric/Hermitian positive-definite `A`.
    #[allow(clippy::too_many_arguments)]
    fn posv(
        layout: Layout,
        uplo: Uplo,
        n: usize,
        nrhs: usize,
        a: &mut [Self],
        lda: usize,
        b: &mut [Self],
        ldb: usize,
    ) -> Result<()>;

    /// Cholesky factorization.
    fn potrf(layout: Layout, uplo: Uplo, n: usize, a: &mut [Self], lda: usize) -> Result<()>;

    /// Back-substitute using a prior Cholesky factorization.
    #[allow(clippy::too_many_arguments)]
    fn potrs(
        layout: Layout,
        uplo: Uplo,
        n: usize,
        nrhs: usize,
        a: &[Self],
        lda: usize,
        b: &mut [Self],
        ldb: usize,
    ) -> Result<()>;

    /// Inverse from a prior Cholesky factorization.
    fn potri(layout: Layout, uplo: Uplo, n: usize, a: &mut [Self], lda: usize) -> Result<()>;

    // --- Symmetric indefinite (Bunch-Kaufman) ----------------------------

    /// Solve `A · X = B` for symmetric indefinite `A` (real or complex
    /// **symmetric**, NOT Hermitian — for complex Hermitian use
    /// [`ComplexScalar::hesv`]).
    #[allow(clippy::too_many_arguments)]
    fn sysv(
        layout: Layout,
        uplo: Uplo,
        n: usize,
        nrhs: usize,
        a: &mut [Self],
        lda: usize,
        ipiv: &mut [i32],
        b: &mut [Self],
        ldb: usize,
    ) -> Result<()>;

    /// Bunch-Kaufman factorization.
    fn sytrf(
        layout: Layout,
        uplo: Uplo,
        n: usize,
        a: &mut [Self],
        lda: usize,
        ipiv: &mut [i32],
    ) -> Result<()>;

    /// Back-substitute using a prior Bunch-Kaufman factorization.
    #[allow(clippy::too_many_arguments)]
    fn sytrs(
        layout: Layout,
        uplo: Uplo,
        n: usize,
        nrhs: usize,
        a: &[Self],
        lda: usize,
        ipiv: &[i32],
        b: &mut [Self],
        ldb: usize,
    ) -> Result<()>;

    // --- Banded ----------------------------------------------------------

    /// Solve `A · X = B` for general banded `A`.
    #[allow(clippy::too_many_arguments)]
    fn gbsv(
        layout: Layout,
        n: usize,
        kl: usize,
        ku: usize,
        nrhs: usize,
        ab: &mut [Self],
        ldab: usize,
        ipiv: &mut [i32],
        b: &mut [Self],
        ldb: usize,
    ) -> Result<()>;

    /// Banded LU factorization.
    #[allow(clippy::too_many_arguments)]
    fn gbtrf(
        layout: Layout,
        m: usize,
        n: usize,
        kl: usize,
        ku: usize,
        ab: &mut [Self],
        ldab: usize,
        ipiv: &mut [i32],
    ) -> Result<()>;

    /// Back-substitute using a prior banded LU factorization.
    #[allow(clippy::too_many_arguments)]
    fn gbtrs(
        layout: Layout,
        trans: Trans,
        n: usize,
        kl: usize,
        ku: usize,
        nrhs: usize,
        ab: &[Self],
        ldab: usize,
        ipiv: &[i32],
        b: &mut [Self],
        ldb: usize,
    ) -> Result<()>;

    // --- Tridiagonal -----------------------------------------------------

    /// Solve `A · X = B` for general tridiagonal `A`.
    #[allow(clippy::too_many_arguments)]
    fn gtsv(
        layout: Layout,
        n: usize,
        nrhs: usize,
        dl: &mut [Self],
        d: &mut [Self],
        du: &mut [Self],
        b: &mut [Self],
        ldb: usize,
    ) -> Result<()>;

    /// Solve `A · X = B` for symmetric/Hermitian positive-definite
    /// tridiagonal `A`. The main diagonal `d` is **real**; `e` is the
    /// (sub-/super-)diagonal whose precision matches `Self`.
    #[allow(clippy::too_many_arguments)]
    fn ptsv(
        layout: Layout,
        n: usize,
        nrhs: usize,
        d: &mut [Self::Real],
        e: &mut [Self],
        b: &mut [Self],
        ldb: usize,
    ) -> Result<()>;

    // --- Least squares ---------------------------------------------------

    /// Solve `min || A·X − B ||₂` (or transposed equivalents).
    #[allow(clippy::too_many_arguments)]
    fn gels(
        layout: Layout,
        trans: Trans,
        m: usize,
        n: usize,
        nrhs: usize,
        a: &mut [Self],
        lda: usize,
        b: &mut [Self],
        ldb: usize,
    ) -> Result<()>;

    // --- QR / LQ ---------------------------------------------------------

    /// QR factorization.
    fn geqrf(
        layout: Layout,
        m: usize,
        n: usize,
        a: &mut [Self],
        lda: usize,
        tau: &mut [Self],
    ) -> Result<()>;

    /// LQ factorization.
    fn gelqf(
        layout: Layout,
        m: usize,
        n: usize,
        a: &mut [Self],
        lda: usize,
        tau: &mut [Self],
    ) -> Result<()>;

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
    fn hesv(
        layout: Layout,
        uplo: Uplo,
        n: usize,
        nrhs: usize,
        a: &mut [Self],
        lda: usize,
        ipiv: &mut [i32],
        b: &mut [Self],
        ldb: usize,
    ) -> Result<()>;

    /// Hermitian Bunch-Kaufman factorization.
    fn hetrf(
        layout: Layout,
        uplo: Uplo,
        n: usize,
        a: &mut [Self],
        lda: usize,
        ipiv: &mut [i32],
    ) -> Result<()>;

    /// Back-substitute using a Hermitian Bunch-Kaufman factorization.
    #[allow(clippy::too_many_arguments)]
    fn hetrs(
        layout: Layout,
        uplo: Uplo,
        n: usize,
        nrhs: usize,
        a: &[Self],
        lda: usize,
        ipiv: &[i32],
        b: &mut [Self],
        ldb: usize,
    ) -> Result<()>;

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

            fn gesv(
                layout: Layout,
                n: usize,
                nrhs: usize,
                a: &mut [Self],
                lda: usize,
                ipiv: &mut [i32],
                b: &mut [Self],
                ldb: usize,
            ) -> Result<()> {
                check_matrix("gesv: A", layout, n, n, lda, a.len())?;
                check_matrix("gesv: B", layout, n, nrhs, ldb, b.len())?;
                if ipiv.len() < n {
                    return Err(Error::InvalidArgument(format!(
                        "gesv: ipiv length {} < n={n}",
                        ipiv.len()
                    )));
                }
                let info = unsafe {
                    sys::$gesv(
                        layout_raw(layout),
                        n as i32,
                        nrhs as i32,
                        a.as_mut_ptr(),
                        lda as i32,
                        ipiv.as_mut_ptr(),
                        b.as_mut_ptr(),
                        ldb as i32,
                    )
                };
                map_info("lapack", info)
            }

            fn getrf(
                layout: Layout,
                m: usize,
                n: usize,
                a: &mut [Self],
                lda: usize,
                ipiv: &mut [i32],
            ) -> Result<()> {
                check_matrix("getrf: A", layout, m, n, lda, a.len())?;
                if ipiv.len() < m.min(n) {
                    return Err(Error::InvalidArgument(format!(
                        "getrf: ipiv length {} < min(m,n)={}",
                        ipiv.len(),
                        m.min(n)
                    )));
                }
                let info = unsafe {
                    sys::$getrf(
                        layout_raw(layout),
                        m as i32,
                        n as i32,
                        a.as_mut_ptr(),
                        lda as i32,
                        ipiv.as_mut_ptr(),
                    )
                };
                map_info("lapack", info)
            }

            fn getrs(
                layout: Layout,
                trans: Trans,
                n: usize,
                nrhs: usize,
                a: &[Self],
                lda: usize,
                ipiv: &[i32],
                b: &mut [Self],
                ldb: usize,
            ) -> Result<()> {
                check_matrix("getrs: A", layout, n, n, lda, a.len())?;
                check_matrix("getrs: B", layout, n, nrhs, ldb, b.len())?;
                let info = unsafe {
                    sys::$getrs(
                        layout_raw(layout),
                        trans_char(trans),
                        n as i32,
                        nrhs as i32,
                        a.as_ptr(),
                        lda as i32,
                        ipiv.as_ptr(),
                        b.as_mut_ptr(),
                        ldb as i32,
                    )
                };
                map_info("lapack", info)
            }

            fn getri(
                layout: Layout,
                n: usize,
                a: &mut [Self],
                lda: usize,
                ipiv: &[i32],
            ) -> Result<()> {
                check_matrix("getri: A", layout, n, n, lda, a.len())?;
                let info = unsafe {
                    sys::$getri(
                        layout_raw(layout),
                        n as i32,
                        a.as_mut_ptr(),
                        lda as i32,
                        ipiv.as_ptr(),
                    )
                };
                map_info("lapack", info)
            }

            fn posv(
                layout: Layout,
                uplo: Uplo,
                n: usize,
                nrhs: usize,
                a: &mut [Self],
                lda: usize,
                b: &mut [Self],
                ldb: usize,
            ) -> Result<()> {
                check_matrix("posv: A", layout, n, n, lda, a.len())?;
                check_matrix("posv: B", layout, n, nrhs, ldb, b.len())?;
                let info = unsafe {
                    sys::$posv(
                        layout_raw(layout),
                        uplo_char(uplo),
                        n as i32,
                        nrhs as i32,
                        a.as_mut_ptr(),
                        lda as i32,
                        b.as_mut_ptr(),
                        ldb as i32,
                    )
                };
                map_info("lapack", info)
            }

            fn potrf(
                layout: Layout,
                uplo: Uplo,
                n: usize,
                a: &mut [Self],
                lda: usize,
            ) -> Result<()> {
                check_matrix("potrf: A", layout, n, n, lda, a.len())?;
                let info = unsafe {
                    sys::$potrf(
                        layout_raw(layout),
                        uplo_char(uplo),
                        n as i32,
                        a.as_mut_ptr(),
                        lda as i32,
                    )
                };
                map_info("lapack", info)
            }

            fn potrs(
                layout: Layout,
                uplo: Uplo,
                n: usize,
                nrhs: usize,
                a: &[Self],
                lda: usize,
                b: &mut [Self],
                ldb: usize,
            ) -> Result<()> {
                check_matrix("potrs: A", layout, n, n, lda, a.len())?;
                check_matrix("potrs: B", layout, n, nrhs, ldb, b.len())?;
                let info = unsafe {
                    sys::$potrs(
                        layout_raw(layout),
                        uplo_char(uplo),
                        n as i32,
                        nrhs as i32,
                        a.as_ptr(),
                        lda as i32,
                        b.as_mut_ptr(),
                        ldb as i32,
                    )
                };
                map_info("lapack", info)
            }

            fn potri(
                layout: Layout,
                uplo: Uplo,
                n: usize,
                a: &mut [Self],
                lda: usize,
            ) -> Result<()> {
                check_matrix("potri: A", layout, n, n, lda, a.len())?;
                let info = unsafe {
                    sys::$potri(
                        layout_raw(layout),
                        uplo_char(uplo),
                        n as i32,
                        a.as_mut_ptr(),
                        lda as i32,
                    )
                };
                map_info("lapack", info)
            }

            fn sysv(
                layout: Layout,
                uplo: Uplo,
                n: usize,
                nrhs: usize,
                a: &mut [Self],
                lda: usize,
                ipiv: &mut [i32],
                b: &mut [Self],
                ldb: usize,
            ) -> Result<()> {
                check_matrix("sysv: A", layout, n, n, lda, a.len())?;
                check_matrix("sysv: B", layout, n, nrhs, ldb, b.len())?;
                let info = unsafe {
                    sys::$sysv(
                        layout_raw(layout),
                        uplo_char(uplo),
                        n as i32,
                        nrhs as i32,
                        a.as_mut_ptr(),
                        lda as i32,
                        ipiv.as_mut_ptr(),
                        b.as_mut_ptr(),
                        ldb as i32,
                    )
                };
                map_info("lapack", info)
            }

            fn sytrf(
                layout: Layout,
                uplo: Uplo,
                n: usize,
                a: &mut [Self],
                lda: usize,
                ipiv: &mut [i32],
            ) -> Result<()> {
                let info = unsafe {
                    sys::$sytrf(
                        layout_raw(layout),
                        uplo_char(uplo),
                        n as i32,
                        a.as_mut_ptr(),
                        lda as i32,
                        ipiv.as_mut_ptr(),
                    )
                };
                map_info("lapack", info)
            }

            fn sytrs(
                layout: Layout,
                uplo: Uplo,
                n: usize,
                nrhs: usize,
                a: &[Self],
                lda: usize,
                ipiv: &[i32],
                b: &mut [Self],
                ldb: usize,
            ) -> Result<()> {
                let info = unsafe {
                    sys::$sytrs(
                        layout_raw(layout),
                        uplo_char(uplo),
                        n as i32,
                        nrhs as i32,
                        a.as_ptr(),
                        lda as i32,
                        ipiv.as_ptr(),
                        b.as_mut_ptr(),
                        ldb as i32,
                    )
                };
                map_info("lapack", info)
            }

            fn gbsv(
                layout: Layout,
                n: usize,
                kl: usize,
                ku: usize,
                nrhs: usize,
                ab: &mut [Self],
                ldab: usize,
                ipiv: &mut [i32],
                b: &mut [Self],
                ldb: usize,
            ) -> Result<()> {
                let info = unsafe {
                    sys::$gbsv(
                        layout_raw(layout),
                        n as i32,
                        kl as i32,
                        ku as i32,
                        nrhs as i32,
                        ab.as_mut_ptr(),
                        ldab as i32,
                        ipiv.as_mut_ptr(),
                        b.as_mut_ptr(),
                        ldb as i32,
                    )
                };
                map_info("lapack", info)
            }

            fn gbtrf(
                layout: Layout,
                m: usize,
                n: usize,
                kl: usize,
                ku: usize,
                ab: &mut [Self],
                ldab: usize,
                ipiv: &mut [i32],
            ) -> Result<()> {
                let info = unsafe {
                    sys::$gbtrf(
                        layout_raw(layout),
                        m as i32,
                        n as i32,
                        kl as i32,
                        ku as i32,
                        ab.as_mut_ptr(),
                        ldab as i32,
                        ipiv.as_mut_ptr(),
                    )
                };
                map_info("lapack", info)
            }

            fn gbtrs(
                layout: Layout,
                trans: Trans,
                n: usize,
                kl: usize,
                ku: usize,
                nrhs: usize,
                ab: &[Self],
                ldab: usize,
                ipiv: &[i32],
                b: &mut [Self],
                ldb: usize,
            ) -> Result<()> {
                let info = unsafe {
                    sys::$gbtrs(
                        layout_raw(layout),
                        trans_char(trans),
                        n as i32,
                        kl as i32,
                        ku as i32,
                        nrhs as i32,
                        ab.as_ptr(),
                        ldab as i32,
                        ipiv.as_ptr(),
                        b.as_mut_ptr(),
                        ldb as i32,
                    )
                };
                map_info("lapack", info)
            }

            fn gtsv(
                layout: Layout,
                n: usize,
                nrhs: usize,
                dl: &mut [Self],
                d: &mut [Self],
                du: &mut [Self],
                b: &mut [Self],
                ldb: usize,
            ) -> Result<()> {
                if n > 0 {
                    if d.len() < n {
                        return Err(Error::InvalidArgument(format!(
                            "gtsv: d length {} < n={n}",
                            d.len()
                        )));
                    }
                    if dl.len() < n - 1 || du.len() < n - 1 {
                        return Err(Error::InvalidArgument(format!(
                            "gtsv: dl/du length must be at least n-1 = {}",
                            n - 1
                        )));
                    }
                }
                let info = unsafe {
                    sys::$gtsv(
                        layout_raw(layout),
                        n as i32,
                        nrhs as i32,
                        dl.as_mut_ptr(),
                        d.as_mut_ptr(),
                        du.as_mut_ptr(),
                        b.as_mut_ptr(),
                        ldb as i32,
                    )
                };
                map_info("lapack", info)
            }

            fn ptsv(
                layout: Layout,
                n: usize,
                nrhs: usize,
                d: &mut [Self::Real],
                e: &mut [Self],
                b: &mut [Self],
                ldb: usize,
            ) -> Result<()> {
                let info = unsafe {
                    sys::$ptsv(
                        layout_raw(layout),
                        n as i32,
                        nrhs as i32,
                        d.as_mut_ptr(),
                        e.as_mut_ptr(),
                        b.as_mut_ptr(),
                        ldb as i32,
                    )
                };
                map_info("lapack", info)
            }

            fn gels(
                layout: Layout,
                trans: Trans,
                m: usize,
                n: usize,
                nrhs: usize,
                a: &mut [Self],
                lda: usize,
                b: &mut [Self],
                ldb: usize,
            ) -> Result<()> {
                let info = unsafe {
                    sys::$gels(
                        layout_raw(layout),
                        trans_char(trans),
                        m as i32,
                        n as i32,
                        nrhs as i32,
                        a.as_mut_ptr(),
                        lda as i32,
                        b.as_mut_ptr(),
                        ldb as i32,
                    )
                };
                map_info("lapack", info)
            }

            fn geqrf(
                layout: Layout,
                m: usize,
                n: usize,
                a: &mut [Self],
                lda: usize,
                tau: &mut [Self],
            ) -> Result<()> {
                check_matrix("geqrf: A", layout, m, n, lda, a.len())?;
                if tau.len() < m.min(n) {
                    return Err(Error::InvalidArgument(format!(
                        "geqrf: tau length {} < min(m,n)={}",
                        tau.len(),
                        m.min(n)
                    )));
                }
                let info = unsafe {
                    sys::$geqrf(
                        layout_raw(layout),
                        m as i32,
                        n as i32,
                        a.as_mut_ptr(),
                        lda as i32,
                        tau.as_mut_ptr(),
                    )
                };
                map_info("lapack", info)
            }

            fn gelqf(
                layout: Layout,
                m: usize,
                n: usize,
                a: &mut [Self],
                lda: usize,
                tau: &mut [Self],
            ) -> Result<()> {
                check_matrix("gelqf: A", layout, m, n, lda, a.len())?;
                if tau.len() < m.min(n) {
                    return Err(Error::InvalidArgument(format!(
                        "gelqf: tau length {} < min(m,n)={}",
                        tau.len(),
                        m.min(n)
                    )));
                }
                let info = unsafe {
                    sys::$gelqf(
                        layout_raw(layout),
                        m as i32,
                        n as i32,
                        a.as_mut_ptr(),
                        lda as i32,
                        tau.as_mut_ptr(),
                    )
                };
                map_info("lapack", info)
            }

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
            ) -> Result<()> {
                let info = unsafe {
                    sys::$gesvd(
                        layout_raw(layout),
                        jobu.job_char(),
                        jobvt.job_char(),
                        m as i32,
                        n as i32,
                        a.as_mut_ptr(),
                        lda as i32,
                        s.as_mut_ptr(),
                        u.as_mut_ptr(),
                        ldu as i32,
                        vt.as_mut_ptr(),
                        ldvt as i32,
                        superb.as_mut_ptr(),
                    )
                };
                map_info("lapack", info)
            }

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
            ) -> Result<()> {
                let info = unsafe {
                    sys::$gesdd(
                        layout_raw(layout),
                        jobz.job_char(),
                        m as i32,
                        n as i32,
                        a.as_mut_ptr(),
                        lda as i32,
                        s.as_mut_ptr(),
                        u.as_mut_ptr(),
                        ldu as i32,
                        vt.as_mut_ptr(),
                        ldvt as i32,
                    )
                };
                map_info("lapack", info)
            }

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
            ) -> Result<()> {
                let info = unsafe {
                    sys::$gelsd(
                        layout_raw(layout),
                        m as i32,
                        n as i32,
                        nrhs as i32,
                        a.as_mut_ptr(),
                        lda as i32,
                        b.as_mut_ptr(),
                        ldb as i32,
                        s.as_mut_ptr(),
                        rcond,
                        rank,
                    )
                };
                map_info("lapack", info)
            }

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
            ) -> Result<()> {
                let info = unsafe {
                    sys::$gelsy(
                        layout_raw(layout),
                        m as i32,
                        n as i32,
                        nrhs as i32,
                        a.as_mut_ptr(),
                        lda as i32,
                        b.as_mut_ptr(),
                        ldb as i32,
                        jpvt.as_mut_ptr(),
                        rcond,
                        rank,
                    )
                };
                map_info("lapack", info)
            }
        }

        impl RealScalar for $t {
            #[allow(clippy::too_many_arguments)]
            fn syev(
                layout: Layout,
                compute: Compute,
                uplo: Uplo,
                n: usize,
                a: &mut [Self],
                lda: usize,
                w: &mut [Self],
            ) -> Result<()> {
                let info = unsafe {
                    sys::$syev(
                        layout_raw(layout),
                        compute.job_char(),
                        uplo_char(uplo),
                        n as i32,
                        a.as_mut_ptr(),
                        lda as i32,
                        w.as_mut_ptr(),
                    )
                };
                map_info("lapack", info)
            }

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
            ) -> Result<()> {
                let info = unsafe {
                    sys::$geev(
                        layout_raw(layout),
                        compute_left.job_char(),
                        compute_right.job_char(),
                        n as i32,
                        a.as_mut_ptr(),
                        lda as i32,
                        wr.as_mut_ptr(),
                        wi.as_mut_ptr(),
                        vl.as_mut_ptr(),
                        ldvl as i32,
                        vr.as_mut_ptr(),
                        ldvr as i32,
                    )
                };
                map_info("lapack", info)
            }
        }
    };
}

impl_real!(
    f32,
    gesv = LAPACKE_sgesv,
    getrf = LAPACKE_sgetrf,
    getrs = LAPACKE_sgetrs,
    getri = LAPACKE_sgetri,
    posv = LAPACKE_sposv,
    potrf = LAPACKE_spotrf,
    potrs = LAPACKE_spotrs,
    potri = LAPACKE_spotri,
    sysv = LAPACKE_ssysv,
    sytrf = LAPACKE_ssytrf,
    sytrs = LAPACKE_ssytrs,
    gbsv = LAPACKE_sgbsv,
    gbtrf = LAPACKE_sgbtrf,
    gbtrs = LAPACKE_sgbtrs,
    gtsv = LAPACKE_sgtsv,
    ptsv = LAPACKE_sptsv,
    gels = LAPACKE_sgels,
    geqrf = LAPACKE_sgeqrf,
    gelqf = LAPACKE_sgelqf,
    syev = LAPACKE_ssyev,
    geev = LAPACKE_sgeev,
    gesvd = LAPACKE_sgesvd,
    gesdd = LAPACKE_sgesdd,
    gelsd = LAPACKE_sgelsd,
    gelsy = LAPACKE_sgelsy
);
impl_real!(
    f64,
    gesv = LAPACKE_dgesv,
    getrf = LAPACKE_dgetrf,
    getrs = LAPACKE_dgetrs,
    getri = LAPACKE_dgetri,
    posv = LAPACKE_dposv,
    potrf = LAPACKE_dpotrf,
    potrs = LAPACKE_dpotrs,
    potri = LAPACKE_dpotri,
    sysv = LAPACKE_dsysv,
    sytrf = LAPACKE_dsytrf,
    sytrs = LAPACKE_dsytrs,
    gbsv = LAPACKE_dgbsv,
    gbtrf = LAPACKE_dgbtrf,
    gbtrs = LAPACKE_dgbtrs,
    gtsv = LAPACKE_dgtsv,
    ptsv = LAPACKE_dptsv,
    gels = LAPACKE_dgels,
    geqrf = LAPACKE_dgeqrf,
    gelqf = LAPACKE_dgelqf,
    syev = LAPACKE_dsyev,
    geev = LAPACKE_dgeev,
    gesvd = LAPACKE_dgesvd,
    gesdd = LAPACKE_dgesdd,
    gelsd = LAPACKE_dgelsd,
    gelsy = LAPACKE_dgelsy
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

            fn gesv(
                layout: Layout,
                n: usize,
                nrhs: usize,
                a: &mut [Self],
                lda: usize,
                ipiv: &mut [i32],
                b: &mut [Self],
                ldb: usize,
            ) -> Result<()> {
                check_matrix("gesv: A", layout, n, n, lda, a.len())?;
                check_matrix("gesv: B", layout, n, nrhs, ldb, b.len())?;
                if ipiv.len() < n {
                    return Err(Error::InvalidArgument(format!(
                        "gesv: ipiv length {} < n={n}",
                        ipiv.len()
                    )));
                }
                let info = unsafe {
                    sys::$gesv(
                        layout_raw(layout),
                        n as i32,
                        nrhs as i32,
                        a.as_mut_ptr() as *mut $cc,
                        lda as i32,
                        ipiv.as_mut_ptr(),
                        b.as_mut_ptr() as *mut $cc,
                        ldb as i32,
                    )
                };
                map_info("lapack", info)
            }

            fn getrf(
                layout: Layout,
                m: usize,
                n: usize,
                a: &mut [Self],
                lda: usize,
                ipiv: &mut [i32],
            ) -> Result<()> {
                check_matrix("getrf: A", layout, m, n, lda, a.len())?;
                let info = unsafe {
                    sys::$getrf(
                        layout_raw(layout),
                        m as i32,
                        n as i32,
                        a.as_mut_ptr() as *mut $cc,
                        lda as i32,
                        ipiv.as_mut_ptr(),
                    )
                };
                map_info("lapack", info)
            }

            fn getrs(
                layout: Layout,
                trans: Trans,
                n: usize,
                nrhs: usize,
                a: &[Self],
                lda: usize,
                ipiv: &[i32],
                b: &mut [Self],
                ldb: usize,
            ) -> Result<()> {
                let info = unsafe {
                    sys::$getrs(
                        layout_raw(layout),
                        trans_char(trans),
                        n as i32,
                        nrhs as i32,
                        a.as_ptr() as *const $cc,
                        lda as i32,
                        ipiv.as_ptr(),
                        b.as_mut_ptr() as *mut $cc,
                        ldb as i32,
                    )
                };
                map_info("lapack", info)
            }

            fn getri(
                layout: Layout,
                n: usize,
                a: &mut [Self],
                lda: usize,
                ipiv: &[i32],
            ) -> Result<()> {
                let info = unsafe {
                    sys::$getri(
                        layout_raw(layout),
                        n as i32,
                        a.as_mut_ptr() as *mut $cc,
                        lda as i32,
                        ipiv.as_ptr(),
                    )
                };
                map_info("lapack", info)
            }

            fn posv(
                layout: Layout,
                uplo: Uplo,
                n: usize,
                nrhs: usize,
                a: &mut [Self],
                lda: usize,
                b: &mut [Self],
                ldb: usize,
            ) -> Result<()> {
                let info = unsafe {
                    sys::$posv(
                        layout_raw(layout),
                        uplo_char(uplo),
                        n as i32,
                        nrhs as i32,
                        a.as_mut_ptr() as *mut $cc,
                        lda as i32,
                        b.as_mut_ptr() as *mut $cc,
                        ldb as i32,
                    )
                };
                map_info("lapack", info)
            }

            fn potrf(
                layout: Layout,
                uplo: Uplo,
                n: usize,
                a: &mut [Self],
                lda: usize,
            ) -> Result<()> {
                let info = unsafe {
                    sys::$potrf(
                        layout_raw(layout),
                        uplo_char(uplo),
                        n as i32,
                        a.as_mut_ptr() as *mut $cc,
                        lda as i32,
                    )
                };
                map_info("lapack", info)
            }

            fn potrs(
                layout: Layout,
                uplo: Uplo,
                n: usize,
                nrhs: usize,
                a: &[Self],
                lda: usize,
                b: &mut [Self],
                ldb: usize,
            ) -> Result<()> {
                let info = unsafe {
                    sys::$potrs(
                        layout_raw(layout),
                        uplo_char(uplo),
                        n as i32,
                        nrhs as i32,
                        a.as_ptr() as *const $cc,
                        lda as i32,
                        b.as_mut_ptr() as *mut $cc,
                        ldb as i32,
                    )
                };
                map_info("lapack", info)
            }

            fn potri(
                layout: Layout,
                uplo: Uplo,
                n: usize,
                a: &mut [Self],
                lda: usize,
            ) -> Result<()> {
                let info = unsafe {
                    sys::$potri(
                        layout_raw(layout),
                        uplo_char(uplo),
                        n as i32,
                        a.as_mut_ptr() as *mut $cc,
                        lda as i32,
                    )
                };
                map_info("lapack", info)
            }

            fn sysv(
                layout: Layout,
                uplo: Uplo,
                n: usize,
                nrhs: usize,
                a: &mut [Self],
                lda: usize,
                ipiv: &mut [i32],
                b: &mut [Self],
                ldb: usize,
            ) -> Result<()> {
                let info = unsafe {
                    sys::$sysv(
                        layout_raw(layout),
                        uplo_char(uplo),
                        n as i32,
                        nrhs as i32,
                        a.as_mut_ptr() as *mut $cc,
                        lda as i32,
                        ipiv.as_mut_ptr(),
                        b.as_mut_ptr() as *mut $cc,
                        ldb as i32,
                    )
                };
                map_info("lapack", info)
            }

            fn sytrf(
                layout: Layout,
                uplo: Uplo,
                n: usize,
                a: &mut [Self],
                lda: usize,
                ipiv: &mut [i32],
            ) -> Result<()> {
                let info = unsafe {
                    sys::$sytrf(
                        layout_raw(layout),
                        uplo_char(uplo),
                        n as i32,
                        a.as_mut_ptr() as *mut $cc,
                        lda as i32,
                        ipiv.as_mut_ptr(),
                    )
                };
                map_info("lapack", info)
            }

            fn sytrs(
                layout: Layout,
                uplo: Uplo,
                n: usize,
                nrhs: usize,
                a: &[Self],
                lda: usize,
                ipiv: &[i32],
                b: &mut [Self],
                ldb: usize,
            ) -> Result<()> {
                let info = unsafe {
                    sys::$sytrs(
                        layout_raw(layout),
                        uplo_char(uplo),
                        n as i32,
                        nrhs as i32,
                        a.as_ptr() as *const $cc,
                        lda as i32,
                        ipiv.as_ptr(),
                        b.as_mut_ptr() as *mut $cc,
                        ldb as i32,
                    )
                };
                map_info("lapack", info)
            }

            fn gbsv(
                layout: Layout,
                n: usize,
                kl: usize,
                ku: usize,
                nrhs: usize,
                ab: &mut [Self],
                ldab: usize,
                ipiv: &mut [i32],
                b: &mut [Self],
                ldb: usize,
            ) -> Result<()> {
                let info = unsafe {
                    sys::$gbsv(
                        layout_raw(layout),
                        n as i32,
                        kl as i32,
                        ku as i32,
                        nrhs as i32,
                        ab.as_mut_ptr() as *mut $cc,
                        ldab as i32,
                        ipiv.as_mut_ptr(),
                        b.as_mut_ptr() as *mut $cc,
                        ldb as i32,
                    )
                };
                map_info("lapack", info)
            }

            fn gbtrf(
                layout: Layout,
                m: usize,
                n: usize,
                kl: usize,
                ku: usize,
                ab: &mut [Self],
                ldab: usize,
                ipiv: &mut [i32],
            ) -> Result<()> {
                let info = unsafe {
                    sys::$gbtrf(
                        layout_raw(layout),
                        m as i32,
                        n as i32,
                        kl as i32,
                        ku as i32,
                        ab.as_mut_ptr() as *mut $cc,
                        ldab as i32,
                        ipiv.as_mut_ptr(),
                    )
                };
                map_info("lapack", info)
            }

            fn gbtrs(
                layout: Layout,
                trans: Trans,
                n: usize,
                kl: usize,
                ku: usize,
                nrhs: usize,
                ab: &[Self],
                ldab: usize,
                ipiv: &[i32],
                b: &mut [Self],
                ldb: usize,
            ) -> Result<()> {
                let info = unsafe {
                    sys::$gbtrs(
                        layout_raw(layout),
                        trans_char(trans),
                        n as i32,
                        kl as i32,
                        ku as i32,
                        nrhs as i32,
                        ab.as_ptr() as *const $cc,
                        ldab as i32,
                        ipiv.as_ptr(),
                        b.as_mut_ptr() as *mut $cc,
                        ldb as i32,
                    )
                };
                map_info("lapack", info)
            }

            fn gtsv(
                layout: Layout,
                n: usize,
                nrhs: usize,
                dl: &mut [Self],
                d: &mut [Self],
                du: &mut [Self],
                b: &mut [Self],
                ldb: usize,
            ) -> Result<()> {
                let info = unsafe {
                    sys::$gtsv(
                        layout_raw(layout),
                        n as i32,
                        nrhs as i32,
                        dl.as_mut_ptr() as *mut $cc,
                        d.as_mut_ptr() as *mut $cc,
                        du.as_mut_ptr() as *mut $cc,
                        b.as_mut_ptr() as *mut $cc,
                        ldb as i32,
                    )
                };
                map_info("lapack", info)
            }

            fn ptsv(
                layout: Layout,
                n: usize,
                nrhs: usize,
                d: &mut [Self::Real],
                e: &mut [Self],
                b: &mut [Self],
                ldb: usize,
            ) -> Result<()> {
                let info = unsafe {
                    sys::$ptsv(
                        layout_raw(layout),
                        n as i32,
                        nrhs as i32,
                        d.as_mut_ptr(),
                        e.as_mut_ptr() as *mut $cc,
                        b.as_mut_ptr() as *mut $cc,
                        ldb as i32,
                    )
                };
                map_info("lapack", info)
            }

            fn gels(
                layout: Layout,
                trans: Trans,
                m: usize,
                n: usize,
                nrhs: usize,
                a: &mut [Self],
                lda: usize,
                b: &mut [Self],
                ldb: usize,
            ) -> Result<()> {
                let info = unsafe {
                    sys::$gels(
                        layout_raw(layout),
                        trans_char(trans),
                        m as i32,
                        n as i32,
                        nrhs as i32,
                        a.as_mut_ptr() as *mut $cc,
                        lda as i32,
                        b.as_mut_ptr() as *mut $cc,
                        ldb as i32,
                    )
                };
                map_info("lapack", info)
            }

            fn geqrf(
                layout: Layout,
                m: usize,
                n: usize,
                a: &mut [Self],
                lda: usize,
                tau: &mut [Self],
            ) -> Result<()> {
                let info = unsafe {
                    sys::$geqrf(
                        layout_raw(layout),
                        m as i32,
                        n as i32,
                        a.as_mut_ptr() as *mut $cc,
                        lda as i32,
                        tau.as_mut_ptr() as *mut $cc,
                    )
                };
                map_info("lapack", info)
            }

            fn gelqf(
                layout: Layout,
                m: usize,
                n: usize,
                a: &mut [Self],
                lda: usize,
                tau: &mut [Self],
            ) -> Result<()> {
                let info = unsafe {
                    sys::$gelqf(
                        layout_raw(layout),
                        m as i32,
                        n as i32,
                        a.as_mut_ptr() as *mut $cc,
                        lda as i32,
                        tau.as_mut_ptr() as *mut $cc,
                    )
                };
                map_info("lapack", info)
            }

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
            ) -> Result<()> {
                let info = unsafe {
                    sys::$gesvd(
                        layout_raw(layout),
                        jobu.job_char(),
                        jobvt.job_char(),
                        m as i32,
                        n as i32,
                        a.as_mut_ptr() as *mut $cc,
                        lda as i32,
                        s.as_mut_ptr(),
                        u.as_mut_ptr() as *mut $cc,
                        ldu as i32,
                        vt.as_mut_ptr() as *mut $cc,
                        ldvt as i32,
                        superb.as_mut_ptr(),
                    )
                };
                map_info("lapack", info)
            }

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
            ) -> Result<()> {
                let info = unsafe {
                    sys::$gesdd(
                        layout_raw(layout),
                        jobz.job_char(),
                        m as i32,
                        n as i32,
                        a.as_mut_ptr() as *mut $cc,
                        lda as i32,
                        s.as_mut_ptr(),
                        u.as_mut_ptr() as *mut $cc,
                        ldu as i32,
                        vt.as_mut_ptr() as *mut $cc,
                        ldvt as i32,
                    )
                };
                map_info("lapack", info)
            }

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
            ) -> Result<()> {
                let info = unsafe {
                    sys::$gelsd(
                        layout_raw(layout),
                        m as i32,
                        n as i32,
                        nrhs as i32,
                        a.as_mut_ptr() as *mut $cc,
                        lda as i32,
                        b.as_mut_ptr() as *mut $cc,
                        ldb as i32,
                        s.as_mut_ptr(),
                        rcond,
                        rank,
                    )
                };
                map_info("lapack", info)
            }

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
            ) -> Result<()> {
                let info = unsafe {
                    sys::$gelsy(
                        layout_raw(layout),
                        m as i32,
                        n as i32,
                        nrhs as i32,
                        a.as_mut_ptr() as *mut $cc,
                        lda as i32,
                        b.as_mut_ptr() as *mut $cc,
                        ldb as i32,
                        jpvt.as_mut_ptr(),
                        rcond,
                        rank,
                    )
                };
                map_info("lapack", info)
            }
        }

        impl ComplexScalar for $t {
            fn hesv(
                layout: Layout,
                uplo: Uplo,
                n: usize,
                nrhs: usize,
                a: &mut [Self],
                lda: usize,
                ipiv: &mut [i32],
                b: &mut [Self],
                ldb: usize,
            ) -> Result<()> {
                let info = unsafe {
                    sys::$hesv(
                        layout_raw(layout),
                        uplo_char(uplo),
                        n as i32,
                        nrhs as i32,
                        a.as_mut_ptr() as *mut $cc,
                        lda as i32,
                        ipiv.as_mut_ptr(),
                        b.as_mut_ptr() as *mut $cc,
                        ldb as i32,
                    )
                };
                map_info("lapack", info)
            }

            fn hetrf(
                layout: Layout,
                uplo: Uplo,
                n: usize,
                a: &mut [Self],
                lda: usize,
                ipiv: &mut [i32],
            ) -> Result<()> {
                let info = unsafe {
                    sys::$hetrf(
                        layout_raw(layout),
                        uplo_char(uplo),
                        n as i32,
                        a.as_mut_ptr() as *mut $cc,
                        lda as i32,
                        ipiv.as_mut_ptr(),
                    )
                };
                map_info("lapack", info)
            }

            fn hetrs(
                layout: Layout,
                uplo: Uplo,
                n: usize,
                nrhs: usize,
                a: &[Self],
                lda: usize,
                ipiv: &[i32],
                b: &mut [Self],
                ldb: usize,
            ) -> Result<()> {
                let info = unsafe {
                    sys::$hetrs(
                        layout_raw(layout),
                        uplo_char(uplo),
                        n as i32,
                        nrhs as i32,
                        a.as_ptr() as *const $cc,
                        lda as i32,
                        ipiv.as_ptr(),
                        b.as_mut_ptr() as *mut $cc,
                        ldb as i32,
                    )
                };
                map_info("lapack", info)
            }

            #[allow(clippy::too_many_arguments)]
            fn heev(
                layout: Layout,
                compute: Compute,
                uplo: Uplo,
                n: usize,
                a: &mut [Self],
                lda: usize,
                w: &mut [Self::Real],
            ) -> Result<()> {
                let info = unsafe {
                    sys::$heev(
                        layout_raw(layout),
                        compute.job_char(),
                        uplo_char(uplo),
                        n as i32,
                        a.as_mut_ptr() as *mut $cc,
                        lda as i32,
                        w.as_mut_ptr(),
                    )
                };
                map_info("lapack", info)
            }

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
            ) -> Result<()> {
                let info = unsafe {
                    sys::$geev(
                        layout_raw(layout),
                        compute_left.job_char(),
                        compute_right.job_char(),
                        n as i32,
                        a.as_mut_ptr() as *mut $cc,
                        lda as i32,
                        w.as_mut_ptr() as *mut $cc,
                        vl.as_mut_ptr() as *mut $cc,
                        ldvl as i32,
                        vr.as_mut_ptr() as *mut $cc,
                        ldvr as i32,
                    )
                };
                map_info("lapack", info)
            }
        }
    };
}

impl_complex!(
    Complex32,
    f32,
    CC32,
    gesv = LAPACKE_cgesv,
    getrf = LAPACKE_cgetrf,
    getrs = LAPACKE_cgetrs,
    getri = LAPACKE_cgetri,
    posv = LAPACKE_cposv,
    potrf = LAPACKE_cpotrf,
    potrs = LAPACKE_cpotrs,
    potri = LAPACKE_cpotri,
    sysv = LAPACKE_csysv,
    sytrf = LAPACKE_csytrf,
    sytrs = LAPACKE_csytrs,
    hesv = LAPACKE_chesv,
    hetrf = LAPACKE_chetrf,
    hetrs = LAPACKE_chetrs,
    gbsv = LAPACKE_cgbsv,
    gbtrf = LAPACKE_cgbtrf,
    gbtrs = LAPACKE_cgbtrs,
    gtsv = LAPACKE_cgtsv,
    ptsv = LAPACKE_cptsv,
    gels = LAPACKE_cgels,
    geqrf = LAPACKE_cgeqrf,
    gelqf = LAPACKE_cgelqf,
    heev = LAPACKE_cheev,
    geev = LAPACKE_cgeev,
    gesvd = LAPACKE_cgesvd,
    gesdd = LAPACKE_cgesdd,
    gelsd = LAPACKE_cgelsd,
    gelsy = LAPACKE_cgelsy
);
impl_complex!(
    Complex64,
    f64,
    CC64,
    gesv = LAPACKE_zgesv,
    getrf = LAPACKE_zgetrf,
    getrs = LAPACKE_zgetrs,
    getri = LAPACKE_zgetri,
    posv = LAPACKE_zposv,
    potrf = LAPACKE_zpotrf,
    potrs = LAPACKE_zpotrs,
    potri = LAPACKE_zpotri,
    sysv = LAPACKE_zsysv,
    sytrf = LAPACKE_zsytrf,
    sytrs = LAPACKE_zsytrs,
    hesv = LAPACKE_zhesv,
    hetrf = LAPACKE_zhetrf,
    hetrs = LAPACKE_zhetrs,
    gbsv = LAPACKE_zgbsv,
    gbtrf = LAPACKE_zgbtrf,
    gbtrs = LAPACKE_zgbtrs,
    gtsv = LAPACKE_zgtsv,
    ptsv = LAPACKE_zptsv,
    gels = LAPACKE_zgels,
    geqrf = LAPACKE_zgeqrf,
    gelqf = LAPACKE_zgelqf,
    heev = LAPACKE_zheev,
    geev = LAPACKE_zgeev,
    gesvd = LAPACKE_zgesvd,
    gesdd = LAPACKE_zgesdd,
    gelsd = LAPACKE_zgelsd,
    gelsy = LAPACKE_zgelsy
);

// =========================================================================
//   Free-function convenience entry points (tightly-packed row-major)
// =========================================================================

pub fn gesv<T: Scalar>(n: usize, a: &mut [T], ipiv: &mut [i32], b: &mut [T]) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    let nrhs = b.len() / n;
    T::gesv(Layout::RowMajor, n, nrhs, a, n, ipiv, b, nrhs.max(1))
}

pub fn getrf<T: Scalar>(m: usize, n: usize, a: &mut [T], ipiv: &mut [i32]) -> Result<()> {
    T::getrf(Layout::RowMajor, m, n, a, n, ipiv)
}

pub fn getrs<T: Scalar>(trans: Trans, n: usize, a: &[T], ipiv: &[i32], b: &mut [T]) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    let nrhs = b.len() / n;
    T::getrs(Layout::RowMajor, trans, n, nrhs, a, n, ipiv, b, nrhs.max(1))
}

pub fn getri<T: Scalar>(n: usize, a: &mut [T], ipiv: &[i32]) -> Result<()> {
    T::getri(Layout::RowMajor, n, a, n, ipiv)
}

pub fn posv<T: Scalar>(uplo: Uplo, n: usize, a: &mut [T], b: &mut [T]) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    let nrhs = b.len() / n;
    T::posv(Layout::RowMajor, uplo, n, nrhs, a, n, b, nrhs.max(1))
}

pub fn potrf<T: Scalar>(uplo: Uplo, n: usize, a: &mut [T]) -> Result<()> {
    T::potrf(Layout::RowMajor, uplo, n, a, n)
}

pub fn potrs<T: Scalar>(uplo: Uplo, n: usize, a: &[T], b: &mut [T]) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    let nrhs = b.len() / n;
    T::potrs(Layout::RowMajor, uplo, n, nrhs, a, n, b, nrhs.max(1))
}

pub fn sysv<T: Scalar>(
    uplo: Uplo,
    n: usize,
    a: &mut [T],
    ipiv: &mut [i32],
    b: &mut [T],
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    let nrhs = b.len() / n;
    T::sysv(Layout::RowMajor, uplo, n, nrhs, a, n, ipiv, b, nrhs.max(1))
}

pub fn hesv<T: ComplexScalar>(
    uplo: Uplo,
    n: usize,
    a: &mut [T],
    ipiv: &mut [i32],
    b: &mut [T],
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    let nrhs = b.len() / n;
    T::hesv(Layout::RowMajor, uplo, n, nrhs, a, n, ipiv, b, nrhs.max(1))
}

pub fn gtsv<T: Scalar>(
    n: usize,
    dl: &mut [T],
    d: &mut [T],
    du: &mut [T],
    b: &mut [T],
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    let nrhs = b.len() / n;
    T::gtsv(Layout::RowMajor, n, nrhs, dl, d, du, b, nrhs.max(1))
}

pub fn ptsv<T: Scalar>(n: usize, d: &mut [T::Real], e: &mut [T], b: &mut [T]) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
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
    T::gesvd(
        Layout::RowMajor,
        jobu,
        jobvt,
        m,
        n,
        a,
        n,
        s,
        u,
        ldu,
        vt,
        ldvt,
        superb,
    )
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
    T::gelsd(
        Layout::RowMajor,
        m,
        n,
        nrhs,
        a,
        n,
        b,
        nrhs.max(1),
        s,
        rcond,
        &mut rank,
    )?;
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
    T::gelsy(
        Layout::RowMajor,
        m,
        n,
        nrhs,
        a,
        n,
        b,
        nrhs.max(1),
        jpvt,
        rcond,
        &mut rank,
    )?;
    Ok(rank)
}

// =========================================================================
//   LAPACK extras: triangular ops, factor inverses, generalised eig,
//   non-symmetric eig, QR with column pivoting, matrix norms / copies
// =========================================================================

/// Char passed to LAPACKE for matrix norms: One / Infinity / Frobenius / Max.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Norm {
    /// Maximum absolute value of any matrix element.
    Max,
    /// Maximum column sum (the 1-norm).
    One,
    /// Maximum row sum (the ∞-norm).
    Inf,
    /// Frobenius norm `√(Σ |aᵢⱼ|²)`.
    Frobenius,
}

impl Norm {
    fn raw(self) -> c_char {
        match self {
            Norm::Max => b'M' as c_char,
            Norm::One => b'1' as c_char,
            Norm::Inf => b'I' as c_char,
            Norm::Frobenius => b'F' as c_char,
        }
    }
}

/// Char for `Diag::Unit` / `Diag::NonUnit` triangular flags.
fn diag_char(d: aocl_types::Diag) -> c_char {
    match d {
        aocl_types::Diag::Unit => b'U' as c_char,
        aocl_types::Diag::NonUnit => b'N' as c_char,
    }
}

// ----- Triangular solve / inverse (real and complex) -----------------------

/// Solve `op(A) · X = B` for triangular `A`. (`f64`)
#[allow(clippy::too_many_arguments)]
pub fn dtrtrs(
    uplo: Uplo,
    trans: Trans,
    diag: aocl_types::Diag,
    n: usize,
    nrhs: usize,
    a: &[f64],
    lda: usize,
    b: &mut [f64],
    ldb: usize,
) -> Result<()> {
    let info = unsafe {
        sys::LAPACKE_dtrtrs(
            ROW_MAJOR,
            uplo_char(uplo),
            trans_char(trans),
            diag_char(diag),
            n as i32,
            nrhs as i32,
            a.as_ptr(),
            lda as i32,
            b.as_mut_ptr(),
            ldb as i32,
        )
    };
    map_info("lapack", info)
}

/// `f32` triangular solve. See [`dtrtrs`].
#[allow(clippy::too_many_arguments)]
pub fn strtrs(
    uplo: Uplo,
    trans: Trans,
    diag: aocl_types::Diag,
    n: usize,
    nrhs: usize,
    a: &[f32],
    lda: usize,
    b: &mut [f32],
    ldb: usize,
) -> Result<()> {
    let info = unsafe {
        sys::LAPACKE_strtrs(
            ROW_MAJOR,
            uplo_char(uplo),
            trans_char(trans),
            diag_char(diag),
            n as i32,
            nrhs as i32,
            a.as_ptr(),
            lda as i32,
            b.as_mut_ptr(),
            ldb as i32,
        )
    };
    map_info("lapack", info)
}

/// Compute the inverse of a triangular matrix in place. (`f64`)
pub fn dtrtri(
    uplo: Uplo,
    diag: aocl_types::Diag,
    n: usize,
    a: &mut [f64],
    lda: usize,
) -> Result<()> {
    let info = unsafe {
        sys::LAPACKE_dtrtri(
            ROW_MAJOR,
            uplo_char(uplo),
            diag_char(diag),
            n as i32,
            a.as_mut_ptr(),
            lda as i32,
        )
    };
    map_info("lapack", info)
}

/// `f32` triangular inverse. See [`dtrtri`].
pub fn strtri(
    uplo: Uplo,
    diag: aocl_types::Diag,
    n: usize,
    a: &mut [f32],
    lda: usize,
) -> Result<()> {
    let info = unsafe {
        sys::LAPACKE_strtri(
            ROW_MAJOR,
            uplo_char(uplo),
            diag_char(diag),
            n as i32,
            a.as_mut_ptr(),
            lda as i32,
        )
    };
    map_info("lapack", info)
}

/// Compute the inverse of a positive-definite matrix from its Cholesky
/// factor (already produced by [`potrf`]). (`f64`)
pub fn dpotri(uplo: Uplo, n: usize, a: &mut [f64], lda: usize) -> Result<()> {
    let info = unsafe {
        sys::LAPACKE_dpotri(
            ROW_MAJOR,
            uplo_char(uplo),
            n as i32,
            a.as_mut_ptr(),
            lda as i32,
        )
    };
    map_info("lapack", info)
}

/// `f32` Cholesky inverse. See [`dpotri`].
pub fn spotri(uplo: Uplo, n: usize, a: &mut [f32], lda: usize) -> Result<()> {
    let info = unsafe {
        sys::LAPACKE_spotri(
            ROW_MAJOR,
            uplo_char(uplo),
            n as i32,
            a.as_mut_ptr(),
            lda as i32,
        )
    };
    map_info("lapack", info)
}

// ----- Form / apply Q after QR or LQ ---------------------------------------

/// Form the explicit `Q` factor from elementary reflectors produced by
/// [`geqrf`]. `m × n` with `n ≤ m`, `tau` of length `k = min(m, n)`. (`f64`)
pub fn dorgqr(m: usize, n: usize, k: usize, a: &mut [f64], lda: usize, tau: &[f64]) -> Result<()> {
    let info = unsafe {
        sys::LAPACKE_dorgqr(
            ROW_MAJOR,
            m as i32,
            n as i32,
            k as i32,
            a.as_mut_ptr(),
            lda as i32,
            tau.as_ptr(),
        )
    };
    map_info("lapack", info)
}

/// `f32` `Q` formation. See [`dorgqr`].
pub fn sorgqr(m: usize, n: usize, k: usize, a: &mut [f32], lda: usize, tau: &[f32]) -> Result<()> {
    let info = unsafe {
        sys::LAPACKE_sorgqr(
            ROW_MAJOR,
            m as i32,
            n as i32,
            k as i32,
            a.as_mut_ptr(),
            lda as i32,
            tau.as_ptr(),
        )
    };
    map_info("lapack", info)
}

/// `Complex64` `Q` formation. See [`dorgqr`].
pub fn zungqr(
    m: usize,
    n: usize,
    k: usize,
    a: &mut [Complex64],
    lda: usize,
    tau: &[Complex64],
) -> Result<()> {
    let info = unsafe {
        sys::LAPACKE_zungqr(
            ROW_MAJOR,
            m as i32,
            n as i32,
            k as i32,
            a.as_mut_ptr() as *mut sys::__BindgenComplex<f64>,
            lda as i32,
            tau.as_ptr() as *const sys::__BindgenComplex<f64>,
        )
    };
    map_info("lapack", info)
}

/// `Complex32` `Q` formation. See [`dorgqr`].
pub fn cungqr(
    m: usize,
    n: usize,
    k: usize,
    a: &mut [Complex32],
    lda: usize,
    tau: &[Complex32],
) -> Result<()> {
    let info = unsafe {
        sys::LAPACKE_cungqr(
            ROW_MAJOR,
            m as i32,
            n as i32,
            k as i32,
            a.as_mut_ptr() as *mut sys::__BindgenComplex<f32>,
            lda as i32,
            tau.as_ptr() as *const sys::__BindgenComplex<f32>,
        )
    };
    map_info("lapack", info)
}

// ----- Non-symmetric eigendecomposition (geev) -----------------------------

/// Compute eigenvalues `λᵢ = wr[i] + wi[i]·j` of `A`, optionally with
/// left (`vl`) and/or right (`vr`) eigenvectors. (`f64`)
#[allow(clippy::too_many_arguments)]
pub fn dgeev(
    compute_vl: bool,
    compute_vr: bool,
    n: usize,
    a: &mut [f64],
    lda: usize,
    wr: &mut [f64],
    wi: &mut [f64],
    vl: &mut [f64],
    ldvl: usize,
    vr: &mut [f64],
    ldvr: usize,
) -> Result<()> {
    let jobvl = if compute_vl { b'V' } else { b'N' } as c_char;
    let jobvr = if compute_vr { b'V' } else { b'N' } as c_char;
    let info = unsafe {
        sys::LAPACKE_dgeev(
            ROW_MAJOR,
            jobvl,
            jobvr,
            n as i32,
            a.as_mut_ptr(),
            lda as i32,
            wr.as_mut_ptr(),
            wi.as_mut_ptr(),
            vl.as_mut_ptr(),
            ldvl as i32,
            vr.as_mut_ptr(),
            ldvr as i32,
        )
    };
    map_info("lapack", info)
}

/// `f32` non-symmetric eig. See [`dgeev`].
#[allow(clippy::too_many_arguments)]
pub fn sgeev(
    compute_vl: bool,
    compute_vr: bool,
    n: usize,
    a: &mut [f32],
    lda: usize,
    wr: &mut [f32],
    wi: &mut [f32],
    vl: &mut [f32],
    ldvl: usize,
    vr: &mut [f32],
    ldvr: usize,
) -> Result<()> {
    let jobvl = if compute_vl { b'V' } else { b'N' } as c_char;
    let jobvr = if compute_vr { b'V' } else { b'N' } as c_char;
    let info = unsafe {
        sys::LAPACKE_sgeev(
            ROW_MAJOR,
            jobvl,
            jobvr,
            n as i32,
            a.as_mut_ptr(),
            lda as i32,
            wr.as_mut_ptr(),
            wi.as_mut_ptr(),
            vl.as_mut_ptr(),
            ldvl as i32,
            vr.as_mut_ptr(),
            ldvr as i32,
        )
    };
    map_info("lapack", info)
}

/// Complex non-symmetric eig: complex eigenvalues `w[i]`. (`Complex64`)
#[allow(clippy::too_many_arguments)]
pub fn zgeev(
    compute_vl: bool,
    compute_vr: bool,
    n: usize,
    a: &mut [Complex64],
    lda: usize,
    w: &mut [Complex64],
    vl: &mut [Complex64],
    ldvl: usize,
    vr: &mut [Complex64],
    ldvr: usize,
) -> Result<()> {
    let jobvl = if compute_vl { b'V' } else { b'N' } as c_char;
    let jobvr = if compute_vr { b'V' } else { b'N' } as c_char;
    let info = unsafe {
        sys::LAPACKE_zgeev(
            ROW_MAJOR,
            jobvl,
            jobvr,
            n as i32,
            a.as_mut_ptr() as *mut sys::__BindgenComplex<f64>,
            lda as i32,
            w.as_mut_ptr() as *mut sys::__BindgenComplex<f64>,
            vl.as_mut_ptr() as *mut sys::__BindgenComplex<f64>,
            ldvl as i32,
            vr.as_mut_ptr() as *mut sys::__BindgenComplex<f64>,
            ldvr as i32,
        )
    };
    map_info("lapack", info)
}

/// `Complex32` non-symmetric eig. See [`zgeev`].
#[allow(clippy::too_many_arguments)]
pub fn cgeev(
    compute_vl: bool,
    compute_vr: bool,
    n: usize,
    a: &mut [Complex32],
    lda: usize,
    w: &mut [Complex32],
    vl: &mut [Complex32],
    ldvl: usize,
    vr: &mut [Complex32],
    ldvr: usize,
) -> Result<()> {
    let jobvl = if compute_vl { b'V' } else { b'N' } as c_char;
    let jobvr = if compute_vr { b'V' } else { b'N' } as c_char;
    let info = unsafe {
        sys::LAPACKE_cgeev(
            ROW_MAJOR,
            jobvl,
            jobvr,
            n as i32,
            a.as_mut_ptr() as *mut sys::__BindgenComplex<f32>,
            lda as i32,
            w.as_mut_ptr() as *mut sys::__BindgenComplex<f32>,
            vl.as_mut_ptr() as *mut sys::__BindgenComplex<f32>,
            ldvl as i32,
            vr.as_mut_ptr() as *mut sys::__BindgenComplex<f32>,
            ldvr as i32,
        )
    };
    map_info("lapack", info)
}

// ----- Generalised symmetric eig (sygv / hegv) -----------------------------

/// Generalised symmetric eigenproblem `A·x = λ·B·x` (or related, per
/// `itype`). `compute_vectors = true` requests eigenvectors in `a`.
/// `B` must be positive definite. (`f64`)
#[allow(clippy::too_many_arguments)]
pub fn dsygv(
    itype: i32,
    compute_vectors: bool,
    uplo: Uplo,
    n: usize,
    a: &mut [f64],
    lda: usize,
    b: &mut [f64],
    ldb: usize,
    w: &mut [f64],
) -> Result<()> {
    let jobz = if compute_vectors { b'V' } else { b'N' } as c_char;
    let info = unsafe {
        sys::LAPACKE_dsygv(
            ROW_MAJOR,
            itype,
            jobz,
            uplo_char(uplo),
            n as i32,
            a.as_mut_ptr(),
            lda as i32,
            b.as_mut_ptr(),
            ldb as i32,
            w.as_mut_ptr(),
        )
    };
    map_info("lapack", info)
}

/// `f32` generalised symmetric eig. See [`dsygv`].
#[allow(clippy::too_many_arguments)]
pub fn ssygv(
    itype: i32,
    compute_vectors: bool,
    uplo: Uplo,
    n: usize,
    a: &mut [f32],
    lda: usize,
    b: &mut [f32],
    ldb: usize,
    w: &mut [f32],
) -> Result<()> {
    let jobz = if compute_vectors { b'V' } else { b'N' } as c_char;
    let info = unsafe {
        sys::LAPACKE_ssygv(
            ROW_MAJOR,
            itype,
            jobz,
            uplo_char(uplo),
            n as i32,
            a.as_mut_ptr(),
            lda as i32,
            b.as_mut_ptr(),
            ldb as i32,
            w.as_mut_ptr(),
        )
    };
    map_info("lapack", info)
}

/// Generalised hermitian eigenproblem. (`Complex64`)
#[allow(clippy::too_many_arguments)]
pub fn zhegv(
    itype: i32,
    compute_vectors: bool,
    uplo: Uplo,
    n: usize,
    a: &mut [Complex64],
    lda: usize,
    b: &mut [Complex64],
    ldb: usize,
    w: &mut [f64],
) -> Result<()> {
    let jobz = if compute_vectors { b'V' } else { b'N' } as c_char;
    let info = unsafe {
        sys::LAPACKE_zhegv(
            ROW_MAJOR,
            itype,
            jobz,
            uplo_char(uplo),
            n as i32,
            a.as_mut_ptr() as *mut sys::__BindgenComplex<f64>,
            lda as i32,
            b.as_mut_ptr() as *mut sys::__BindgenComplex<f64>,
            ldb as i32,
            w.as_mut_ptr(),
        )
    };
    map_info("lapack", info)
}

/// `Complex32` generalised hermitian eig. See [`zhegv`].
#[allow(clippy::too_many_arguments)]
pub fn chegv(
    itype: i32,
    compute_vectors: bool,
    uplo: Uplo,
    n: usize,
    a: &mut [Complex32],
    lda: usize,
    b: &mut [Complex32],
    ldb: usize,
    w: &mut [f32],
) -> Result<()> {
    let jobz = if compute_vectors { b'V' } else { b'N' } as c_char;
    let info = unsafe {
        sys::LAPACKE_chegv(
            ROW_MAJOR,
            itype,
            jobz,
            uplo_char(uplo),
            n as i32,
            a.as_mut_ptr() as *mut sys::__BindgenComplex<f32>,
            lda as i32,
            b.as_mut_ptr() as *mut sys::__BindgenComplex<f32>,
            ldb as i32,
            w.as_mut_ptr(),
        )
    };
    map_info("lapack", info)
}

// ----- QR with column pivoting --------------------------------------------

/// QR factorisation with column pivoting. `jpvt[i] != 0` (1-based) means
/// the `i`-th column is fixed at the front; on output, `jpvt` carries
/// the actual permutation. (`f64`)
pub fn dgeqp3(
    m: usize,
    n: usize,
    a: &mut [f64],
    lda: usize,
    jpvt: &mut [i32],
    tau: &mut [f64],
) -> Result<()> {
    let info = unsafe {
        sys::LAPACKE_dgeqp3(
            ROW_MAJOR,
            m as i32,
            n as i32,
            a.as_mut_ptr(),
            lda as i32,
            jpvt.as_mut_ptr(),
            tau.as_mut_ptr(),
        )
    };
    map_info("lapack", info)
}

/// `f32` QR with column pivoting. See [`dgeqp3`].
pub fn sgeqp3(
    m: usize,
    n: usize,
    a: &mut [f32],
    lda: usize,
    jpvt: &mut [i32],
    tau: &mut [f32],
) -> Result<()> {
    let info = unsafe {
        sys::LAPACKE_sgeqp3(
            ROW_MAJOR,
            m as i32,
            n as i32,
            a.as_mut_ptr(),
            lda as i32,
            jpvt.as_mut_ptr(),
            tau.as_mut_ptr(),
        )
    };
    map_info("lapack", info)
}

/// `Complex64` QR with column pivoting. See [`dgeqp3`].
pub fn zgeqp3(
    m: usize,
    n: usize,
    a: &mut [Complex64],
    lda: usize,
    jpvt: &mut [i32],
    tau: &mut [Complex64],
) -> Result<()> {
    let info = unsafe {
        sys::LAPACKE_zgeqp3(
            ROW_MAJOR,
            m as i32,
            n as i32,
            a.as_mut_ptr() as *mut sys::__BindgenComplex<f64>,
            lda as i32,
            jpvt.as_mut_ptr(),
            tau.as_mut_ptr() as *mut sys::__BindgenComplex<f64>,
        )
    };
    map_info("lapack", info)
}

/// `Complex32` QR with column pivoting. See [`dgeqp3`].
pub fn cgeqp3(
    m: usize,
    n: usize,
    a: &mut [Complex32],
    lda: usize,
    jpvt: &mut [i32],
    tau: &mut [Complex32],
) -> Result<()> {
    let info = unsafe {
        sys::LAPACKE_cgeqp3(
            ROW_MAJOR,
            m as i32,
            n as i32,
            a.as_mut_ptr() as *mut sys::__BindgenComplex<f32>,
            lda as i32,
            jpvt.as_mut_ptr(),
            tau.as_mut_ptr() as *mut sys::__BindgenComplex<f32>,
        )
    };
    map_info("lapack", info)
}

// ----- Matrix copy and norm ------------------------------------------------

/// Copy a sub-block of a matrix: `B := A` (or just the upper / lower
/// triangle). Use `Uplo::Upper` or `Uplo::Lower` to copy only that
/// triangle, or the value `b'A'` (general full copy) via the alternate
/// underlying char — pass any non-Upper/non-Lower if you need that
/// pathway. (`f64`)
pub fn dlacpy(uplo: Uplo, m: usize, n: usize, a: &[f64], lda: usize, b: &mut [f64], ldb: usize) {
    unsafe {
        sys::LAPACKE_dlacpy(
            ROW_MAJOR,
            uplo_char(uplo),
            m as i32,
            n as i32,
            a.as_ptr(),
            lda as i32,
            b.as_mut_ptr(),
            ldb as i32,
        );
    }
}

/// `f32` matrix copy. See [`dlacpy`].
pub fn slacpy(uplo: Uplo, m: usize, n: usize, a: &[f32], lda: usize, b: &mut [f32], ldb: usize) {
    unsafe {
        sys::LAPACKE_slacpy(
            ROW_MAJOR,
            uplo_char(uplo),
            m as i32,
            n as i32,
            a.as_ptr(),
            lda as i32,
            b.as_mut_ptr(),
            ldb as i32,
        );
    }
}

/// `Complex64` matrix copy. See [`dlacpy`].
pub fn zlacpy(
    uplo: Uplo,
    m: usize,
    n: usize,
    a: &[Complex64],
    lda: usize,
    b: &mut [Complex64],
    ldb: usize,
) {
    unsafe {
        sys::LAPACKE_zlacpy(
            ROW_MAJOR,
            uplo_char(uplo),
            m as i32,
            n as i32,
            a.as_ptr() as *const sys::__BindgenComplex<f64>,
            lda as i32,
            b.as_mut_ptr() as *mut sys::__BindgenComplex<f64>,
            ldb as i32,
        );
    }
}

/// `Complex32` matrix copy. See [`dlacpy`].
pub fn clacpy(
    uplo: Uplo,
    m: usize,
    n: usize,
    a: &[Complex32],
    lda: usize,
    b: &mut [Complex32],
    ldb: usize,
) {
    unsafe {
        sys::LAPACKE_clacpy(
            ROW_MAJOR,
            uplo_char(uplo),
            m as i32,
            n as i32,
            a.as_ptr() as *const sys::__BindgenComplex<f32>,
            lda as i32,
            b.as_mut_ptr() as *mut sys::__BindgenComplex<f32>,
            ldb as i32,
        );
    }
}

/// Compute a matrix norm (`Norm::Max`, `One`, `Inf`, `Frobenius`). (`f64`)
pub fn dlange(norm: Norm, m: usize, n: usize, a: &[f64], lda: usize) -> f64 {
    unsafe {
        sys::LAPACKE_dlange(
            ROW_MAJOR,
            norm.raw(),
            m as i32,
            n as i32,
            a.as_ptr(),
            lda as i32,
        )
    }
}

/// `f32` matrix norm. See [`dlange`].
pub fn slange(norm: Norm, m: usize, n: usize, a: &[f32], lda: usize) -> f32 {
    unsafe {
        sys::LAPACKE_slange(
            ROW_MAJOR,
            norm.raw(),
            m as i32,
            n as i32,
            a.as_ptr(),
            lda as i32,
        )
    }
}

/// `Complex64` matrix norm (returns the underlying real). See [`dlange`].
pub fn zlange(norm: Norm, m: usize, n: usize, a: &[Complex64], lda: usize) -> f64 {
    unsafe {
        sys::LAPACKE_zlange(
            ROW_MAJOR,
            norm.raw(),
            m as i32,
            n as i32,
            a.as_ptr() as *const sys::__BindgenComplex<f64>,
            lda as i32,
        )
    }
}

/// `Complex32` matrix norm (returns the underlying real). See [`dlange`].
pub fn clange(norm: Norm, m: usize, n: usize, a: &[Complex32], lda: usize) -> f32 {
    unsafe {
        sys::LAPACKE_clange(
            ROW_MAJOR,
            norm.raw(),
            m as i32,
            n as i32,
            a.as_ptr() as *const sys::__BindgenComplex<f32>,
            lda as i32,
        )
    }
}

// =========================================================================
//   Advanced eigen / SVD: divide-and-conquer, relative-robust, partial,
//   generalized, Schur, Bunch-Kaufman, condition numbers, Householder
// =========================================================================

/// Range selector for the partial-eigenvalue / partial-SVD routines.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Range {
    /// All eigenvalues / singular values.
    All,
    /// Half-open value range `[vl, vu)`.
    Values,
    /// Index range `[il, iu]` (1-based).
    Indices,
}

impl Range {
    fn raw(self) -> c_char {
        match self {
            Range::All => b'A' as c_char,
            Range::Values => b'V' as c_char,
            Range::Indices => b'I' as c_char,
        }
    }
}

/// Macro to compactly define an LAPACKE wrapper that returns
/// `Result<()>` from `info`.
macro_rules! lapack_call {
    ($name:expr, $fn:ident, $($arg:expr),* $(,)?) => {{
        let info = unsafe { sys::$fn($($arg,)*) };
        map_info($name, info)
    }};
}

// ----- syevd: divide-and-conquer symmetric eigen -----
/// Divide-and-conquer symmetric eigendecomposition. (`f64`)
pub fn dsyevd(
    jobz_v: bool,
    uplo: Uplo,
    n: usize,
    a: &mut [f64],
    lda: usize,
    w: &mut [f64],
) -> Result<()> {
    let jobz = if jobz_v { b'V' } else { b'N' } as c_char;
    lapack_call!(
        "lapack",
        LAPACKE_dsyevd,
        ROW_MAJOR,
        jobz,
        uplo_char(uplo),
        n as i32,
        a.as_mut_ptr(),
        lda as i32,
        w.as_mut_ptr()
    )
}
/// `f32` divide-and-conquer eigen. See [`dsyevd`].
pub fn ssyevd(
    jobz_v: bool,
    uplo: Uplo,
    n: usize,
    a: &mut [f32],
    lda: usize,
    w: &mut [f32],
) -> Result<()> {
    let jobz = if jobz_v { b'V' } else { b'N' } as c_char;
    lapack_call!(
        "lapack",
        LAPACKE_ssyevd,
        ROW_MAJOR,
        jobz,
        uplo_char(uplo),
        n as i32,
        a.as_mut_ptr(),
        lda as i32,
        w.as_mut_ptr()
    )
}
/// `Complex64` divide-and-conquer Hermitian eigen. See [`dsyevd`].
pub fn zheevd(
    jobz_v: bool,
    uplo: Uplo,
    n: usize,
    a: &mut [Complex64],
    lda: usize,
    w: &mut [f64],
) -> Result<()> {
    let jobz = if jobz_v { b'V' } else { b'N' } as c_char;
    lapack_call!(
        "lapack",
        LAPACKE_zheevd,
        ROW_MAJOR,
        jobz,
        uplo_char(uplo),
        n as i32,
        a.as_mut_ptr() as *mut sys::__BindgenComplex<f64>,
        lda as i32,
        w.as_mut_ptr()
    )
}
/// `Complex32` divide-and-conquer Hermitian eigen. See [`dsyevd`].
pub fn cheevd(
    jobz_v: bool,
    uplo: Uplo,
    n: usize,
    a: &mut [Complex32],
    lda: usize,
    w: &mut [f32],
) -> Result<()> {
    let jobz = if jobz_v { b'V' } else { b'N' } as c_char;
    lapack_call!(
        "lapack",
        LAPACKE_cheevd,
        ROW_MAJOR,
        jobz,
        uplo_char(uplo),
        n as i32,
        a.as_mut_ptr() as *mut sys::__BindgenComplex<f32>,
        lda as i32,
        w.as_mut_ptr()
    )
}

// ----- syevr/heevr: relative-robust eigen with optional range -----

/// Relative-robust symmetric eigendecomposition with optional range
/// selection. Returns the count of eigenvalues actually computed (in
/// the selected range). (`f64`)
#[allow(clippy::too_many_arguments)]
pub fn dsyevr(
    jobz_v: bool,
    range: Range,
    uplo: Uplo,
    n: usize,
    a: &mut [f64],
    lda: usize,
    vl: f64,
    vu: f64,
    il: i32,
    iu: i32,
    abstol: f64,
    w: &mut [f64],
    z: &mut [f64],
    ldz: usize,
    isuppz: &mut [i32],
) -> Result<usize> {
    let jobz = if jobz_v { b'V' } else { b'N' } as c_char;
    let mut m: i32 = 0;
    let info = unsafe {
        sys::LAPACKE_dsyevr(
            ROW_MAJOR,
            jobz,
            range.raw(),
            uplo_char(uplo),
            n as i32,
            a.as_mut_ptr(),
            lda as i32,
            vl,
            vu,
            il,
            iu,
            abstol,
            &mut m,
            w.as_mut_ptr(),
            z.as_mut_ptr(),
            ldz as i32,
            isuppz.as_mut_ptr(),
        )
    };
    map_info("lapack", info)?;
    Ok(m as usize)
}

/// `f32` relative-robust eigen. See [`dsyevr`].
#[allow(clippy::too_many_arguments)]
pub fn ssyevr(
    jobz_v: bool,
    range: Range,
    uplo: Uplo,
    n: usize,
    a: &mut [f32],
    lda: usize,
    vl: f32,
    vu: f32,
    il: i32,
    iu: i32,
    abstol: f32,
    w: &mut [f32],
    z: &mut [f32],
    ldz: usize,
    isuppz: &mut [i32],
) -> Result<usize> {
    let jobz = if jobz_v { b'V' } else { b'N' } as c_char;
    let mut m: i32 = 0;
    let info = unsafe {
        sys::LAPACKE_ssyevr(
            ROW_MAJOR,
            jobz,
            range.raw(),
            uplo_char(uplo),
            n as i32,
            a.as_mut_ptr(),
            lda as i32,
            vl,
            vu,
            il,
            iu,
            abstol,
            &mut m,
            w.as_mut_ptr(),
            z.as_mut_ptr(),
            ldz as i32,
            isuppz.as_mut_ptr(),
        )
    };
    map_info("lapack", info)?;
    Ok(m as usize)
}

/// `Complex64` relative-robust Hermitian eigen. See [`dsyevr`].
#[allow(clippy::too_many_arguments)]
pub fn zheevr(
    jobz_v: bool,
    range: Range,
    uplo: Uplo,
    n: usize,
    a: &mut [Complex64],
    lda: usize,
    vl: f64,
    vu: f64,
    il: i32,
    iu: i32,
    abstol: f64,
    w: &mut [f64],
    z: &mut [Complex64],
    ldz: usize,
    isuppz: &mut [i32],
) -> Result<usize> {
    let jobz = if jobz_v { b'V' } else { b'N' } as c_char;
    let mut m: i32 = 0;
    let info = unsafe {
        sys::LAPACKE_zheevr(
            ROW_MAJOR,
            jobz,
            range.raw(),
            uplo_char(uplo),
            n as i32,
            a.as_mut_ptr() as *mut sys::__BindgenComplex<f64>,
            lda as i32,
            vl,
            vu,
            il,
            iu,
            abstol,
            &mut m,
            w.as_mut_ptr(),
            z.as_mut_ptr() as *mut sys::__BindgenComplex<f64>,
            ldz as i32,
            isuppz.as_mut_ptr(),
        )
    };
    map_info("lapack", info)?;
    Ok(m as usize)
}

/// `Complex32` relative-robust Hermitian eigen. See [`dsyevr`].
#[allow(clippy::too_many_arguments)]
pub fn cheevr(
    jobz_v: bool,
    range: Range,
    uplo: Uplo,
    n: usize,
    a: &mut [Complex32],
    lda: usize,
    vl: f32,
    vu: f32,
    il: i32,
    iu: i32,
    abstol: f32,
    w: &mut [f32],
    z: &mut [Complex32],
    ldz: usize,
    isuppz: &mut [i32],
) -> Result<usize> {
    let jobz = if jobz_v { b'V' } else { b'N' } as c_char;
    let mut m: i32 = 0;
    let info = unsafe {
        sys::LAPACKE_cheevr(
            ROW_MAJOR,
            jobz,
            range.raw(),
            uplo_char(uplo),
            n as i32,
            a.as_mut_ptr() as *mut sys::__BindgenComplex<f32>,
            lda as i32,
            vl,
            vu,
            il,
            iu,
            abstol,
            &mut m,
            w.as_mut_ptr(),
            z.as_mut_ptr() as *mut sys::__BindgenComplex<f32>,
            ldz as i32,
            isuppz.as_mut_ptr(),
        )
    };
    map_info("lapack", info)?;
    Ok(m as usize)
}

// ----- gesvdx: partial SVD with index/value range -----

/// Partial SVD by value or index range. (`f64`) Returns the number of
/// singular values actually computed.
#[allow(clippy::too_many_arguments)]
pub fn dgesvdx(
    jobu_v: bool,
    jobvt_v: bool,
    range: Range,
    m: usize,
    n: usize,
    a: &mut [f64],
    lda: usize,
    vl: f64,
    vu: f64,
    il: i32,
    iu: i32,
    s: &mut [f64],
    u: &mut [f64],
    ldu: usize,
    vt: &mut [f64],
    ldvt: usize,
    superb: &mut [i32],
) -> Result<usize> {
    let jobu = if jobu_v { b'V' } else { b'N' } as c_char;
    let jobvt = if jobvt_v { b'V' } else { b'N' } as c_char;
    let mut ns: i32 = 0;
    let info = unsafe {
        sys::LAPACKE_dgesvdx(
            ROW_MAJOR,
            jobu,
            jobvt,
            range.raw(),
            m as i32,
            n as i32,
            a.as_mut_ptr(),
            lda as i32,
            vl,
            vu,
            il,
            iu,
            &mut ns,
            s.as_mut_ptr(),
            u.as_mut_ptr(),
            ldu as i32,
            vt.as_mut_ptr(),
            ldvt as i32,
            superb.as_mut_ptr(),
        )
    };
    map_info("lapack", info)?;
    Ok(ns as usize)
}

/// `f32` partial SVD. See [`dgesvdx`].
#[allow(clippy::too_many_arguments)]
pub fn sgesvdx(
    jobu_v: bool,
    jobvt_v: bool,
    range: Range,
    m: usize,
    n: usize,
    a: &mut [f32],
    lda: usize,
    vl: f32,
    vu: f32,
    il: i32,
    iu: i32,
    s: &mut [f32],
    u: &mut [f32],
    ldu: usize,
    vt: &mut [f32],
    ldvt: usize,
    superb: &mut [i32],
) -> Result<usize> {
    let jobu = if jobu_v { b'V' } else { b'N' } as c_char;
    let jobvt = if jobvt_v { b'V' } else { b'N' } as c_char;
    let mut ns: i32 = 0;
    let info = unsafe {
        sys::LAPACKE_sgesvdx(
            ROW_MAJOR,
            jobu,
            jobvt,
            range.raw(),
            m as i32,
            n as i32,
            a.as_mut_ptr(),
            lda as i32,
            vl,
            vu,
            il,
            iu,
            &mut ns,
            s.as_mut_ptr(),
            u.as_mut_ptr(),
            ldu as i32,
            vt.as_mut_ptr(),
            ldvt as i32,
            superb.as_mut_ptr(),
        )
    };
    map_info("lapack", info)?;
    Ok(ns as usize)
}

/// `Complex64` partial SVD. See [`dgesvdx`].
#[allow(clippy::too_many_arguments)]
pub fn zgesvdx(
    jobu_v: bool,
    jobvt_v: bool,
    range: Range,
    m: usize,
    n: usize,
    a: &mut [Complex64],
    lda: usize,
    vl: f64,
    vu: f64,
    il: i32,
    iu: i32,
    s: &mut [f64],
    u: &mut [Complex64],
    ldu: usize,
    vt: &mut [Complex64],
    ldvt: usize,
    superb: &mut [i32],
) -> Result<usize> {
    let jobu = if jobu_v { b'V' } else { b'N' } as c_char;
    let jobvt = if jobvt_v { b'V' } else { b'N' } as c_char;
    let mut ns: i32 = 0;
    let info = unsafe {
        sys::LAPACKE_zgesvdx(
            ROW_MAJOR,
            jobu,
            jobvt,
            range.raw(),
            m as i32,
            n as i32,
            a.as_mut_ptr() as *mut sys::__BindgenComplex<f64>,
            lda as i32,
            vl,
            vu,
            il,
            iu,
            &mut ns,
            s.as_mut_ptr(),
            u.as_mut_ptr() as *mut sys::__BindgenComplex<f64>,
            ldu as i32,
            vt.as_mut_ptr() as *mut sys::__BindgenComplex<f64>,
            ldvt as i32,
            superb.as_mut_ptr(),
        )
    };
    map_info("lapack", info)?;
    Ok(ns as usize)
}

/// `Complex32` partial SVD. See [`dgesvdx`].
#[allow(clippy::too_many_arguments)]
pub fn cgesvdx(
    jobu_v: bool,
    jobvt_v: bool,
    range: Range,
    m: usize,
    n: usize,
    a: &mut [Complex32],
    lda: usize,
    vl: f32,
    vu: f32,
    il: i32,
    iu: i32,
    s: &mut [f32],
    u: &mut [Complex32],
    ldu: usize,
    vt: &mut [Complex32],
    ldvt: usize,
    superb: &mut [i32],
) -> Result<usize> {
    let jobu = if jobu_v { b'V' } else { b'N' } as c_char;
    let jobvt = if jobvt_v { b'V' } else { b'N' } as c_char;
    let mut ns: i32 = 0;
    let info = unsafe {
        sys::LAPACKE_cgesvdx(
            ROW_MAJOR,
            jobu,
            jobvt,
            range.raw(),
            m as i32,
            n as i32,
            a.as_mut_ptr() as *mut sys::__BindgenComplex<f32>,
            lda as i32,
            vl,
            vu,
            il,
            iu,
            &mut ns,
            s.as_mut_ptr(),
            u.as_mut_ptr() as *mut sys::__BindgenComplex<f32>,
            ldu as i32,
            vt.as_mut_ptr() as *mut sys::__BindgenComplex<f32>,
            ldvt as i32,
            superb.as_mut_ptr(),
        )
    };
    map_info("lapack", info)?;
    Ok(ns as usize)
}

// ----- ggev: generalised non-symmetric eigenproblem -----

/// Generalised eigenproblem `A·x = λ·B·x`. Real version returns
/// eigenvalues as `(alphar + alphai·j) / beta`. (`f64`)
#[allow(clippy::too_many_arguments)]
pub fn dggev(
    compute_vl: bool,
    compute_vr: bool,
    n: usize,
    a: &mut [f64],
    lda: usize,
    b: &mut [f64],
    ldb: usize,
    alphar: &mut [f64],
    alphai: &mut [f64],
    beta: &mut [f64],
    vl: &mut [f64],
    ldvl: usize,
    vr: &mut [f64],
    ldvr: usize,
) -> Result<()> {
    let jobvl = if compute_vl { b'V' } else { b'N' } as c_char;
    let jobvr = if compute_vr { b'V' } else { b'N' } as c_char;
    lapack_call!(
        "lapack",
        LAPACKE_dggev,
        ROW_MAJOR,
        jobvl,
        jobvr,
        n as i32,
        a.as_mut_ptr(),
        lda as i32,
        b.as_mut_ptr(),
        ldb as i32,
        alphar.as_mut_ptr(),
        alphai.as_mut_ptr(),
        beta.as_mut_ptr(),
        vl.as_mut_ptr(),
        ldvl as i32,
        vr.as_mut_ptr(),
        ldvr as i32
    )
}

/// `f32` generalised eig. See [`dggev`].
#[allow(clippy::too_many_arguments)]
pub fn sggev(
    compute_vl: bool,
    compute_vr: bool,
    n: usize,
    a: &mut [f32],
    lda: usize,
    b: &mut [f32],
    ldb: usize,
    alphar: &mut [f32],
    alphai: &mut [f32],
    beta: &mut [f32],
    vl: &mut [f32],
    ldvl: usize,
    vr: &mut [f32],
    ldvr: usize,
) -> Result<()> {
    let jobvl = if compute_vl { b'V' } else { b'N' } as c_char;
    let jobvr = if compute_vr { b'V' } else { b'N' } as c_char;
    lapack_call!(
        "lapack",
        LAPACKE_sggev,
        ROW_MAJOR,
        jobvl,
        jobvr,
        n as i32,
        a.as_mut_ptr(),
        lda as i32,
        b.as_mut_ptr(),
        ldb as i32,
        alphar.as_mut_ptr(),
        alphai.as_mut_ptr(),
        beta.as_mut_ptr(),
        vl.as_mut_ptr(),
        ldvl as i32,
        vr.as_mut_ptr(),
        ldvr as i32
    )
}

/// `Complex64` generalised eig. Returns single complex `alpha` (no
/// split). See [`dggev`].
#[allow(clippy::too_many_arguments)]
pub fn zggev(
    compute_vl: bool,
    compute_vr: bool,
    n: usize,
    a: &mut [Complex64],
    lda: usize,
    b: &mut [Complex64],
    ldb: usize,
    alpha: &mut [Complex64],
    beta: &mut [Complex64],
    vl: &mut [Complex64],
    ldvl: usize,
    vr: &mut [Complex64],
    ldvr: usize,
) -> Result<()> {
    let jobvl = if compute_vl { b'V' } else { b'N' } as c_char;
    let jobvr = if compute_vr { b'V' } else { b'N' } as c_char;
    lapack_call!(
        "lapack",
        LAPACKE_zggev,
        ROW_MAJOR,
        jobvl,
        jobvr,
        n as i32,
        a.as_mut_ptr() as *mut sys::__BindgenComplex<f64>,
        lda as i32,
        b.as_mut_ptr() as *mut sys::__BindgenComplex<f64>,
        ldb as i32,
        alpha.as_mut_ptr() as *mut sys::__BindgenComplex<f64>,
        beta.as_mut_ptr() as *mut sys::__BindgenComplex<f64>,
        vl.as_mut_ptr() as *mut sys::__BindgenComplex<f64>,
        ldvl as i32,
        vr.as_mut_ptr() as *mut sys::__BindgenComplex<f64>,
        ldvr as i32
    )
}

/// `Complex32` generalised eig. See [`zggev`].
#[allow(clippy::too_many_arguments)]
pub fn cggev(
    compute_vl: bool,
    compute_vr: bool,
    n: usize,
    a: &mut [Complex32],
    lda: usize,
    b: &mut [Complex32],
    ldb: usize,
    alpha: &mut [Complex32],
    beta: &mut [Complex32],
    vl: &mut [Complex32],
    ldvl: usize,
    vr: &mut [Complex32],
    ldvr: usize,
) -> Result<()> {
    let jobvl = if compute_vl { b'V' } else { b'N' } as c_char;
    let jobvr = if compute_vr { b'V' } else { b'N' } as c_char;
    lapack_call!(
        "lapack",
        LAPACKE_cggev,
        ROW_MAJOR,
        jobvl,
        jobvr,
        n as i32,
        a.as_mut_ptr() as *mut sys::__BindgenComplex<f32>,
        lda as i32,
        b.as_mut_ptr() as *mut sys::__BindgenComplex<f32>,
        ldb as i32,
        alpha.as_mut_ptr() as *mut sys::__BindgenComplex<f32>,
        beta.as_mut_ptr() as *mut sys::__BindgenComplex<f32>,
        vl.as_mut_ptr() as *mut sys::__BindgenComplex<f32>,
        ldvl as i32,
        vr.as_mut_ptr() as *mut sys::__BindgenComplex<f32>,
        ldvr as i32
    )
}

// ----- gees: Schur decomposition (no eigenvalue selection callback) -----

/// Schur decomposition `A = Z · T · Zᵀ`. The eigenvalue-selection
/// callback (`select`) is passed `None` here; use the sys crate
/// directly if you need to filter eigenvalues. Returns `sdim`, the
/// number of eigenvalues that satisfied the (absent) selection. (`f64`)
#[allow(clippy::too_many_arguments)]
pub fn dgees(
    compute_vs: bool,
    n: usize,
    a: &mut [f64],
    lda: usize,
    wr: &mut [f64],
    wi: &mut [f64],
    vs: &mut [f64],
    ldvs: usize,
) -> Result<usize> {
    let jobvs = if compute_vs { b'V' } else { b'N' } as c_char;
    let mut sdim: i32 = 0;
    let info = unsafe {
        sys::LAPACKE_dgees(
            ROW_MAJOR,
            jobvs,
            b'N' as c_char,
            None,
            n as i32,
            a.as_mut_ptr(),
            lda as i32,
            &mut sdim,
            wr.as_mut_ptr(),
            wi.as_mut_ptr(),
            vs.as_mut_ptr(),
            ldvs as i32,
        )
    };
    map_info("lapack", info)?;
    Ok(sdim as usize)
}

/// `f32` Schur decomposition. See [`dgees`].
#[allow(clippy::too_many_arguments)]
pub fn sgees(
    compute_vs: bool,
    n: usize,
    a: &mut [f32],
    lda: usize,
    wr: &mut [f32],
    wi: &mut [f32],
    vs: &mut [f32],
    ldvs: usize,
) -> Result<usize> {
    let jobvs = if compute_vs { b'V' } else { b'N' } as c_char;
    let mut sdim: i32 = 0;
    let info = unsafe {
        sys::LAPACKE_sgees(
            ROW_MAJOR,
            jobvs,
            b'N' as c_char,
            None,
            n as i32,
            a.as_mut_ptr(),
            lda as i32,
            &mut sdim,
            wr.as_mut_ptr(),
            wi.as_mut_ptr(),
            vs.as_mut_ptr(),
            ldvs as i32,
        )
    };
    map_info("lapack", info)?;
    Ok(sdim as usize)
}

// ----- sytrf / sytrs / sytri: Bunch-Kaufman family for symmetric -----

/// Bunch-Kaufman factorisation `A = L·D·Lᵀ` for symmetric indefinite. (`f64`)
pub fn dsytrf(uplo: Uplo, n: usize, a: &mut [f64], lda: usize, ipiv: &mut [i32]) -> Result<()> {
    lapack_call!(
        "lapack",
        LAPACKE_dsytrf,
        ROW_MAJOR,
        uplo_char(uplo),
        n as i32,
        a.as_mut_ptr(),
        lda as i32,
        ipiv.as_mut_ptr()
    )
}
/// `f32` Bunch-Kaufman factorisation. See [`dsytrf`].
pub fn ssytrf(uplo: Uplo, n: usize, a: &mut [f32], lda: usize, ipiv: &mut [i32]) -> Result<()> {
    lapack_call!(
        "lapack",
        LAPACKE_ssytrf,
        ROW_MAJOR,
        uplo_char(uplo),
        n as i32,
        a.as_mut_ptr(),
        lda as i32,
        ipiv.as_mut_ptr()
    )
}
/// `Complex64` symmetric Bunch-Kaufman factorisation. See [`dsytrf`].
pub fn zsytrf(
    uplo: Uplo,
    n: usize,
    a: &mut [Complex64],
    lda: usize,
    ipiv: &mut [i32],
) -> Result<()> {
    lapack_call!(
        "lapack",
        LAPACKE_zsytrf,
        ROW_MAJOR,
        uplo_char(uplo),
        n as i32,
        a.as_mut_ptr() as *mut sys::__BindgenComplex<f64>,
        lda as i32,
        ipiv.as_mut_ptr()
    )
}
/// `Complex32` symmetric Bunch-Kaufman factorisation. See [`dsytrf`].
pub fn csytrf(
    uplo: Uplo,
    n: usize,
    a: &mut [Complex32],
    lda: usize,
    ipiv: &mut [i32],
) -> Result<()> {
    lapack_call!(
        "lapack",
        LAPACKE_csytrf,
        ROW_MAJOR,
        uplo_char(uplo),
        n as i32,
        a.as_mut_ptr() as *mut sys::__BindgenComplex<f32>,
        lda as i32,
        ipiv.as_mut_ptr()
    )
}

/// Solve after [`dsytrf`].
#[allow(clippy::too_many_arguments)]
pub fn dsytrs(
    uplo: Uplo,
    n: usize,
    nrhs: usize,
    a: &[f64],
    lda: usize,
    ipiv: &[i32],
    b: &mut [f64],
    ldb: usize,
) -> Result<()> {
    lapack_call!(
        "lapack",
        LAPACKE_dsytrs,
        ROW_MAJOR,
        uplo_char(uplo),
        n as i32,
        nrhs as i32,
        a.as_ptr(),
        lda as i32,
        ipiv.as_ptr(),
        b.as_mut_ptr(),
        ldb as i32
    )
}
/// `f32` solve after [`ssytrf`].
#[allow(clippy::too_many_arguments)]
pub fn ssytrs(
    uplo: Uplo,
    n: usize,
    nrhs: usize,
    a: &[f32],
    lda: usize,
    ipiv: &[i32],
    b: &mut [f32],
    ldb: usize,
) -> Result<()> {
    lapack_call!(
        "lapack",
        LAPACKE_ssytrs,
        ROW_MAJOR,
        uplo_char(uplo),
        n as i32,
        nrhs as i32,
        a.as_ptr(),
        lda as i32,
        ipiv.as_ptr(),
        b.as_mut_ptr(),
        ldb as i32
    )
}
/// `Complex64` solve after [`zsytrf`].
#[allow(clippy::too_many_arguments)]
pub fn zsytrs(
    uplo: Uplo,
    n: usize,
    nrhs: usize,
    a: &[Complex64],
    lda: usize,
    ipiv: &[i32],
    b: &mut [Complex64],
    ldb: usize,
) -> Result<()> {
    lapack_call!(
        "lapack",
        LAPACKE_zsytrs,
        ROW_MAJOR,
        uplo_char(uplo),
        n as i32,
        nrhs as i32,
        a.as_ptr() as *const sys::__BindgenComplex<f64>,
        lda as i32,
        ipiv.as_ptr(),
        b.as_mut_ptr() as *mut sys::__BindgenComplex<f64>,
        ldb as i32
    )
}
/// `Complex32` solve after [`csytrf`].
#[allow(clippy::too_many_arguments)]
pub fn csytrs(
    uplo: Uplo,
    n: usize,
    nrhs: usize,
    a: &[Complex32],
    lda: usize,
    ipiv: &[i32],
    b: &mut [Complex32],
    ldb: usize,
) -> Result<()> {
    lapack_call!(
        "lapack",
        LAPACKE_csytrs,
        ROW_MAJOR,
        uplo_char(uplo),
        n as i32,
        nrhs as i32,
        a.as_ptr() as *const sys::__BindgenComplex<f32>,
        lda as i32,
        ipiv.as_ptr(),
        b.as_mut_ptr() as *mut sys::__BindgenComplex<f32>,
        ldb as i32
    )
}

/// Inverse from [`dsytrf`].
pub fn dsytri(uplo: Uplo, n: usize, a: &mut [f64], lda: usize, ipiv: &[i32]) -> Result<()> {
    lapack_call!(
        "lapack",
        LAPACKE_dsytri,
        ROW_MAJOR,
        uplo_char(uplo),
        n as i32,
        a.as_mut_ptr(),
        lda as i32,
        ipiv.as_ptr()
    )
}
/// `f32` inverse from [`ssytrf`].
pub fn ssytri(uplo: Uplo, n: usize, a: &mut [f32], lda: usize, ipiv: &[i32]) -> Result<()> {
    lapack_call!(
        "lapack",
        LAPACKE_ssytri,
        ROW_MAJOR,
        uplo_char(uplo),
        n as i32,
        a.as_mut_ptr(),
        lda as i32,
        ipiv.as_ptr()
    )
}
/// `Complex64` inverse from [`zsytrf`].
pub fn zsytri(uplo: Uplo, n: usize, a: &mut [Complex64], lda: usize, ipiv: &[i32]) -> Result<()> {
    lapack_call!(
        "lapack",
        LAPACKE_zsytri,
        ROW_MAJOR,
        uplo_char(uplo),
        n as i32,
        a.as_mut_ptr() as *mut sys::__BindgenComplex<f64>,
        lda as i32,
        ipiv.as_ptr()
    )
}
/// `Complex32` inverse from [`csytrf`].
pub fn csytri(uplo: Uplo, n: usize, a: &mut [Complex32], lda: usize, ipiv: &[i32]) -> Result<()> {
    lapack_call!(
        "lapack",
        LAPACKE_csytri,
        ROW_MAJOR,
        uplo_char(uplo),
        n as i32,
        a.as_mut_ptr() as *mut sys::__BindgenComplex<f32>,
        lda as i32,
        ipiv.as_ptr()
    )
}

// ----- hetrf / hetrs / hetri: Hermitian Bunch-Kaufman (complex only) -----

/// Hermitian Bunch-Kaufman factorisation. (`Complex64`)
pub fn zhetrf(
    uplo: Uplo,
    n: usize,
    a: &mut [Complex64],
    lda: usize,
    ipiv: &mut [i32],
) -> Result<()> {
    lapack_call!(
        "lapack",
        LAPACKE_zhetrf,
        ROW_MAJOR,
        uplo_char(uplo),
        n as i32,
        a.as_mut_ptr() as *mut sys::__BindgenComplex<f64>,
        lda as i32,
        ipiv.as_mut_ptr()
    )
}
/// `Complex32` Hermitian Bunch-Kaufman. See [`zhetrf`].
pub fn chetrf(
    uplo: Uplo,
    n: usize,
    a: &mut [Complex32],
    lda: usize,
    ipiv: &mut [i32],
) -> Result<()> {
    lapack_call!(
        "lapack",
        LAPACKE_chetrf,
        ROW_MAJOR,
        uplo_char(uplo),
        n as i32,
        a.as_mut_ptr() as *mut sys::__BindgenComplex<f32>,
        lda as i32,
        ipiv.as_mut_ptr()
    )
}
/// `Complex64` solve after [`zhetrf`].
#[allow(clippy::too_many_arguments)]
pub fn zhetrs(
    uplo: Uplo,
    n: usize,
    nrhs: usize,
    a: &[Complex64],
    lda: usize,
    ipiv: &[i32],
    b: &mut [Complex64],
    ldb: usize,
) -> Result<()> {
    lapack_call!(
        "lapack",
        LAPACKE_zhetrs,
        ROW_MAJOR,
        uplo_char(uplo),
        n as i32,
        nrhs as i32,
        a.as_ptr() as *const sys::__BindgenComplex<f64>,
        lda as i32,
        ipiv.as_ptr(),
        b.as_mut_ptr() as *mut sys::__BindgenComplex<f64>,
        ldb as i32
    )
}
/// `Complex32` solve after [`chetrf`].
#[allow(clippy::too_many_arguments)]
pub fn chetrs(
    uplo: Uplo,
    n: usize,
    nrhs: usize,
    a: &[Complex32],
    lda: usize,
    ipiv: &[i32],
    b: &mut [Complex32],
    ldb: usize,
) -> Result<()> {
    lapack_call!(
        "lapack",
        LAPACKE_chetrs,
        ROW_MAJOR,
        uplo_char(uplo),
        n as i32,
        nrhs as i32,
        a.as_ptr() as *const sys::__BindgenComplex<f32>,
        lda as i32,
        ipiv.as_ptr(),
        b.as_mut_ptr() as *mut sys::__BindgenComplex<f32>,
        ldb as i32
    )
}
/// `Complex64` inverse from [`zhetrf`].
pub fn zhetri(uplo: Uplo, n: usize, a: &mut [Complex64], lda: usize, ipiv: &[i32]) -> Result<()> {
    lapack_call!(
        "lapack",
        LAPACKE_zhetri,
        ROW_MAJOR,
        uplo_char(uplo),
        n as i32,
        a.as_mut_ptr() as *mut sys::__BindgenComplex<f64>,
        lda as i32,
        ipiv.as_ptr()
    )
}
/// `Complex32` inverse from [`chetrf`].
pub fn chetri(uplo: Uplo, n: usize, a: &mut [Complex32], lda: usize, ipiv: &[i32]) -> Result<()> {
    lapack_call!(
        "lapack",
        LAPACKE_chetri,
        ROW_MAJOR,
        uplo_char(uplo),
        n as i32,
        a.as_mut_ptr() as *mut sys::__BindgenComplex<f32>,
        lda as i32,
        ipiv.as_ptr()
    )
}

// ----- gelss: SVD-based least-squares -----

/// SVD-based least-squares solve. Returns the rank found. (`f64`)
#[allow(clippy::too_many_arguments)]
pub fn dgelss(
    m: usize,
    n: usize,
    nrhs: usize,
    a: &mut [f64],
    lda: usize,
    b: &mut [f64],
    ldb: usize,
    s: &mut [f64],
    rcond: f64,
) -> Result<usize> {
    let mut rank: i32 = 0;
    let info = unsafe {
        sys::LAPACKE_dgelss(
            ROW_MAJOR,
            m as i32,
            n as i32,
            nrhs as i32,
            a.as_mut_ptr(),
            lda as i32,
            b.as_mut_ptr(),
            ldb as i32,
            s.as_mut_ptr(),
            rcond,
            &mut rank,
        )
    };
    map_info("lapack", info)?;
    Ok(rank as usize)
}
/// `f32` SVD-LSQ. See [`dgelss`].
#[allow(clippy::too_many_arguments)]
pub fn sgelss(
    m: usize,
    n: usize,
    nrhs: usize,
    a: &mut [f32],
    lda: usize,
    b: &mut [f32],
    ldb: usize,
    s: &mut [f32],
    rcond: f32,
) -> Result<usize> {
    let mut rank: i32 = 0;
    let info = unsafe {
        sys::LAPACKE_sgelss(
            ROW_MAJOR,
            m as i32,
            n as i32,
            nrhs as i32,
            a.as_mut_ptr(),
            lda as i32,
            b.as_mut_ptr(),
            ldb as i32,
            s.as_mut_ptr(),
            rcond,
            &mut rank,
        )
    };
    map_info("lapack", info)?;
    Ok(rank as usize)
}
/// `Complex64` SVD-LSQ. See [`dgelss`].
#[allow(clippy::too_many_arguments)]
pub fn zgelss(
    m: usize,
    n: usize,
    nrhs: usize,
    a: &mut [Complex64],
    lda: usize,
    b: &mut [Complex64],
    ldb: usize,
    s: &mut [f64],
    rcond: f64,
) -> Result<usize> {
    let mut rank: i32 = 0;
    let info = unsafe {
        sys::LAPACKE_zgelss(
            ROW_MAJOR,
            m as i32,
            n as i32,
            nrhs as i32,
            a.as_mut_ptr() as *mut sys::__BindgenComplex<f64>,
            lda as i32,
            b.as_mut_ptr() as *mut sys::__BindgenComplex<f64>,
            ldb as i32,
            s.as_mut_ptr(),
            rcond,
            &mut rank,
        )
    };
    map_info("lapack", info)?;
    Ok(rank as usize)
}
/// `Complex32` SVD-LSQ. See [`dgelss`].
#[allow(clippy::too_many_arguments)]
pub fn cgelss(
    m: usize,
    n: usize,
    nrhs: usize,
    a: &mut [Complex32],
    lda: usize,
    b: &mut [Complex32],
    ldb: usize,
    s: &mut [f32],
    rcond: f32,
) -> Result<usize> {
    let mut rank: i32 = 0;
    let info = unsafe {
        sys::LAPACKE_cgelss(
            ROW_MAJOR,
            m as i32,
            n as i32,
            nrhs as i32,
            a.as_mut_ptr() as *mut sys::__BindgenComplex<f32>,
            lda as i32,
            b.as_mut_ptr() as *mut sys::__BindgenComplex<f32>,
            ldb as i32,
            s.as_mut_ptr(),
            rcond,
            &mut rank,
        )
    };
    map_info("lapack", info)?;
    Ok(rank as usize)
}

// ----- ormqr / unmqr: apply Q from QR factorisation -----

/// Apply `Q` from [`geqrf`] to `C`: `C := op(Q) · C` or `C · op(Q)`.
/// `side = b'L'` for left, `b'R'` for right. `trans = b'N'` or `b'T'`. (`f64`)
#[allow(clippy::too_many_arguments)]
pub fn dormqr(
    side_l: bool,
    trans: Trans,
    m: usize,
    n: usize,
    k: usize,
    a: &[f64],
    lda: usize,
    tau: &[f64],
    c: &mut [f64],
    ldc: usize,
) -> Result<()> {
    let side = if side_l { b'L' } else { b'R' } as c_char;
    lapack_call!(
        "lapack",
        LAPACKE_dormqr,
        ROW_MAJOR,
        side,
        trans_char(trans),
        m as i32,
        n as i32,
        k as i32,
        a.as_ptr(),
        lda as i32,
        tau.as_ptr(),
        c.as_mut_ptr(),
        ldc as i32
    )
}
/// `f32` apply Q from QR. See [`dormqr`].
#[allow(clippy::too_many_arguments)]
pub fn sormqr(
    side_l: bool,
    trans: Trans,
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    lda: usize,
    tau: &[f32],
    c: &mut [f32],
    ldc: usize,
) -> Result<()> {
    let side = if side_l { b'L' } else { b'R' } as c_char;
    lapack_call!(
        "lapack",
        LAPACKE_sormqr,
        ROW_MAJOR,
        side,
        trans_char(trans),
        m as i32,
        n as i32,
        k as i32,
        a.as_ptr(),
        lda as i32,
        tau.as_ptr(),
        c.as_mut_ptr(),
        ldc as i32
    )
}
/// `Complex64` apply Q from QR. `trans = b'N'` or `b'C'` (conj-transpose).
#[allow(clippy::too_many_arguments)]
pub fn zunmqr(
    side_l: bool,
    trans: Trans,
    m: usize,
    n: usize,
    k: usize,
    a: &[Complex64],
    lda: usize,
    tau: &[Complex64],
    c: &mut [Complex64],
    ldc: usize,
) -> Result<()> {
    let side = if side_l { b'L' } else { b'R' } as c_char;
    lapack_call!(
        "lapack",
        LAPACKE_zunmqr,
        ROW_MAJOR,
        side,
        trans_char(trans),
        m as i32,
        n as i32,
        k as i32,
        a.as_ptr() as *const sys::__BindgenComplex<f64>,
        lda as i32,
        tau.as_ptr() as *const sys::__BindgenComplex<f64>,
        c.as_mut_ptr() as *mut sys::__BindgenComplex<f64>,
        ldc as i32
    )
}
/// `Complex32` apply Q from QR. See [`zunmqr`].
#[allow(clippy::too_many_arguments)]
pub fn cunmqr(
    side_l: bool,
    trans: Trans,
    m: usize,
    n: usize,
    k: usize,
    a: &[Complex32],
    lda: usize,
    tau: &[Complex32],
    c: &mut [Complex32],
    ldc: usize,
) -> Result<()> {
    let side = if side_l { b'L' } else { b'R' } as c_char;
    lapack_call!(
        "lapack",
        LAPACKE_cunmqr,
        ROW_MAJOR,
        side,
        trans_char(trans),
        m as i32,
        n as i32,
        k as i32,
        a.as_ptr() as *const sys::__BindgenComplex<f32>,
        lda as i32,
        tau.as_ptr() as *const sys::__BindgenComplex<f32>,
        c.as_mut_ptr() as *mut sys::__BindgenComplex<f32>,
        ldc as i32
    )
}

// ----- orglq / unglq + ormlq / unmlq: form/apply Q from LQ -----

/// Form `Q` from [`gelqf`]. See [`dorgqr`] for analogous QR. (`f64`)
pub fn dorglq(m: usize, n: usize, k: usize, a: &mut [f64], lda: usize, tau: &[f64]) -> Result<()> {
    lapack_call!(
        "lapack",
        LAPACKE_dorglq,
        ROW_MAJOR,
        m as i32,
        n as i32,
        k as i32,
        a.as_mut_ptr(),
        lda as i32,
        tau.as_ptr()
    )
}
/// `f32` form Q from LQ. See [`dorglq`].
pub fn sorglq(m: usize, n: usize, k: usize, a: &mut [f32], lda: usize, tau: &[f32]) -> Result<()> {
    lapack_call!(
        "lapack",
        LAPACKE_sorglq,
        ROW_MAJOR,
        m as i32,
        n as i32,
        k as i32,
        a.as_mut_ptr(),
        lda as i32,
        tau.as_ptr()
    )
}
/// `Complex64` form Q from LQ. See [`dorglq`].
pub fn zunglq(
    m: usize,
    n: usize,
    k: usize,
    a: &mut [Complex64],
    lda: usize,
    tau: &[Complex64],
) -> Result<()> {
    lapack_call!(
        "lapack",
        LAPACKE_zunglq,
        ROW_MAJOR,
        m as i32,
        n as i32,
        k as i32,
        a.as_mut_ptr() as *mut sys::__BindgenComplex<f64>,
        lda as i32,
        tau.as_ptr() as *const sys::__BindgenComplex<f64>
    )
}
/// `Complex32` form Q from LQ. See [`dorglq`].
pub fn cunglq(
    m: usize,
    n: usize,
    k: usize,
    a: &mut [Complex32],
    lda: usize,
    tau: &[Complex32],
) -> Result<()> {
    lapack_call!(
        "lapack",
        LAPACKE_cunglq,
        ROW_MAJOR,
        m as i32,
        n as i32,
        k as i32,
        a.as_mut_ptr() as *mut sys::__BindgenComplex<f32>,
        lda as i32,
        tau.as_ptr() as *const sys::__BindgenComplex<f32>
    )
}

/// Apply Q from LQ. (`f64`)
#[allow(clippy::too_many_arguments)]
pub fn dormlq(
    side_l: bool,
    trans: Trans,
    m: usize,
    n: usize,
    k: usize,
    a: &[f64],
    lda: usize,
    tau: &[f64],
    c: &mut [f64],
    ldc: usize,
) -> Result<()> {
    let side = if side_l { b'L' } else { b'R' } as c_char;
    lapack_call!(
        "lapack",
        LAPACKE_dormlq,
        ROW_MAJOR,
        side,
        trans_char(trans),
        m as i32,
        n as i32,
        k as i32,
        a.as_ptr(),
        lda as i32,
        tau.as_ptr(),
        c.as_mut_ptr(),
        ldc as i32
    )
}
/// `f32` apply Q from LQ. See [`dormlq`].
#[allow(clippy::too_many_arguments)]
pub fn sormlq(
    side_l: bool,
    trans: Trans,
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    lda: usize,
    tau: &[f32],
    c: &mut [f32],
    ldc: usize,
) -> Result<()> {
    let side = if side_l { b'L' } else { b'R' } as c_char;
    lapack_call!(
        "lapack",
        LAPACKE_sormlq,
        ROW_MAJOR,
        side,
        trans_char(trans),
        m as i32,
        n as i32,
        k as i32,
        a.as_ptr(),
        lda as i32,
        tau.as_ptr(),
        c.as_mut_ptr(),
        ldc as i32
    )
}
/// `Complex64` apply Q from LQ. See [`dormlq`].
#[allow(clippy::too_many_arguments)]
pub fn zunmlq(
    side_l: bool,
    trans: Trans,
    m: usize,
    n: usize,
    k: usize,
    a: &[Complex64],
    lda: usize,
    tau: &[Complex64],
    c: &mut [Complex64],
    ldc: usize,
) -> Result<()> {
    let side = if side_l { b'L' } else { b'R' } as c_char;
    lapack_call!(
        "lapack",
        LAPACKE_zunmlq,
        ROW_MAJOR,
        side,
        trans_char(trans),
        m as i32,
        n as i32,
        k as i32,
        a.as_ptr() as *const sys::__BindgenComplex<f64>,
        lda as i32,
        tau.as_ptr() as *const sys::__BindgenComplex<f64>,
        c.as_mut_ptr() as *mut sys::__BindgenComplex<f64>,
        ldc as i32
    )
}
/// `Complex32` apply Q from LQ. See [`dormlq`].
#[allow(clippy::too_many_arguments)]
pub fn cunmlq(
    side_l: bool,
    trans: Trans,
    m: usize,
    n: usize,
    k: usize,
    a: &[Complex32],
    lda: usize,
    tau: &[Complex32],
    c: &mut [Complex32],
    ldc: usize,
) -> Result<()> {
    let side = if side_l { b'L' } else { b'R' } as c_char;
    lapack_call!(
        "lapack",
        LAPACKE_cunmlq,
        ROW_MAJOR,
        side,
        trans_char(trans),
        m as i32,
        n as i32,
        k as i32,
        a.as_ptr() as *const sys::__BindgenComplex<f32>,
        lda as i32,
        tau.as_ptr() as *const sys::__BindgenComplex<f32>,
        c.as_mut_ptr() as *mut sys::__BindgenComplex<f32>,
        ldc as i32
    )
}

// ----- gehrd / gebrd: Hessenberg / bidiagonal reduction -----

/// Reduce general matrix to upper Hessenberg form. (`f64`)
pub fn dgehrd(
    n: usize,
    ilo: i32,
    ihi: i32,
    a: &mut [f64],
    lda: usize,
    tau: &mut [f64],
) -> Result<()> {
    lapack_call!(
        "lapack",
        LAPACKE_dgehrd,
        ROW_MAJOR,
        n as i32,
        ilo,
        ihi,
        a.as_mut_ptr(),
        lda as i32,
        tau.as_mut_ptr()
    )
}
/// `f32` Hessenberg reduction. See [`dgehrd`].
pub fn sgehrd(
    n: usize,
    ilo: i32,
    ihi: i32,
    a: &mut [f32],
    lda: usize,
    tau: &mut [f32],
) -> Result<()> {
    lapack_call!(
        "lapack",
        LAPACKE_sgehrd,
        ROW_MAJOR,
        n as i32,
        ilo,
        ihi,
        a.as_mut_ptr(),
        lda as i32,
        tau.as_mut_ptr()
    )
}
/// `Complex64` Hessenberg reduction. See [`dgehrd`].
pub fn zgehrd(
    n: usize,
    ilo: i32,
    ihi: i32,
    a: &mut [Complex64],
    lda: usize,
    tau: &mut [Complex64],
) -> Result<()> {
    lapack_call!(
        "lapack",
        LAPACKE_zgehrd,
        ROW_MAJOR,
        n as i32,
        ilo,
        ihi,
        a.as_mut_ptr() as *mut sys::__BindgenComplex<f64>,
        lda as i32,
        tau.as_mut_ptr() as *mut sys::__BindgenComplex<f64>
    )
}
/// `Complex32` Hessenberg reduction. See [`dgehrd`].
pub fn cgehrd(
    n: usize,
    ilo: i32,
    ihi: i32,
    a: &mut [Complex32],
    lda: usize,
    tau: &mut [Complex32],
) -> Result<()> {
    lapack_call!(
        "lapack",
        LAPACKE_cgehrd,
        ROW_MAJOR,
        n as i32,
        ilo,
        ihi,
        a.as_mut_ptr() as *mut sys::__BindgenComplex<f32>,
        lda as i32,
        tau.as_mut_ptr() as *mut sys::__BindgenComplex<f32>
    )
}

/// Reduce general matrix to bidiagonal form (preprocess for SVD). (`f64`)
#[allow(clippy::too_many_arguments)]
pub fn dgebrd(
    m: usize,
    n: usize,
    a: &mut [f64],
    lda: usize,
    d: &mut [f64],
    e: &mut [f64],
    tauq: &mut [f64],
    taup: &mut [f64],
) -> Result<()> {
    lapack_call!(
        "lapack",
        LAPACKE_dgebrd,
        ROW_MAJOR,
        m as i32,
        n as i32,
        a.as_mut_ptr(),
        lda as i32,
        d.as_mut_ptr(),
        e.as_mut_ptr(),
        tauq.as_mut_ptr(),
        taup.as_mut_ptr()
    )
}
/// `f32` bidiag reduction. See [`dgebrd`].
#[allow(clippy::too_many_arguments)]
pub fn sgebrd(
    m: usize,
    n: usize,
    a: &mut [f32],
    lda: usize,
    d: &mut [f32],
    e: &mut [f32],
    tauq: &mut [f32],
    taup: &mut [f32],
) -> Result<()> {
    lapack_call!(
        "lapack",
        LAPACKE_sgebrd,
        ROW_MAJOR,
        m as i32,
        n as i32,
        a.as_mut_ptr(),
        lda as i32,
        d.as_mut_ptr(),
        e.as_mut_ptr(),
        tauq.as_mut_ptr(),
        taup.as_mut_ptr()
    )
}
/// `Complex64` bidiag reduction. `d` and `e` are real.
#[allow(clippy::too_many_arguments)]
pub fn zgebrd(
    m: usize,
    n: usize,
    a: &mut [Complex64],
    lda: usize,
    d: &mut [f64],
    e: &mut [f64],
    tauq: &mut [Complex64],
    taup: &mut [Complex64],
) -> Result<()> {
    lapack_call!(
        "lapack",
        LAPACKE_zgebrd,
        ROW_MAJOR,
        m as i32,
        n as i32,
        a.as_mut_ptr() as *mut sys::__BindgenComplex<f64>,
        lda as i32,
        d.as_mut_ptr(),
        e.as_mut_ptr(),
        tauq.as_mut_ptr() as *mut sys::__BindgenComplex<f64>,
        taup.as_mut_ptr() as *mut sys::__BindgenComplex<f64>
    )
}
/// `Complex32` bidiag reduction. See [`zgebrd`].
#[allow(clippy::too_many_arguments)]
pub fn cgebrd(
    m: usize,
    n: usize,
    a: &mut [Complex32],
    lda: usize,
    d: &mut [f32],
    e: &mut [f32],
    tauq: &mut [Complex32],
    taup: &mut [Complex32],
) -> Result<()> {
    lapack_call!(
        "lapack",
        LAPACKE_cgebrd,
        ROW_MAJOR,
        m as i32,
        n as i32,
        a.as_mut_ptr() as *mut sys::__BindgenComplex<f32>,
        lda as i32,
        d.as_mut_ptr(),
        e.as_mut_ptr(),
        tauq.as_mut_ptr() as *mut sys::__BindgenComplex<f32>,
        taup.as_mut_ptr() as *mut sys::__BindgenComplex<f32>
    )
}

// ----- orgtr/ungtr: form Q from Hessenberg/tridiag reduction -----

/// Form `Q` from a tridiagonal reduction. (`f64`)
pub fn dorgtr(uplo: Uplo, n: usize, a: &mut [f64], lda: usize, tau: &[f64]) -> Result<()> {
    lapack_call!(
        "lapack",
        LAPACKE_dorgtr,
        ROW_MAJOR,
        uplo_char(uplo),
        n as i32,
        a.as_mut_ptr(),
        lda as i32,
        tau.as_ptr()
    )
}
/// `f32` form Q from tridiag reduction.
pub fn sorgtr(uplo: Uplo, n: usize, a: &mut [f32], lda: usize, tau: &[f32]) -> Result<()> {
    lapack_call!(
        "lapack",
        LAPACKE_sorgtr,
        ROW_MAJOR,
        uplo_char(uplo),
        n as i32,
        a.as_mut_ptr(),
        lda as i32,
        tau.as_ptr()
    )
}
/// `Complex64` form Q from Hermitian tridiag reduction.
pub fn zungtr(
    uplo: Uplo,
    n: usize,
    a: &mut [Complex64],
    lda: usize,
    tau: &[Complex64],
) -> Result<()> {
    lapack_call!(
        "lapack",
        LAPACKE_zungtr,
        ROW_MAJOR,
        uplo_char(uplo),
        n as i32,
        a.as_mut_ptr() as *mut sys::__BindgenComplex<f64>,
        lda as i32,
        tau.as_ptr() as *const sys::__BindgenComplex<f64>
    )
}
/// `Complex32` form Q from Hermitian tridiag reduction.
pub fn cungtr(
    uplo: Uplo,
    n: usize,
    a: &mut [Complex32],
    lda: usize,
    tau: &[Complex32],
) -> Result<()> {
    lapack_call!(
        "lapack",
        LAPACKE_cungtr,
        ROW_MAJOR,
        uplo_char(uplo),
        n as i32,
        a.as_mut_ptr() as *mut sys::__BindgenComplex<f32>,
        lda as i32,
        tau.as_ptr() as *const sys::__BindgenComplex<f32>
    )
}

// ----- orgbr/ungbr: form Q (or P) from bidiag reduction -----

/// Which factor of a bidiag reduction to form: Q (left) or P (right).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BidiagVect {
    Q,
    P,
}
impl BidiagVect {
    fn raw(self) -> c_char {
        match self {
            BidiagVect::Q => b'Q' as c_char,
            BidiagVect::P => b'P' as c_char,
        }
    }
}

/// Form `Q` (or `P`) from [`dgebrd`]. (`f64`)
pub fn dorgbr(
    vect: BidiagVect,
    m: usize,
    n: usize,
    k: usize,
    a: &mut [f64],
    lda: usize,
    tau: &[f64],
) -> Result<()> {
    lapack_call!(
        "lapack",
        LAPACKE_dorgbr,
        ROW_MAJOR,
        vect.raw(),
        m as i32,
        n as i32,
        k as i32,
        a.as_mut_ptr(),
        lda as i32,
        tau.as_ptr()
    )
}
/// `f32` form Q/P from bidiag reduction.
pub fn sorgbr(
    vect: BidiagVect,
    m: usize,
    n: usize,
    k: usize,
    a: &mut [f32],
    lda: usize,
    tau: &[f32],
) -> Result<()> {
    lapack_call!(
        "lapack",
        LAPACKE_sorgbr,
        ROW_MAJOR,
        vect.raw(),
        m as i32,
        n as i32,
        k as i32,
        a.as_mut_ptr(),
        lda as i32,
        tau.as_ptr()
    )
}
/// `Complex64` form Q/P from bidiag reduction.
pub fn zungbr(
    vect: BidiagVect,
    m: usize,
    n: usize,
    k: usize,
    a: &mut [Complex64],
    lda: usize,
    tau: &[Complex64],
) -> Result<()> {
    lapack_call!(
        "lapack",
        LAPACKE_zungbr,
        ROW_MAJOR,
        vect.raw(),
        m as i32,
        n as i32,
        k as i32,
        a.as_mut_ptr() as *mut sys::__BindgenComplex<f64>,
        lda as i32,
        tau.as_ptr() as *const sys::__BindgenComplex<f64>
    )
}
/// `Complex32` form Q/P from bidiag reduction.
pub fn cungbr(
    vect: BidiagVect,
    m: usize,
    n: usize,
    k: usize,
    a: &mut [Complex32],
    lda: usize,
    tau: &[Complex32],
) -> Result<()> {
    lapack_call!(
        "lapack",
        LAPACKE_cungbr,
        ROW_MAJOR,
        vect.raw(),
        m as i32,
        n as i32,
        k as i32,
        a.as_mut_ptr() as *mut sys::__BindgenComplex<f32>,
        lda as i32,
        tau.as_ptr() as *const sys::__BindgenComplex<f32>
    )
}

// ----- gecon/pocon: condition-number estimators -----

/// Reciprocal condition number of a general matrix (after [`getrf`]).
/// `anorm` is the input matrix's `Norm::One` or `Norm::Inf` value
/// (compute via [`dlange`]). (`f64`)
pub fn dgecon(norm: Norm, n: usize, a: &[f64], lda: usize, anorm: f64) -> Result<f64> {
    let mut rcond: f64 = 0.0;
    let info = unsafe {
        sys::LAPACKE_dgecon(
            ROW_MAJOR,
            norm.raw(),
            n as i32,
            a.as_ptr(),
            lda as i32,
            anorm,
            &mut rcond,
        )
    };
    map_info("lapack", info)?;
    Ok(rcond)
}
/// `f32` general condition number. See [`dgecon`].
pub fn sgecon(norm: Norm, n: usize, a: &[f32], lda: usize, anorm: f32) -> Result<f32> {
    let mut rcond: f32 = 0.0;
    let info = unsafe {
        sys::LAPACKE_sgecon(
            ROW_MAJOR,
            norm.raw(),
            n as i32,
            a.as_ptr(),
            lda as i32,
            anorm,
            &mut rcond,
        )
    };
    map_info("lapack", info)?;
    Ok(rcond)
}
/// `Complex64` general condition number. See [`dgecon`].
pub fn zgecon(norm: Norm, n: usize, a: &[Complex64], lda: usize, anorm: f64) -> Result<f64> {
    let mut rcond: f64 = 0.0;
    let info = unsafe {
        sys::LAPACKE_zgecon(
            ROW_MAJOR,
            norm.raw(),
            n as i32,
            a.as_ptr() as *const sys::__BindgenComplex<f64>,
            lda as i32,
            anorm,
            &mut rcond,
        )
    };
    map_info("lapack", info)?;
    Ok(rcond)
}
/// `Complex32` general condition number. See [`dgecon`].
pub fn cgecon(norm: Norm, n: usize, a: &[Complex32], lda: usize, anorm: f32) -> Result<f32> {
    let mut rcond: f32 = 0.0;
    let info = unsafe {
        sys::LAPACKE_cgecon(
            ROW_MAJOR,
            norm.raw(),
            n as i32,
            a.as_ptr() as *const sys::__BindgenComplex<f32>,
            lda as i32,
            anorm,
            &mut rcond,
        )
    };
    map_info("lapack", info)?;
    Ok(rcond)
}

/// Reciprocal condition number of a SPD matrix (after [`potrf`]). (`f64`)
pub fn dpocon(uplo: Uplo, n: usize, a: &[f64], lda: usize, anorm: f64) -> Result<f64> {
    let mut rcond: f64 = 0.0;
    let info = unsafe {
        sys::LAPACKE_dpocon(
            ROW_MAJOR,
            uplo_char(uplo),
            n as i32,
            a.as_ptr(),
            lda as i32,
            anorm,
            &mut rcond,
        )
    };
    map_info("lapack", info)?;
    Ok(rcond)
}
/// `f32` SPD condition number. See [`dpocon`].
pub fn spocon(uplo: Uplo, n: usize, a: &[f32], lda: usize, anorm: f32) -> Result<f32> {
    let mut rcond: f32 = 0.0;
    let info = unsafe {
        sys::LAPACKE_spocon(
            ROW_MAJOR,
            uplo_char(uplo),
            n as i32,
            a.as_ptr(),
            lda as i32,
            anorm,
            &mut rcond,
        )
    };
    map_info("lapack", info)?;
    Ok(rcond)
}
/// `Complex64` SPD/Hermitian condition number. See [`dpocon`].
pub fn zpocon(uplo: Uplo, n: usize, a: &[Complex64], lda: usize, anorm: f64) -> Result<f64> {
    let mut rcond: f64 = 0.0;
    let info = unsafe {
        sys::LAPACKE_zpocon(
            ROW_MAJOR,
            uplo_char(uplo),
            n as i32,
            a.as_ptr() as *const sys::__BindgenComplex<f64>,
            lda as i32,
            anorm,
            &mut rcond,
        )
    };
    map_info("lapack", info)?;
    Ok(rcond)
}
/// `Complex32` SPD/Hermitian condition number. See [`dpocon`].
pub fn cpocon(uplo: Uplo, n: usize, a: &[Complex32], lda: usize, anorm: f32) -> Result<f32> {
    let mut rcond: f32 = 0.0;
    let info = unsafe {
        sys::LAPACKE_cpocon(
            ROW_MAJOR,
            uplo_char(uplo),
            n as i32,
            a.as_ptr() as *const sys::__BindgenComplex<f32>,
            lda as i32,
            anorm,
            &mut rcond,
        )
    };
    map_info("lapack", info)?;
    Ok(rcond)
}

// ----- larfg: generate elementary Householder reflector -----

/// Generate an elementary Householder reflector `H = I − τ·v·vᵀ` such
/// that `H · (α, x)ᵀ = (β, 0…0)ᵀ`. On exit `*alpha = β` and `tau` is
/// returned. (`f64`)
pub fn dlarfg(n: usize, alpha: &mut f64, x: &mut [f64], incx: i32) -> Result<f64> {
    let mut tau: f64 = 0.0;
    let info = unsafe { sys::LAPACKE_dlarfg(n as i32, alpha, x.as_mut_ptr(), incx, &mut tau) };
    map_info("lapack", info)?;
    Ok(tau)
}
/// `f32` Householder generator. See [`dlarfg`].
pub fn slarfg(n: usize, alpha: &mut f32, x: &mut [f32], incx: i32) -> Result<f32> {
    let mut tau: f32 = 0.0;
    let info = unsafe { sys::LAPACKE_slarfg(n as i32, alpha, x.as_mut_ptr(), incx, &mut tau) };
    map_info("lapack", info)?;
    Ok(tau)
}
/// `Complex64` Householder generator. See [`dlarfg`].
pub fn zlarfg(
    n: usize,
    alpha: &mut Complex64,
    x: &mut [Complex64],
    incx: i32,
) -> Result<Complex64> {
    let mut tau = Complex64::ZERO;
    let info = unsafe {
        sys::LAPACKE_zlarfg(
            n as i32,
            alpha as *mut _ as *mut sys::__BindgenComplex<f64>,
            x.as_mut_ptr() as *mut sys::__BindgenComplex<f64>,
            incx,
            &mut tau as *mut _ as *mut sys::__BindgenComplex<f64>,
        )
    };
    map_info("lapack", info)?;
    Ok(tau)
}
/// `Complex32` Householder generator. See [`dlarfg`].
pub fn clarfg(
    n: usize,
    alpha: &mut Complex32,
    x: &mut [Complex32],
    incx: i32,
) -> Result<Complex32> {
    let mut tau = Complex32::ZERO;
    let info = unsafe {
        sys::LAPACKE_clarfg(
            n as i32,
            alpha as *mut _ as *mut sys::__BindgenComplex<f32>,
            x.as_mut_ptr() as *mut sys::__BindgenComplex<f32>,
            incx,
            &mut tau as *mut _ as *mut sys::__BindgenComplex<f32>,
        )
    };
    map_info("lapack", info)?;
    Ok(tau)
}

// ----- lasrt: sort an array (real only) -----

/// Sort a real vector in increasing (`b'I'`) or decreasing (`b'D'`)
/// order. `id` is the sort direction char. (`f64`)
pub fn dlasrt(id: u8, d: &mut [f64]) -> Result<()> {
    lapack_call!(
        "lapack",
        LAPACKE_dlasrt,
        id as c_char,
        d.len() as i32,
        d.as_mut_ptr()
    )
}
/// `f32` sort. See [`dlasrt`].
pub fn slasrt(id: u8, d: &mut [f32]) -> Result<()> {
    lapack_call!(
        "lapack",
        LAPACKE_slasrt,
        id as c_char,
        d.len() as i32,
        d.as_mut_ptr()
    )
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
        for v in &b {
            approx(*v, 1.0, 1e-12);
        }
    }

    #[test]
    fn ptsv_solve_real() {
        let mut d = [2.0_f64, 2.0, 2.0];
        let mut e = [1.0_f64, 1.0];
        let mut b = [3.0_f64, 4.0, 3.0];
        ptsv::<f64>(3, &mut d, &mut e, &mut b).unwrap();
        for v in &b {
            approx(*v, 1.0, 1e-12);
        }
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
        let mut a = [
            Complex64::ONE,
            Complex64::ZERO,
            Complex64::ZERO,
            Complex64::ONE,
        ];
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
        let mut a = [3.0_f64, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0];
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
            Complex64::new(2.0, 0.0),
            Complex64::new(0.0, 1.0),
            Complex64::ZERO,
            Complex64::new(2.0, 0.0),
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
            SvdJob::All,
            SvdJob::All,
            2,
            2,
            &mut a,
            &mut s,
            &mut u,
            2,
            &mut vt,
            2,
            &mut superb,
        )
        .unwrap();
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
