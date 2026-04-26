//! Safe wrappers for AOCL-LAPACK (libFLAME) via the LAPACKE C interface.

#![warn(missing_debug_implementations)]
#![cfg_attr(docsrs, feature(doc_cfg))]

use aocl_lapack_sys as sys;
pub use aocl_error::{Error, Result};
pub use aocl_types::Layout;
use aocl_types::sealed::Sealed;

const ROW_MAJOR: i32 = 101;
const COL_MAJOR: i32 = 102;

fn layout_raw(l: Layout) -> i32 {
    match l {
        Layout::RowMajor => ROW_MAJOR,
        Layout::ColMajor => COL_MAJOR,
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
            message: format!(
                "factorization completed but matrix is singular: U({i},{i}) = 0",
                i = info
            ),
        }),
    }
}

/// Scalar element type usable with the wrapped LAPACK routines.
pub trait Scalar: Copy + Sized + Sealed {
    /// Solve `A · X = B` for X, overwriting `B` with the solution and `A`
    /// with its LU factorization.
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

    /// LU factorization of an `m × n` matrix `A` with partial pivoting.
    fn getrf(
        layout: Layout,
        m: usize,
        n: usize,
        a: &mut [Self],
        lda: usize,
        ipiv: &mut [i32],
    ) -> Result<()>;
}

fn check_matrix_len(
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

macro_rules! impl_scalar {
    ($t:ty, $gesv:ident, $getrf:ident) => {
        impl Scalar for $t {
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
                check_matrix_len("gesv: A", layout, n, n, lda, a.len())?;
                check_matrix_len("gesv: B", layout, n, nrhs, ldb, b.len())?;
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
                check_matrix_len("getrf: A", layout, m, n, lda, a.len())?;
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
        }
    };
}

impl_scalar!(f32, LAPACKE_sgesv, LAPACKE_sgetrf);
impl_scalar!(f64, LAPACKE_dgesv, LAPACKE_dgetrf);

/// Solve `A · X = B` for tightly-packed row-major matrices.
pub fn gesv<T: Scalar>(n: usize, a: &mut [T], ipiv: &mut [i32], b: &mut [T]) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    let nrhs = b.len() / n;
    T::gesv(Layout::RowMajor, n, nrhs, a, n, ipiv, b, nrhs.max(1))
}

/// LU factorization of a tightly-packed row-major `m × n` matrix.
pub fn getrf<T: Scalar>(m: usize, n: usize, a: &mut [T], ipiv: &mut [i32]) -> Result<()> {
    T::getrf(Layout::RowMajor, m, n, a, n, ipiv)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gesv_2x2() {
        let mut a = [2.0_f64, 1.0, 1.0, 3.0];
        let mut b = [1.0_f64, 2.0];
        let mut ipiv = [0_i32; 2];
        gesv(2, &mut a, &mut ipiv, &mut b).unwrap();
        assert!((b[0] - 0.2).abs() < 1e-12, "got {}", b[0]);
        assert!((b[1] - 0.6).abs() < 1e-12, "got {}", b[1]);
    }

    #[test]
    fn gesv_singular() {
        let mut a = [1.0_f64, 2.0, 2.0, 4.0];
        let mut b = [3.0_f64, 6.0];
        let mut ipiv = [0_i32; 2];
        let err = gesv(2, &mut a, &mut ipiv, &mut b).unwrap_err();
        match err {
            Error::Status { code, .. } => assert!(code > 0, "expected positive info, got {code}"),
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn getrf_then_back_solve_via_dim_check() {
        let mut a = [4.0_f64, 3.0, 6.0, 3.0];
        let mut ipiv = [0_i32; 2];
        getrf(2, 2, &mut a, &mut ipiv).unwrap();
        assert!(ipiv.iter().all(|&p| p > 0));
    }

    #[test]
    fn dim_mismatch_is_error() {
        let mut a = [1.0_f64; 3];
        let mut ipiv = [0_i32; 2];
        let mut b = [1.0_f64, 2.0];
        let err = gesv(2, &mut a, &mut ipiv, &mut b).unwrap_err();
        matches!(err, Error::InvalidArgument(_));
    }
}
