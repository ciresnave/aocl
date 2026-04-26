//! Safe wrappers for AOCL-Sparse.

#![warn(missing_debug_implementations)]
#![cfg_attr(docsrs, feature(doc_cfg))]

use aocl_sparse_sys as sys;
pub use aocl_error::{Error, Result};
pub use aocl_types::Trans;
use aocl_types::sealed::Sealed;

fn trans_raw(t: Trans) -> sys::aoclsparse_operation {
    match t {
        Trans::No => sys::aoclsparse_operation__aoclsparse_operation_none,
        Trans::T => sys::aoclsparse_operation__aoclsparse_operation_transpose,
        Trans::C => sys::aoclsparse_operation__aoclsparse_operation_conjugate_transpose,
    }
}

fn check_status(component: &'static str, status: sys::aoclsparse_status) -> Result<()> {
    if status == sys::aoclsparse_status__aoclsparse_status_success {
        return Ok(());
    }
    let message = match status {
        s if s == sys::aoclsparse_status__aoclsparse_status_not_implemented => "not implemented",
        s if s == sys::aoclsparse_status__aoclsparse_status_invalid_pointer => "invalid pointer",
        s if s == sys::aoclsparse_status__aoclsparse_status_invalid_size => "invalid size",
        s if s == sys::aoclsparse_status__aoclsparse_status_internal_error => "internal error",
        s if s == sys::aoclsparse_status__aoclsparse_status_invalid_value => "invalid value",
        s if s == sys::aoclsparse_status__aoclsparse_status_invalid_index_value => {
            "invalid index value"
        }
        s if s == sys::aoclsparse_status__aoclsparse_status_maxit => "max iterations reached",
        s if s == sys::aoclsparse_status__aoclsparse_status_user_stop => "user stop",
        s if s == sys::aoclsparse_status__aoclsparse_status_wrong_type => "wrong type",
        s if s == sys::aoclsparse_status__aoclsparse_status_memory_error => "memory error",
        _ => "unknown sparse status",
    }
    .to_string();
    Err(Error::Status {
        component,
        code: status as i64,
        message,
    })
}

/// RAII wrapper for `aoclsparse_mat_descr`.
pub struct MatDescr {
    raw: sys::aoclsparse_mat_descr,
}

impl MatDescr {
    /// Create a fresh descriptor with library defaults.
    pub fn new() -> Result<Self> {
        let mut raw: sys::aoclsparse_mat_descr = std::ptr::null_mut();
        let status = unsafe { sys::aoclsparse_create_mat_descr(&mut raw) };
        check_status("sparse", status)?;
        if raw.is_null() {
            return Err(Error::AllocationFailed("sparse"));
        }
        Ok(MatDescr { raw })
    }

    /// Borrow the underlying handle for raw FFI calls.
    ///
    /// # Safety
    /// The returned pointer is valid only for the lifetime of `self`.
    /// Do not call `aoclsparse_destroy_mat_descr` on it.
    pub fn as_raw(&self) -> sys::aoclsparse_mat_descr {
        self.raw
    }
}

impl Drop for MatDescr {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                let _ = sys::aoclsparse_destroy_mat_descr(self.raw);
            }
            self.raw = std::ptr::null_mut();
        }
    }
}

impl std::fmt::Debug for MatDescr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MatDescr").finish_non_exhaustive()
    }
}

/// Scalar element type usable with the wrapped sparse routines.
pub trait Scalar: Copy + Sized + Sealed {
    /// `y := α · op(A) · x + β · y` where `A` is in CSR format.
    #[allow(clippy::too_many_arguments)]
    fn csrmv(
        op: Trans,
        alpha: Self,
        m: usize,
        n: usize,
        csr_val: &[Self],
        csr_col_ind: &[sys::aoclsparse_int],
        csr_row_ptr: &[sys::aoclsparse_int],
        descr: &MatDescr,
        x: &[Self],
        beta: Self,
        y: &mut [Self],
    ) -> Result<()>;

    /// Sparse `y[indx] := α·x + y[indx]`.
    fn axpyi(alpha: Self, x: &[Self], indx: &[sys::aoclsparse_int], y: &mut [Self]) -> Result<()>;

    /// Sparse gather: `x[i] := y[indx[i]]`.
    fn gthr(y: &[Self], indx: &[sys::aoclsparse_int], x: &mut [Self]) -> Result<()>;

    /// Sparse scatter: `y[indx[i]] := x[i]`.
    fn sctr(x: &[Self], indx: &[sys::aoclsparse_int], y: &mut [Self]) -> Result<()>;

    /// Solve `op(A) · y = α · x` where `A` is sparse triangular (CSR).
    #[allow(clippy::too_many_arguments)]
    fn csrsv(
        op: Trans,
        alpha: Self,
        m: usize,
        csr_val: &[Self],
        csr_col_ind: &[sys::aoclsparse_int],
        csr_row_ptr: &[sys::aoclsparse_int],
        descr: &MatDescr,
        x: &[Self],
        y: &mut [Self],
    ) -> Result<()>;

    /// Convert a CSR matrix to a dense `m × n` matrix.
    #[allow(clippy::too_many_arguments)]
    fn csr_to_dense(
        m: usize,
        n: usize,
        descr: &MatDescr,
        csr_val: &[Self],
        csr_row_ptr: &[sys::aoclsparse_int],
        csr_col_ind: &[sys::aoclsparse_int],
        a: &mut [Self],
        ld: usize,
        order: Order,
    ) -> Result<()>;

    /// Convert CSR → CSC.
    #[allow(clippy::too_many_arguments)]
    fn csr_to_csc(
        m: usize,
        n: usize,
        descr: &MatDescr,
        base_csc: IndexBase,
        csr_row_ptr: &[sys::aoclsparse_int],
        csr_col_ind: &[sys::aoclsparse_int],
        csr_val: &[Self],
        csc_row_ind: &mut [sys::aoclsparse_int],
        csc_col_ptr: &mut [sys::aoclsparse_int],
        csc_val: &mut [Self],
    ) -> Result<()>;
}

macro_rules! impl_scalar {
    (
        $t:ty,
        csrmv = $csrmv:ident,
        axpyi = $axpyi:ident,
        gthr = $gthr:ident,
        sctr = $sctr:ident,
        csrsv = $csrsv:ident,
        csr2dense = $csr2dense:ident,
        csr2csc = $csr2csc:ident
    ) => {
        impl Scalar for $t {
            fn csrmv(
                op: Trans,
                alpha: Self,
                m: usize,
                n: usize,
                csr_val: &[Self],
                csr_col_ind: &[sys::aoclsparse_int],
                csr_row_ptr: &[sys::aoclsparse_int],
                descr: &MatDescr,
                x: &[Self],
                beta: Self,
                y: &mut [Self],
            ) -> Result<()> {
                if csr_row_ptr.len() != m + 1 {
                    return Err(Error::InvalidArgument(format!(
                        "csrmv: csr_row_ptr length {} != m+1 = {}",
                        csr_row_ptr.len(),
                        m + 1
                    )));
                }
                let nnz = csr_val.len();
                if csr_col_ind.len() != nnz {
                    return Err(Error::InvalidArgument(format!(
                        "csrmv: csr_col_ind length {} != csr_val length {}",
                        csr_col_ind.len(),
                        nnz
                    )));
                }
                let (x_len, y_len) = match op {
                    Trans::No => (n, m),
                    Trans::T | Trans::C => (m, n),
                };
                if x.len() < x_len {
                    return Err(Error::InvalidArgument(format!(
                        "csrmv: x length {} < expected {x_len}",
                        x.len()
                    )));
                }
                if y.len() < y_len {
                    return Err(Error::InvalidArgument(format!(
                        "csrmv: y length {} < expected {y_len}",
                        y.len()
                    )));
                }

                let status = unsafe {
                    sys::$csrmv(
                        trans_raw(op),
                        &alpha,
                        m as sys::aoclsparse_int,
                        n as sys::aoclsparse_int,
                        nnz as sys::aoclsparse_int,
                        csr_val.as_ptr(),
                        csr_col_ind.as_ptr(),
                        csr_row_ptr.as_ptr(),
                        descr.as_raw(),
                        x.as_ptr(),
                        &beta,
                        y.as_mut_ptr(),
                    )
                };
                check_status("sparse", status)
            }

            fn axpyi(
                alpha: Self,
                x: &[Self],
                indx: &[sys::aoclsparse_int],
                y: &mut [Self],
            ) -> Result<()> {
                let status = unsafe {
                    sys::$axpyi(
                        x.len() as sys::aoclsparse_int,
                        alpha,
                        x.as_ptr(),
                        indx.as_ptr(),
                        y.as_mut_ptr(),
                    )
                };
                check_status("sparse", status)
            }

            fn gthr(
                y: &[Self],
                indx: &[sys::aoclsparse_int],
                x: &mut [Self],
            ) -> Result<()> {
                let status = unsafe {
                    sys::$gthr(
                        x.len() as sys::aoclsparse_int,
                        y.as_ptr(),
                        x.as_mut_ptr(),
                        indx.as_ptr(),
                    )
                };
                check_status("sparse", status)
            }

            fn sctr(
                x: &[Self],
                indx: &[sys::aoclsparse_int],
                y: &mut [Self],
            ) -> Result<()> {
                let status = unsafe {
                    sys::$sctr(
                        x.len() as sys::aoclsparse_int,
                        x.as_ptr(),
                        indx.as_ptr(),
                        y.as_mut_ptr(),
                    )
                };
                check_status("sparse", status)
            }

            #[allow(clippy::too_many_arguments)]
            fn csrsv(
                op: Trans,
                alpha: Self,
                m: usize,
                csr_val: &[Self],
                csr_col_ind: &[sys::aoclsparse_int],
                csr_row_ptr: &[sys::aoclsparse_int],
                descr: &MatDescr,
                x: &[Self],
                y: &mut [Self],
            ) -> Result<()> {
                if csr_row_ptr.len() != m + 1 {
                    return Err(Error::InvalidArgument(format!(
                        "csrsv: csr_row_ptr length {} != m+1 = {}",
                        csr_row_ptr.len(), m + 1
                    )));
                }
                if x.len() < m || y.len() < m {
                    return Err(Error::InvalidArgument(format!(
                        "csrsv: x.len()={}, y.len()={}, m={m}",
                        x.len(), y.len()
                    )));
                }
                let status = unsafe {
                    sys::$csrsv(
                        trans_raw(op),
                        &alpha,
                        m as sys::aoclsparse_int,
                        csr_val.as_ptr(),
                        csr_col_ind.as_ptr(),
                        csr_row_ptr.as_ptr(),
                        descr.as_raw(),
                        x.as_ptr(),
                        y.as_mut_ptr(),
                    )
                };
                check_status("sparse", status)
            }

            #[allow(clippy::too_many_arguments)]
            fn csr_to_dense(
                m: usize,
                n: usize,
                descr: &MatDescr,
                csr_val: &[Self],
                csr_row_ptr: &[sys::aoclsparse_int],
                csr_col_ind: &[sys::aoclsparse_int],
                a: &mut [Self],
                ld: usize,
                order: Order,
            ) -> Result<()> {
                let status = unsafe {
                    sys::$csr2dense(
                        m as sys::aoclsparse_int,
                        n as sys::aoclsparse_int,
                        descr.as_raw(),
                        csr_val.as_ptr(),
                        csr_row_ptr.as_ptr(),
                        csr_col_ind.as_ptr(),
                        a.as_mut_ptr(),
                        ld as sys::aoclsparse_int,
                        order.raw(),
                    )
                };
                check_status("sparse", status)
            }

            #[allow(clippy::too_many_arguments)]
            fn csr_to_csc(
                m: usize,
                n: usize,
                descr: &MatDescr,
                base_csc: IndexBase,
                csr_row_ptr: &[sys::aoclsparse_int],
                csr_col_ind: &[sys::aoclsparse_int],
                csr_val: &[Self],
                csc_row_ind: &mut [sys::aoclsparse_int],
                csc_col_ptr: &mut [sys::aoclsparse_int],
                csc_val: &mut [Self],
            ) -> Result<()> {
                let nnz = csr_val.len();
                let status = unsafe {
                    sys::$csr2csc(
                        m as sys::aoclsparse_int,
                        n as sys::aoclsparse_int,
                        nnz as sys::aoclsparse_int,
                        descr.as_raw(),
                        base_csc.raw(),
                        csr_row_ptr.as_ptr(),
                        csr_col_ind.as_ptr(),
                        csr_val.as_ptr(),
                        csc_row_ind.as_mut_ptr(),
                        csc_col_ptr.as_mut_ptr(),
                        csc_val.as_mut_ptr(),
                    )
                };
                check_status("sparse", status)
            }
        }
    };
}

impl_scalar!(
    f32,
    csrmv = aoclsparse_scsrmv,
    axpyi = aoclsparse_saxpyi,
    gthr = aoclsparse_sgthr,
    sctr = aoclsparse_ssctr,
    csrsv = aoclsparse_scsrsv,
    csr2dense = aoclsparse_scsr2dense,
    csr2csc = aoclsparse_scsr2csc
);
impl_scalar!(
    f64,
    csrmv = aoclsparse_dcsrmv,
    axpyi = aoclsparse_daxpyi,
    gthr = aoclsparse_dgthr,
    sctr = aoclsparse_dsctr,
    csrsv = aoclsparse_dcsrsv,
    csr2dense = aoclsparse_dcsr2dense,
    csr2csc = aoclsparse_dcsr2csc
);

/// Compute `y := α · A · x + β · y` for a CSR matrix `A`.
#[allow(clippy::too_many_arguments)]
pub fn csrmv<T: Scalar>(
    alpha: T,
    m: usize,
    n: usize,
    csr_val: &[T],
    csr_col_ind: &[sys::aoclsparse_int],
    csr_row_ptr: &[sys::aoclsparse_int],
    descr: &MatDescr,
    x: &[T],
    beta: T,
    y: &mut [T],
) -> Result<()> {
    T::csrmv(
        Trans::No,
        alpha,
        m,
        n,
        csr_val,
        csr_col_ind,
        csr_row_ptr,
        descr,
        x,
        beta,
        y,
    )
}

// =========================================================================
//   Sparse vector operations (axpyi, gather/scatter)
// =========================================================================

/// Sparse `y[indx] := α·x + y[indx]` (sparse vector AXPY).
///
/// `x` and `indx` must have equal length (`nnz`); each `indx[i]` indexes
/// into `y`.
pub fn axpyi<T: Scalar>(alpha: T, x: &[T], indx: &[sys::aoclsparse_int], y: &mut [T]) -> Result<()> {
    if x.len() != indx.len() {
        return Err(Error::InvalidArgument(format!(
            "axpyi: x.len()={}, indx.len()={}", x.len(), indx.len()
        )));
    }
    T::axpyi(alpha, x, indx, y)
}

/// Sparse gather: `x[i] := y[indx[i]]` for `i ∈ [0, nnz)`.
pub fn gthr<T: Scalar>(y: &[T], indx: &[sys::aoclsparse_int], x: &mut [T]) -> Result<()> {
    if x.len() != indx.len() {
        return Err(Error::InvalidArgument(format!(
            "gthr: x.len()={}, indx.len()={}", x.len(), indx.len()
        )));
    }
    T::gthr(y, indx, x)
}

/// Sparse scatter: `y[indx[i]] := x[i]` for `i ∈ [0, nnz)`.
pub fn sctr<T: Scalar>(x: &[T], indx: &[sys::aoclsparse_int], y: &mut [T]) -> Result<()> {
    if x.len() != indx.len() {
        return Err(Error::InvalidArgument(format!(
            "sctr: x.len()={}, indx.len()={}", x.len(), indx.len()
        )));
    }
    T::sctr(x, indx, y)
}

// =========================================================================
//   Sparse triangular solve (csrsv)
// =========================================================================

/// Solve `op(A) · y = α · x` where `A` is sparse triangular in CSR
/// format. The triangle is determined by the `descr`'s fill mode (set
/// via the `aoclsparse_set_mat_*` C-API; defaults to upper, non-unit).
#[allow(clippy::too_many_arguments)]
pub fn csrsv<T: Scalar>(
    op: Trans,
    alpha: T,
    m: usize,
    csr_val: &[T],
    csr_col_ind: &[sys::aoclsparse_int],
    csr_row_ptr: &[sys::aoclsparse_int],
    descr: &MatDescr,
    x: &[T],
    y: &mut [T],
) -> Result<()> {
    T::csrsv(op, alpha, m, csr_val, csr_col_ind, csr_row_ptr, descr, x, y)
}

// =========================================================================
//   Format conversion: csr → dense, csr → csc
// =========================================================================

/// Storage order used when converting from CSR to a dense matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Order {
    RowMajor,
    ColMajor,
}

impl Order {
    fn raw(self) -> sys::aoclsparse_order {
        match self {
            Order::RowMajor => sys::aoclsparse_order__aoclsparse_order_row,
            Order::ColMajor => sys::aoclsparse_order__aoclsparse_order_column,
        }
    }
}

/// Convert a CSR sparse matrix to a dense `m × n` matrix in `out`.
#[allow(clippy::too_many_arguments)]
pub fn csr_to_dense<T: Scalar>(
    m: usize,
    n: usize,
    descr: &MatDescr,
    csr_val: &[T],
    csr_row_ptr: &[sys::aoclsparse_int],
    csr_col_ind: &[sys::aoclsparse_int],
    a: &mut [T],
    ld: usize,
    order: Order,
) -> Result<()> {
    if csr_row_ptr.len() != m + 1 {
        return Err(Error::InvalidArgument(format!(
            "csr_to_dense: csr_row_ptr length {} != m+1 = {}",
            csr_row_ptr.len(), m + 1
        )));
    }
    let needed = match order {
        Order::RowMajor => m.saturating_sub(1) * ld + n,
        Order::ColMajor => n.saturating_sub(1) * ld + m,
    };
    if a.len() < needed {
        return Err(Error::InvalidArgument(format!(
            "csr_to_dense: A length {} < needed {needed}", a.len()
        )));
    }
    T::csr_to_dense(m, n, descr, csr_val, csr_row_ptr, csr_col_ind, a, ld, order)
}

/// Convert a CSR matrix to CSC. Output arrays must be pre-sized.
#[allow(clippy::too_many_arguments)]
pub fn csr_to_csc<T: Scalar>(
    m: usize,
    n: usize,
    descr: &MatDescr,
    base_csc: IndexBase,
    csr_row_ptr: &[sys::aoclsparse_int],
    csr_col_ind: &[sys::aoclsparse_int],
    csr_val: &[T],
    csc_row_ind: &mut [sys::aoclsparse_int],
    csc_col_ptr: &mut [sys::aoclsparse_int],
    csc_val: &mut [T],
) -> Result<()> {
    let nnz = csr_val.len();
    if csr_col_ind.len() != nnz || csc_row_ind.len() < nnz || csc_val.len() < nnz {
        return Err(Error::InvalidArgument(format!(
            "csr_to_csc: nnz mismatch (csr_val={}, csr_col_ind={}, csc_row_ind={}, csc_val={})",
            nnz, csr_col_ind.len(), csc_row_ind.len(), csc_val.len()
        )));
    }
    if csr_row_ptr.len() != m + 1 || csc_col_ptr.len() != n + 1 {
        return Err(Error::InvalidArgument(format!(
            "csr_to_csc: row_ptr length {} != m+1 = {} or col_ptr length {} != n+1 = {}",
            csr_row_ptr.len(), m + 1, csc_col_ptr.len(), n + 1
        )));
    }
    T::csr_to_csc(m, n, descr, base_csc, csr_row_ptr, csr_col_ind, csr_val,
                  csc_row_ind, csc_col_ptr, csc_val)
}

/// Index base for sparse-format index arrays.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IndexBase {
    /// 0-based indexing (C convention).
    Zero,
    /// 1-based indexing (Fortran convention).
    One,
}

impl IndexBase {
    fn raw(self) -> sys::aoclsparse_index_base {
        match self {
            IndexBase::Zero => sys::aoclsparse_index_base__aoclsparse_index_base_zero,
            IndexBase::One => sys::aoclsparse_index_base__aoclsparse_index_base_one,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csrmv_2x2_identity_f64() {
        let val = [1.0_f64, 1.0];
        let col: [sys::aoclsparse_int; 2] = [0, 1];
        let rowptr: [sys::aoclsparse_int; 3] = [0, 1, 2];
        let x = [3.0_f64, 4.0];
        let mut y = [0.0_f64; 2];
        let descr = MatDescr::new().unwrap();
        csrmv(1.0_f64, 2, 2, &val, &col, &rowptr, &descr, &x, 0.0, &mut y).unwrap();
        assert!((y[0] - 3.0).abs() < 1e-12);
        assert!((y[1] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn csrmv_simple_2x3() {
        let val = [1.0_f64, 2.0, 3.0];
        let col: [sys::aoclsparse_int; 3] = [0, 1, 2];
        let rowptr: [sys::aoclsparse_int; 3] = [0, 2, 3];
        let x = [1.0_f64; 3];
        let mut y = [0.0_f64; 2];
        let descr = MatDescr::new().unwrap();
        csrmv(1.0_f64, 2, 3, &val, &col, &rowptr, &descr, &x, 0.0, &mut y).unwrap();
        assert!((y[0] - 3.0).abs() < 1e-12, "got {}", y[0]);
        assert!((y[1] - 3.0).abs() < 1e-12, "got {}", y[1]);
    }

    #[test]
    fn dim_mismatch_is_error() {
        let val = [1.0_f64];
        let col: [sys::aoclsparse_int; 1] = [0];
        let rowptr: [sys::aoclsparse_int; 2] = [0, 1];
        let x = [1.0_f64];
        let mut y = [0.0_f64; 2];
        let descr = MatDescr::new().unwrap();
        let err = csrmv(1.0_f64, 2, 1, &val, &col, &rowptr, &descr, &x, 0.0, &mut y).unwrap_err();
        matches!(err, Error::InvalidArgument(_));
    }

    #[test]
    fn axpyi_scatter() {
        // y = [10, 20, 30, 40], x = [1, 2], indx = [0, 2], α = 3
        // → y[0] += 3·1 = 13; y[2] += 3·2 = 36
        let mut y = [10.0_f64, 20.0, 30.0, 40.0];
        let x = [1.0_f64, 2.0];
        let indx: [sys::aoclsparse_int; 2] = [0, 2];
        axpyi(3.0_f64, &x, &indx, &mut y).unwrap();
        assert_eq!(y, [13.0, 20.0, 36.0, 40.0]);
    }

    #[test]
    fn gthr_scatter_round_trip() {
        // Gather from y at indx → x; scatter x at indx → into a fresh y2.
        let y = [10.0_f64, 20.0, 30.0, 40.0];
        let indx: [sys::aoclsparse_int; 2] = [1, 3];
        let mut x = [0.0_f64; 2];
        gthr(&y, &indx, &mut x).unwrap();
        assert_eq!(x, [20.0, 40.0]);

        let mut y2 = [0.0_f64; 4];
        sctr(&x, &indx, &mut y2).unwrap();
        assert_eq!(y2, [0.0, 20.0, 0.0, 40.0]);
    }

    #[test]
    fn csr_to_dense_round_trip() {
        // 2×3 CSR: [[1,0,2],[0,3,0]] → val=[1,2,3], col=[0,2,1], rp=[0,2,3]
        let val = [1.0_f64, 2.0, 3.0];
        let col: [sys::aoclsparse_int; 3] = [0, 2, 1];
        let rp: [sys::aoclsparse_int; 3] = [0, 2, 3];
        let descr = MatDescr::new().unwrap();
        let mut dense = [0.0_f64; 6];
        csr_to_dense::<f64>(2, 3, &descr, &val, &rp, &col, &mut dense, 3, Order::RowMajor).unwrap();
        assert_eq!(dense, [1.0, 0.0, 2.0, 0.0, 3.0, 0.0]);
    }
}
