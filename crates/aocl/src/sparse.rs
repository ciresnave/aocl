//! Safe wrappers for AOCL-Sparse.
//!
//! Currently exposes a thin RAII handle for matrix descriptors plus
//! `csrmv`, the workhorse sparse matrix–dense vector product `y := α·op(A)·x + β·y`
//! for matrices stored in CSR format. Real-precision (`f32` / `f64`) only;
//! complex precisions and additional formats (CSC, COO, ELL, …) will follow.
//!
//! For routines not yet wrapped, drop down to [`aocl_sys::sparse`].

use crate::error::{Error, Result};
use aocl_sys::sparse as sys;

/// Whether the matrix is applied as-is, transposed, or conjugate-transposed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Operation {
    None,
    Transpose,
    ConjugateTranspose,
}

impl Operation {
    fn raw(self) -> sys::aoclsparse_operation {
        match self {
            Operation::None => sys::aoclsparse_operation__aoclsparse_operation_none,
            Operation::Transpose => sys::aoclsparse_operation__aoclsparse_operation_transpose,
            Operation::ConjugateTranspose => {
                sys::aoclsparse_operation__aoclsparse_operation_conjugate_transpose
            }
        }
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
///
/// Holds the descriptor that AOCL-Sparse uses to record matrix properties
/// (general / symmetric / triangular, fill mode, diagonal, index base).
/// Created with default settings (general, zero-based indexing).
pub struct MatDescr {
    raw: sys::aoclsparse_mat_descr,
}

impl MatDescr {
    /// Create a fresh descriptor with library defaults.
    pub fn new() -> Result<Self> {
        let mut raw: sys::aoclsparse_mat_descr = std::ptr::null_mut();
        // SAFETY: We pass a writable out-pointer; AOCL fills it with an owned handle.
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
    /// The returned pointer is valid only for the lifetime of `self`. Do
    /// not call `aoclsparse_destroy_mat_descr` on it; that is `Drop`'s job.
    pub fn as_raw(&self) -> sys::aoclsparse_mat_descr {
        self.raw
    }
}

impl Drop for MatDescr {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            // SAFETY: We own this handle; nothing else aliases it.
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
pub trait Scalar: Copy + Sized + private::Sealed {
    /// `y := α · op(A) · x + β · y` where `A` is in CSR format.
    #[allow(clippy::too_many_arguments)]
    fn csrmv(
        op: Operation,
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
}

mod private {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

macro_rules! impl_scalar {
    ($t:ty, $csrmv:ident) => {
        impl Scalar for $t {
            fn csrmv(
                op: Operation,
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
                    Operation::None => (n, m),
                    Operation::Transpose | Operation::ConjugateTranspose => (m, n),
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

                // SAFETY: All pointers are derived from valid Rust slices
                // we have just length-checked. AOCL reads csr_val,
                // csr_col_ind, csr_row_ptr, x; writes y.
                let status = unsafe {
                    sys::$csrmv(
                        op.raw(),
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
        }
    };
}

impl_scalar!(f32, aoclsparse_scsrmv);
impl_scalar!(f64, aoclsparse_dcsrmv);

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
        Operation::None,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csrmv_2x2_identity_f64() {
        // A = I_2 in CSR (zero-based):
        //   val=[1, 1], col=[0, 1], rowptr=[0, 1, 2]
        // x = [3, 4]; y = α·A·x + β·y = 1·[3, 4] + 0·[0, 0] = [3, 4]
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
        // A =
        //   [[1, 2, 0],
        //    [0, 0, 3]]
        // CSR: val=[1, 2, 3], col=[0, 1, 2], rowptr=[0, 2, 3]
        // x = [1, 1, 1]; A·x = [3, 3]
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
        let rowptr: [sys::aoclsparse_int; 2] = [0, 1]; // m+1 = 2 → m=1; we pass m=2 to force mismatch
        let x = [1.0_f64];
        let mut y = [0.0_f64; 2];
        let descr = MatDescr::new().unwrap();
        let err = csrmv(1.0_f64, 2, 1, &val, &col, &rowptr, &descr, &x, 0.0, &mut y).unwrap_err();
        matches!(err, Error::InvalidArgument(_));
    }
}
