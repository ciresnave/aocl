//! Safe wrappers for AOCL-Sparse.

#![warn(missing_debug_implementations)]
#![cfg_attr(docsrs, feature(doc_cfg))]

use std::ffi::CString;
use std::marker::PhantomData;

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

    /// `y := α · op(A) · x + β · y` where `A` is in ELLPACK format.
    /// Library only supports `op = aoclsparse_operation_none`.
    #[allow(clippy::too_many_arguments)]
    fn ellmv(
        op: Trans,
        alpha: Self,
        m: usize,
        n: usize,
        ell_val: &[Self],
        ell_col_ind: &[sys::aoclsparse_int],
        ell_width: usize,
        descr: &MatDescr,
        x: &[Self],
        beta: Self,
        y: &mut [Self],
    ) -> Result<()>;

    /// `y := α · op(A) · x + β · y` where `A` is in BSR format.
    /// `mb`/`nb` are block-row / block-column counts and `bsr_dim` is the
    /// block edge length. Library only supports `op = aoclsparse_operation_none`.
    #[allow(clippy::too_many_arguments)]
    fn bsrmv(
        op: Trans,
        alpha: Self,
        mb: usize,
        nb: usize,
        bsr_dim: usize,
        bsr_val: &[Self],
        bsr_col_ind: &[sys::aoclsparse_int],
        bsr_row_ptr: &[sys::aoclsparse_int],
        descr: &MatDescr,
        x: &[Self],
        beta: Self,
        y: &mut [Self],
    ) -> Result<()>;

    /// Wrap raw CSR pointers in a fresh `aoclsparse_matrix` handle. The
    /// library does not copy the underlying arrays — keep them alive for as
    /// long as the returned handle is in use.
    #[allow(clippy::too_many_arguments)]
    fn create_csr(
        base: IndexBase,
        m: usize,
        n: usize,
        nnz: usize,
        row_ptr: *mut sys::aoclsparse_int,
        col_idx: *mut sys::aoclsparse_int,
        val: *mut Self,
    ) -> Result<sys::aoclsparse_matrix>;

    /// Read out a library-owned CSR matrix's metadata and array pointers.
    /// Pointers reference internal storage; do not modify or free them.
    fn export_csr(
        mat: sys::aoclsparse_matrix,
    ) -> Result<(IndexBase, usize, usize, usize, *mut sys::aoclsparse_int, *mut sys::aoclsparse_int, *mut Self)>;

    /// ILU(0) smoother: applies one ILU step in place to `x`.
    fn ilu_smoother(
        op: Trans,
        a: sys::aoclsparse_matrix,
        descr: &MatDescr,
        x: &mut [Self],
        b: &[Self],
    ) -> Result<()>;

    /// Initialise an iterative-solver handle for this scalar type.
    fn itsol_init(handle: &mut sys::aoclsparse_itsol_handle) -> Result<()>;

    /// Run the iterative solver's forward (direct) interface.
    #[allow(clippy::too_many_arguments)]
    fn itsol_solve(
        handle: sys::aoclsparse_itsol_handle,
        n: usize,
        mat: sys::aoclsparse_matrix,
        descr: &MatDescr,
        b: &[Self],
        x: &mut [Self],
        rinfo: &mut [Self; 100],
    ) -> Result<()>;

    /// Sparse-sparse matrix product producing a new CSR matrix. The output
    /// pointer is written through `*out`.
    ///
    /// # Safety
    /// Caller is responsible for the validity of all `aoclsparse_*` handle
    /// arguments and for adopting `*out` (e.g. via
    /// [`SparseMatrix::from_library_owned`]) on success.
    #[allow(clippy::too_many_arguments)]
    unsafe fn csr2m_ffi(
        op_a: sys::aoclsparse_operation,
        descr_a: sys::aoclsparse_mat_descr,
        a: sys::aoclsparse_matrix,
        op_b: sys::aoclsparse_operation,
        descr_b: sys::aoclsparse_mat_descr,
        b: sys::aoclsparse_matrix,
        request: sys::aoclsparse_request,
        out: *mut sys::aoclsparse_matrix,
    ) -> sys::aoclsparse_status;
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
        csr2csc = $csr2csc:ident,
        ellmv = $ellmv:ident,
        bsrmv = $bsrmv:ident,
        create_csr = $create_csr:ident,
        export_csr = $export_csr:ident,
        ilu_smoother = $ilu_smoother:ident,
        itsol_init = $itsol_init:ident,
        itsol_solve = $itsol_solve:ident,
        csr2m = $csr2m:ident
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

            #[allow(clippy::too_many_arguments)]
            fn ellmv(
                op: Trans,
                alpha: Self,
                m: usize,
                n: usize,
                ell_val: &[Self],
                ell_col_ind: &[sys::aoclsparse_int],
                ell_width: usize,
                descr: &MatDescr,
                x: &[Self],
                beta: Self,
                y: &mut [Self],
            ) -> Result<()> {
                let nnz = ell_val.len();
                if ell_col_ind.len() != nnz {
                    return Err(Error::InvalidArgument(format!(
                        "ellmv: ell_col_ind length {} != ell_val length {nnz}",
                        ell_col_ind.len()
                    )));
                }
                let needed = m
                    .checked_mul(ell_width)
                    .ok_or_else(|| Error::InvalidArgument(
                        "ellmv: m * ell_width overflows".into()))?;
                if nnz < needed {
                    return Err(Error::InvalidArgument(format!(
                        "ellmv: ell_val length {nnz} < m*ell_width = {needed}"
                    )));
                }
                let (x_len, y_len) = match op {
                    Trans::No => (n, m),
                    Trans::T | Trans::C => (m, n),
                };
                if x.len() < x_len || y.len() < y_len {
                    return Err(Error::InvalidArgument(format!(
                        "ellmv: x.len()={}, y.len()={}, expected ({x_len}, {y_len})",
                        x.len(), y.len()
                    )));
                }
                let status = unsafe {
                    sys::$ellmv(
                        trans_raw(op),
                        &alpha,
                        m as sys::aoclsparse_int,
                        n as sys::aoclsparse_int,
                        nnz as sys::aoclsparse_int,
                        ell_val.as_ptr(),
                        ell_col_ind.as_ptr(),
                        ell_width as sys::aoclsparse_int,
                        descr.as_raw(),
                        x.as_ptr(),
                        &beta,
                        y.as_mut_ptr(),
                    )
                };
                check_status("sparse", status)
            }

            #[allow(clippy::too_many_arguments)]
            fn bsrmv(
                op: Trans,
                alpha: Self,
                mb: usize,
                nb: usize,
                bsr_dim: usize,
                bsr_val: &[Self],
                bsr_col_ind: &[sys::aoclsparse_int],
                bsr_row_ptr: &[sys::aoclsparse_int],
                descr: &MatDescr,
                x: &[Self],
                beta: Self,
                y: &mut [Self],
            ) -> Result<()> {
                if bsr_row_ptr.len() != mb + 1 {
                    return Err(Error::InvalidArgument(format!(
                        "bsrmv: bsr_row_ptr length {} != mb+1 = {}",
                        bsr_row_ptr.len(), mb + 1
                    )));
                }
                let block_area = bsr_dim
                    .checked_mul(bsr_dim)
                    .ok_or_else(|| Error::InvalidArgument(
                        "bsrmv: bsr_dim*bsr_dim overflows".into()))?;
                let nnzb = bsr_col_ind.len();
                if bsr_val.len() < nnzb * block_area {
                    return Err(Error::InvalidArgument(format!(
                        "bsrmv: bsr_val length {} < nnzb*bsr_dim^2 = {}",
                        bsr_val.len(), nnzb * block_area
                    )));
                }
                let (x_len, y_len) = match op {
                    Trans::No => (nb * bsr_dim, mb * bsr_dim),
                    Trans::T | Trans::C => (mb * bsr_dim, nb * bsr_dim),
                };
                if x.len() < x_len || y.len() < y_len {
                    return Err(Error::InvalidArgument(format!(
                        "bsrmv: x.len()={}, y.len()={}, expected ({x_len}, {y_len})",
                        x.len(), y.len()
                    )));
                }
                let status = unsafe {
                    sys::$bsrmv(
                        trans_raw(op),
                        &alpha,
                        mb as sys::aoclsparse_int,
                        nb as sys::aoclsparse_int,
                        bsr_dim as sys::aoclsparse_int,
                        bsr_val.as_ptr(),
                        bsr_col_ind.as_ptr(),
                        bsr_row_ptr.as_ptr(),
                        descr.as_raw(),
                        x.as_ptr(),
                        &beta,
                        y.as_mut_ptr(),
                    )
                };
                check_status("sparse", status)
            }

            fn create_csr(
                base: IndexBase,
                m: usize,
                n: usize,
                nnz: usize,
                row_ptr: *mut sys::aoclsparse_int,
                col_idx: *mut sys::aoclsparse_int,
                val: *mut Self,
            ) -> Result<sys::aoclsparse_matrix> {
                let mut raw: sys::aoclsparse_matrix = std::ptr::null_mut();
                let status = unsafe {
                    sys::$create_csr(
                        &mut raw,
                        base.raw(),
                        m as sys::aoclsparse_int,
                        n as sys::aoclsparse_int,
                        nnz as sys::aoclsparse_int,
                        row_ptr,
                        col_idx,
                        val,
                    )
                };
                check_status("sparse", status)?;
                if raw.is_null() {
                    return Err(Error::AllocationFailed("sparse"));
                }
                Ok(raw)
            }

            fn export_csr(
                mat: sys::aoclsparse_matrix,
            ) -> Result<(IndexBase, usize, usize, usize,
                         *mut sys::aoclsparse_int, *mut sys::aoclsparse_int, *mut Self)>
            {
                let mut base: sys::aoclsparse_index_base = 0;
                let mut m: sys::aoclsparse_int = 0;
                let mut n: sys::aoclsparse_int = 0;
                let mut nnz: sys::aoclsparse_int = 0;
                let mut row_ptr: *mut sys::aoclsparse_int = std::ptr::null_mut();
                let mut col_ind: *mut sys::aoclsparse_int = std::ptr::null_mut();
                let mut val: *mut Self = std::ptr::null_mut();
                let status = unsafe {
                    sys::$export_csr(
                        mat,
                        &mut base,
                        &mut m,
                        &mut n,
                        &mut nnz,
                        &mut row_ptr,
                        &mut col_ind,
                        &mut val,
                    )
                };
                check_status("sparse", status)?;
                let base_e = if base == sys::aoclsparse_index_base__aoclsparse_index_base_one {
                    IndexBase::One
                } else {
                    IndexBase::Zero
                };
                Ok((base_e, m as usize, n as usize, nnz as usize, row_ptr, col_ind, val))
            }

            fn ilu_smoother(
                op: Trans,
                a: sys::aoclsparse_matrix,
                descr: &MatDescr,
                x: &mut [Self],
                b: &[Self],
            ) -> Result<()> {
                let mut precond_csr_val: *mut Self = std::ptr::null_mut();
                let status = unsafe {
                    sys::$ilu_smoother(
                        trans_raw(op),
                        a,
                        descr.as_raw(),
                        &mut precond_csr_val,
                        std::ptr::null(),
                        x.as_mut_ptr(),
                        b.as_ptr(),
                    )
                };
                check_status("sparse", status)
            }

            fn itsol_init(handle: &mut sys::aoclsparse_itsol_handle) -> Result<()> {
                let status = unsafe { sys::$itsol_init(handle) };
                check_status("sparse", status)
            }

            fn itsol_solve(
                handle: sys::aoclsparse_itsol_handle,
                n: usize,
                mat: sys::aoclsparse_matrix,
                descr: &MatDescr,
                b: &[Self],
                x: &mut [Self],
                rinfo: &mut [Self; 100],
            ) -> Result<()> {
                if b.len() < n || x.len() < n {
                    return Err(Error::InvalidArgument(format!(
                        "itsol_solve: b.len()={}, x.len()={}, n={n}", b.len(), x.len()
                    )));
                }
                let status = unsafe {
                    sys::$itsol_solve(
                        handle,
                        n as sys::aoclsparse_int,
                        mat,
                        descr.as_raw(),
                        b.as_ptr(),
                        x.as_mut_ptr(),
                        rinfo.as_mut_ptr(),
                        None,
                        None,
                        std::ptr::null_mut(),
                    )
                };
                check_status("sparse", status)
            }

            unsafe fn csr2m_ffi(
                op_a: sys::aoclsparse_operation,
                descr_a: sys::aoclsparse_mat_descr,
                a: sys::aoclsparse_matrix,
                op_b: sys::aoclsparse_operation,
                descr_b: sys::aoclsparse_mat_descr,
                b: sys::aoclsparse_matrix,
                request: sys::aoclsparse_request,
                out: *mut sys::aoclsparse_matrix,
            ) -> sys::aoclsparse_status {
                sys::$csr2m(op_a, descr_a, a, op_b, descr_b, b, request, out)
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
    csr2csc = aoclsparse_scsr2csc,
    ellmv = aoclsparse_sellmv,
    bsrmv = aoclsparse_sbsrmv,
    create_csr = aoclsparse_create_scsr,
    export_csr = aoclsparse_export_scsr,
    ilu_smoother = aoclsparse_silu_smoother,
    itsol_init = aoclsparse_itsol_s_init,
    itsol_solve = aoclsparse_itsol_s_solve,
    csr2m = aoclsparse_scsr2m
);
impl_scalar!(
    f64,
    csrmv = aoclsparse_dcsrmv,
    axpyi = aoclsparse_daxpyi,
    gthr = aoclsparse_dgthr,
    sctr = aoclsparse_dsctr,
    csrsv = aoclsparse_dcsrsv,
    csr2dense = aoclsparse_dcsr2dense,
    csr2csc = aoclsparse_dcsr2csc,
    ellmv = aoclsparse_dellmv,
    bsrmv = aoclsparse_dbsrmv,
    create_csr = aoclsparse_create_dcsr,
    export_csr = aoclsparse_export_dcsr,
    ilu_smoother = aoclsparse_dilu_smoother,
    itsol_init = aoclsparse_itsol_d_init,
    itsol_solve = aoclsparse_itsol_d_solve,
    csr2m = aoclsparse_dcsr2m
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

// =========================================================================
//   ELLPACK and BSR mat-vec
// =========================================================================

/// Compute `y := α · op(A) · x + β · y` for an ELLPACK-format `A`.
///
/// `ell_val` and `ell_col_ind` each have length `m * ell_width`, where rows
/// shorter than `ell_width` are padded.
///
/// AOCL only implements `op = Trans::No` for ELLPACK at present.
#[allow(clippy::too_many_arguments)]
pub fn ellmv<T: Scalar>(
    op: Trans,
    alpha: T,
    m: usize,
    n: usize,
    ell_val: &[T],
    ell_col_ind: &[sys::aoclsparse_int],
    ell_width: usize,
    descr: &MatDescr,
    x: &[T],
    beta: T,
    y: &mut [T],
) -> Result<()> {
    T::ellmv(op, alpha, m, n, ell_val, ell_col_ind, ell_width, descr, x, beta, y)
}

/// Compute `y := α · op(A) · x + β · y` for a BSR-format `A`.
///
/// `mb` and `nb` count blocks (so `A` is `mb·bsr_dim × nb·bsr_dim`).
/// `bsr_val` is laid out as `nnzb` consecutive `bsr_dim × bsr_dim` blocks.
///
/// AOCL only implements `op = Trans::No` for BSR at present.
#[allow(clippy::too_many_arguments)]
pub fn bsrmv<T: Scalar>(
    op: Trans,
    alpha: T,
    mb: usize,
    nb: usize,
    bsr_dim: usize,
    bsr_val: &[T],
    bsr_col_ind: &[sys::aoclsparse_int],
    bsr_row_ptr: &[sys::aoclsparse_int],
    descr: &MatDescr,
    x: &[T],
    beta: T,
    y: &mut [T],
) -> Result<()> {
    T::bsrmv(op, alpha, mb, nb, bsr_dim, bsr_val, bsr_col_ind, bsr_row_ptr,
             descr, x, beta, y)
}

// =========================================================================
//   High-level matrix handle (aoclsparse_matrix)
// =========================================================================

/// Stage of a multi-pass sparse-sparse matrix product (see [`csr2m`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Stage {
    /// Analyse only; reserve nnz count without writing values.
    NnzCount,
    /// Compute values, given a prior `NnzCount` call.
    Finalize,
    /// Single-shot full computation.
    FullComputation,
}

impl Stage {
    fn raw(self) -> sys::aoclsparse_request {
        match self {
            Stage::NnzCount => sys::aoclsparse_request__aoclsparse_stage_nnz_count,
            Stage::Finalize => sys::aoclsparse_request__aoclsparse_stage_finalize,
            Stage::FullComputation => sys::aoclsparse_request__aoclsparse_stage_full_computation,
        }
    }
}

enum CsrStorage<T: Scalar> {
    /// Arrays are owned by the Rust side (we keep them alive until drop).
    Owned {
        _row_ptr: Vec<sys::aoclsparse_int>,
        _col_ind: Vec<sys::aoclsparse_int>,
        _val: Vec<T>,
    },
    /// Arrays are owned by the library; only the handle needs destroying.
    LibraryOwned,
}

/// RAII wrapper around an `aoclsparse_matrix` handle holding a CSR matrix.
///
/// Construct from raw CSR vectors with [`SparseMatrix::from_csr`] (the
/// values are copied into the wrapper) or as the result of an operation
/// like [`csr2m`].
pub struct SparseMatrix<T: Scalar> {
    raw: sys::aoclsparse_matrix,
    #[allow(dead_code)] // kept alive so the library can keep its pointers
    storage: CsrStorage<T>,
    base: IndexBase,
    m: usize,
    n: usize,
    nnz: usize,
}

impl<T: Scalar> SparseMatrix<T> {
    /// Build a new matrix handle from CSR arrays. The arrays are copied
    /// into the wrapper; the caller's slices are not retained.
    pub fn from_csr(
        base: IndexBase,
        m: usize,
        n: usize,
        row_ptr: &[sys::aoclsparse_int],
        col_ind: &[sys::aoclsparse_int],
        val: &[T],
    ) -> Result<Self> {
        if row_ptr.len() != m + 1 {
            return Err(Error::InvalidArgument(format!(
                "from_csr: row_ptr length {} != m+1 = {}",
                row_ptr.len(), m + 1
            )));
        }
        let nnz = val.len();
        if col_ind.len() != nnz {
            return Err(Error::InvalidArgument(format!(
                "from_csr: col_ind length {} != val length {nnz}",
                col_ind.len()
            )));
        }
        let mut row_ptr = row_ptr.to_vec();
        let mut col_ind = col_ind.to_vec();
        let mut val = val.to_vec();
        let raw = T::create_csr(
            base,
            m,
            n,
            nnz,
            row_ptr.as_mut_ptr(),
            col_ind.as_mut_ptr(),
            val.as_mut_ptr(),
        )?;
        Ok(Self {
            raw,
            storage: CsrStorage::Owned { _row_ptr: row_ptr, _col_ind: col_ind, _val: val },
            base,
            m,
            n,
            nnz,
        })
    }

    /// Adopt a handle returned by an AOCL routine that allocates its own
    /// CSR storage (e.g. `aoclsparse_dcsr2m`). The library will free the
    /// arrays when the handle is destroyed.
    ///
    /// # Safety
    /// `raw` must be a valid `aoclsparse_matrix` whose internal storage is
    /// owned by the AOCL library and whose precision matches `T`.
    pub unsafe fn from_library_owned(raw: sys::aoclsparse_matrix) -> Result<Self> {
        if raw.is_null() {
            return Err(Error::AllocationFailed("sparse"));
        }
        let (base, m, n, nnz, _, _, _) = T::export_csr(raw)?;
        Ok(Self {
            raw,
            storage: CsrStorage::LibraryOwned,
            base,
            m,
            n,
            nnz,
        })
    }

    /// `(m, n)` dimensions of the matrix.
    pub fn dims(&self) -> (usize, usize) {
        (self.m, self.n)
    }

    /// Number of explicitly stored non-zeros.
    pub fn nnz(&self) -> usize {
        self.nnz
    }

    /// Index base used by this matrix's row-pointer and column-index arrays.
    pub fn base(&self) -> IndexBase {
        self.base
    }

    /// Borrow the raw underlying handle for raw FFI calls.
    ///
    /// # Safety
    /// The returned pointer is valid only for the lifetime of `self`.
    /// Do not call `aoclsparse_destroy` on it.
    pub fn as_raw(&self) -> sys::aoclsparse_matrix {
        self.raw
    }

    /// Read out the CSR contents as freshly allocated `Vec`s.
    pub fn export_csr(&self) -> Result<(IndexBase, Vec<sys::aoclsparse_int>, Vec<sys::aoclsparse_int>, Vec<T>)> {
        let (base, m, _, nnz, row_ptr, col_ind, val) = T::export_csr(self.raw)?;
        let row_ptr = unsafe { std::slice::from_raw_parts(row_ptr, m + 1).to_vec() };
        let col_ind = unsafe { std::slice::from_raw_parts(col_ind, nnz).to_vec() };
        let val = unsafe { std::slice::from_raw_parts(val, nnz).to_vec() };
        Ok((base, row_ptr, col_ind, val))
    }
}

impl<T: Scalar> Drop for SparseMatrix<T> {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                let _ = sys::aoclsparse_destroy(&mut self.raw);
            }
            self.raw = std::ptr::null_mut();
        }
    }
}

impl<T: Scalar> std::fmt::Debug for SparseMatrix<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SparseMatrix")
            .field("m", &self.m)
            .field("n", &self.n)
            .field("nnz", &self.nnz)
            .field("base", &self.base)
            .finish()
    }
}

// =========================================================================
//   Sparse-sparse matrix product (csr2m)
// =========================================================================

/// Compute `C := op_A(A) · op_B(B)` between two CSR matrices, returning a
/// freshly allocated CSR result. Both inputs must have matching scalar
/// precision; complex precisions are not yet exposed by these wrappers.
///
/// Use `Stage::FullComputation` for a one-shot call. The two-stage form
/// (`NnzCount` followed by `Finalize`) is for repeated multiplies sharing
/// the same sparsity pattern.
#[allow(clippy::too_many_arguments)]
pub fn csr2m<T: Scalar>(
    op_a: Trans,
    descr_a: &MatDescr,
    a: &SparseMatrix<T>,
    op_b: Trans,
    descr_b: &MatDescr,
    b: &SparseMatrix<T>,
    stage: Stage,
) -> Result<SparseMatrix<T>> {
    let mut c_raw: sys::aoclsparse_matrix = std::ptr::null_mut();
    let status = unsafe {
        // We need a per-precision dispatch. Encode by calling the matching
        // FFI symbol via a small helper trait.
        T::csr2m_ffi(
            trans_raw(op_a),
            descr_a.as_raw(),
            a.raw,
            trans_raw(op_b),
            descr_b.as_raw(),
            b.raw,
            stage.raw(),
            &mut c_raw,
        )
    };
    check_status("sparse", status)?;
    unsafe { SparseMatrix::from_library_owned(c_raw) }
}

// =========================================================================
//   ILU smoother
// =========================================================================

/// Apply one ILU(0) smoothing step in-place to `x`, with right-hand side `b`.
///
/// On the first call against a matrix this also factorizes; subsequent
/// calls re-use the cached factors stored on the matrix handle.
pub fn ilu_smoother<T: Scalar>(
    op: Trans,
    a: &SparseMatrix<T>,
    descr: &MatDescr,
    x: &mut [T],
    b: &[T],
) -> Result<()> {
    if x.len() < a.n || b.len() < a.m {
        return Err(Error::InvalidArgument(format!(
            "ilu_smoother: x.len()={}, b.len()={}, dims=({}, {})",
            x.len(), b.len(), a.m, a.n
        )));
    }
    T::ilu_smoother(op, a.raw, descr, x, b)
}

// =========================================================================
//   Iterative solver (CG / GMRES) — direct interface
// =========================================================================

/// RAII handle for the AOCL-Sparse iterative-solver suite.
///
/// Configure the solver type and tolerances with [`IterSolver::set_option`]
/// (e.g. `set_option("iterative method", "cg")` or `"gmres"`), then call
/// [`IterSolver::solve`] with the system matrix and right-hand side.
pub struct IterSolver<T: Scalar> {
    handle: sys::aoclsparse_itsol_handle,
    _marker: PhantomData<T>,
}

impl<T: Scalar> IterSolver<T> {
    /// Initialise a new iterative-solver handle for this scalar type.
    pub fn new() -> Result<Self> {
        let mut handle: sys::aoclsparse_itsol_handle = std::ptr::null_mut();
        T::itsol_init(&mut handle)?;
        if handle.is_null() {
            return Err(Error::AllocationFailed("sparse"));
        }
        Ok(Self { handle, _marker: PhantomData })
    }

    /// Set a string-keyed solver option. See AOCL-Sparse's
    /// `aoclsparse_itsol_option_set` for the full option list. Common
    /// keys: `"iterative method"` (`"cg"`/`"gmres"`/`"pcg"`), `"cg
    /// iteration limit"`, `"cg rel tolerance"`, `"gmres preconditioner"`.
    pub fn set_option(&mut self, name: &str, value: &str) -> Result<()> {
        let c_name = CString::new(name)
            .map_err(|_| Error::InvalidArgument("set_option: name has interior NUL".into()))?;
        let c_value = CString::new(value)
            .map_err(|_| Error::InvalidArgument("set_option: value has interior NUL".into()))?;
        let status = unsafe {
            sys::aoclsparse_itsol_option_set(self.handle, c_name.as_ptr(), c_value.as_ptr())
        };
        check_status("sparse", status)
    }

    /// Solve `A · x = b`. On entry `x` should hold an initial guess (zero
    /// is fine if you have nothing better); on success it contains the
    /// approximate solution. Returns the solver's `rinfo[100]` array of
    /// statistics (iteration counts, residual norms, etc.).
    pub fn solve(
        &mut self,
        mat: &SparseMatrix<T>,
        descr: &MatDescr,
        b: &[T],
        x: &mut [T],
    ) -> Result<Box<[T; 100]>>
    where
        T: Default,
    {
        let n = mat.n;
        if mat.m != mat.n {
            return Err(Error::InvalidArgument(format!(
                "iterative solve requires square matrix; got ({}, {})", mat.m, mat.n
            )));
        }
        let mut rinfo: Box<[T; 100]> = Box::new([T::default(); 100]);
        T::itsol_solve(self.handle, n, mat.raw, descr, b, x, &mut rinfo)?;
        Ok(rinfo)
    }
}

impl<T: Scalar> Drop for IterSolver<T> {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                sys::aoclsparse_itsol_destroy(&mut self.handle);
            }
            self.handle = std::ptr::null_mut();
        }
    }
}

impl<T: Scalar> std::fmt::Debug for IterSolver<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IterSolver").finish_non_exhaustive()
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
    fn ellmv_2x3_f64() {
        // 2×3: [[1, 0, 2], [3, 4, 0]]; ell_width = 2 (max nnz/row).
        // Padding indices/values for the "missing" slot use index 0, value 0.
        let val: [f64; 4] = [1.0, 2.0, 3.0, 4.0];
        let col: [sys::aoclsparse_int; 4] = [0, 2, 0, 1];
        let descr = MatDescr::new().unwrap();
        let x = [10.0_f64, 20.0, 30.0];
        let mut y = [0.0_f64; 2];
        ellmv(Trans::No, 1.0_f64, 2, 3, &val, &col, 2, &descr, &x, 0.0, &mut y).unwrap();
        // y[0] = 1*10 + 2*30 = 70; y[1] = 3*10 + 4*20 = 110
        assert!((y[0] - 70.0).abs() < 1e-12, "got {}", y[0]);
        assert!((y[1] - 110.0).abs() < 1e-12, "got {}", y[1]);
    }

    #[test]
    fn bsrmv_2x2_blocks_f64() {
        // 4×4 matrix laid out as 2×2 blocks of size 2×2:
        // block (0,0) = [[1,0],[0,1]] (identity), block (1,1) = [[2,0],[0,2]]
        // bsr_dim = 2, mb = 2, nb = 2, nnzb = 2
        let val: [f64; 8] = [1.0, 0.0, 0.0, 1.0,  2.0, 0.0, 0.0, 2.0];
        let col: [sys::aoclsparse_int; 2] = [0, 1];
        let rp: [sys::aoclsparse_int; 3] = [0, 1, 2];
        let descr = MatDescr::new().unwrap();
        let x = [1.0_f64, 2.0, 3.0, 4.0];
        let mut y = [0.0_f64; 4];
        bsrmv(Trans::No, 1.0_f64, 2, 2, 2, &val, &col, &rp, &descr, &x, 0.0, &mut y).unwrap();
        // y = diag([1,1,2,2]) * x = [1, 2, 6, 8]
        assert!((y[0] - 1.0).abs() < 1e-12);
        assert!((y[1] - 2.0).abs() < 1e-12);
        assert!((y[2] - 6.0).abs() < 1e-12);
        assert!((y[3] - 8.0).abs() < 1e-12);
    }

    #[test]
    fn sparse_matrix_round_trip() {
        // 2×3 matrix [[1,0,2],[0,3,0]]
        let val = [1.0_f64, 2.0, 3.0];
        let col: [sys::aoclsparse_int; 3] = [0, 2, 1];
        let rp: [sys::aoclsparse_int; 3] = [0, 2, 3];
        let mat = SparseMatrix::<f64>::from_csr(IndexBase::Zero, 2, 3, &rp, &col, &val).unwrap();
        assert_eq!(mat.dims(), (2, 3));
        assert_eq!(mat.nnz(), 3);
        assert_eq!(mat.base(), IndexBase::Zero);
        let (base, rp2, col2, val2) = mat.export_csr().unwrap();
        assert_eq!(base, IndexBase::Zero);
        assert_eq!(rp2, [0, 2, 3]);
        assert_eq!(col2, [0, 2, 1]);
        assert_eq!(val2, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn csr2m_identity_squared_is_identity() {
        // 3×3 identity in CSR.
        let val = [1.0_f64; 3];
        let col: [sys::aoclsparse_int; 3] = [0, 1, 2];
        let rp: [sys::aoclsparse_int; 4] = [0, 1, 2, 3];
        let a = SparseMatrix::<f64>::from_csr(IndexBase::Zero, 3, 3, &rp, &col, &val).unwrap();
        let b = SparseMatrix::<f64>::from_csr(IndexBase::Zero, 3, 3, &rp, &col, &val).unwrap();
        let descr = MatDescr::new().unwrap();
        let c = csr2m(Trans::No, &descr, &a, Trans::No, &descr, &b,
                      Stage::FullComputation).unwrap();
        assert_eq!(c.dims(), (3, 3));
        let (_, rp_c, col_c, val_c) = c.export_csr().unwrap();
        assert_eq!(rp_c, [0, 1, 2, 3]);
        assert_eq!(col_c, [0, 1, 2]);
        for v in &val_c {
            assert!((v - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn iter_solver_cg_diagonal_3x3() {
        // Solve diag(2,2,2) · x = b, with b = [4, 6, 10]; expected x = [2, 3, 5].
        let val = [2.0_f64, 2.0, 2.0];
        let col: [sys::aoclsparse_int; 3] = [0, 1, 2];
        let rp: [sys::aoclsparse_int; 4] = [0, 1, 2, 3];
        let mat = SparseMatrix::<f64>::from_csr(IndexBase::Zero, 3, 3, &rp, &col, &val).unwrap();

        let descr = MatDescr::new().unwrap();
        unsafe {
            sys::aoclsparse_set_mat_type(
                descr.as_raw(),
                sys::aoclsparse_matrix_type__aoclsparse_matrix_type_symmetric,
            );
        }

        let b = [4.0_f64, 6.0, 10.0];
        let mut x = [0.0_f64; 3];
        let mut solver = IterSolver::<f64>::new().unwrap();
        solver.set_option("iterative method", "cg").unwrap();
        solver.set_option("cg rel tolerance", "1e-10").unwrap();
        solver.set_option("cg iteration limit", "200").unwrap();
        solver.solve(&mat, &descr, &b, &mut x).unwrap();
        assert!((x[0] - 2.0).abs() < 1e-6, "x[0] = {}", x[0]);
        assert!((x[1] - 3.0).abs() < 1e-6, "x[1] = {}", x[1]);
        assert!((x[2] - 5.0).abs() < 1e-6, "x[2] = {}", x[2]);
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
