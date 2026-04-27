//! Complex-precision sparse operations (`Complex32` / `Complex64`).
//!
//! These mirror the real-precision API: sparse vector AXPY / gather /
//! scatter, plus a [`ComplexSparseMatrix`] handle wrapping AOCL's
//! `aoclsparse_matrix` and operations against it: matrix–vector product
//! [`mv`], triangular solve [`trsv`], sparse-sparse product [`csr2m`],
//! ILU(0) smoother [`ilu_smoother`], and the iterative-solver suite via
//! [`ComplexIterSolver`].

use std::ffi::CString;
use std::marker::PhantomData;

use aocl_sparse_sys as sys;
use aocl_types::{sealed::Sealed, Complex32, Complex64};

use crate::{check_status, trans_raw, Error, IndexBase, MatDescr, Order, Result, SorType, Trans};

/// Scalar element type for the complex sparse routines (`c32`, `c64`).
pub trait ComplexScalar: Copy + Sized + Sealed {
    /// `y[indx] := α·x + y[indx]` (sparse vector AXPY).
    fn axpyi(alpha: Self, x: &[Self], indx: &[sys::aoclsparse_int], y: &mut [Self]) -> Result<()>;

    /// `x[i] := y[indx[i]]` (gather).
    fn gthr(y: &[Self], indx: &[sys::aoclsparse_int], x: &mut [Self]) -> Result<()>;

    /// `y[indx[i]] := x[i]` (scatter).
    fn sctr(x: &[Self], indx: &[sys::aoclsparse_int], y: &mut [Self]) -> Result<()>;

    /// Wrap raw CSR pointers in an `aoclsparse_matrix` handle.
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
    fn export_csr(
        mat: sys::aoclsparse_matrix,
    ) -> Result<(IndexBase, usize, usize, usize,
                 *mut sys::aoclsparse_int, *mut sys::aoclsparse_int, *mut Self)>;

    /// `y := α · op(A) · x + β · y` via the high-level matrix handle.
    #[allow(clippy::too_many_arguments)]
    fn mv(
        op: Trans,
        alpha: Self,
        mat: sys::aoclsparse_matrix,
        descr: &MatDescr,
        x: &[Self],
        beta: Self,
        y: &mut [Self],
    ) -> Result<()>;

    /// Solve `op(A) · x = α · b` where `A` is sparse triangular.
    #[allow(clippy::too_many_arguments)]
    fn trsv(
        op: Trans,
        alpha: Self,
        mat: sys::aoclsparse_matrix,
        descr: &MatDescr,
        b: &[Self],
        x: &mut [Self],
    ) -> Result<()>;

    /// One ILU(0) smoothing step against the matrix handle.
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
        rinfo: &mut [Self::Real; 100],
    ) -> Result<()>;

    /// `C := α · op(A) · B + β · C` where `A` is sparse (CSR via the
    /// matrix handle) and `B`, `C` are dense.
    #[allow(clippy::too_many_arguments)]
    fn csrmm(
        op: Trans,
        alpha: Self,
        a: sys::aoclsparse_matrix,
        descr: &MatDescr,
        order: Order,
        b: &[Self],
        n: usize,
        ldb: usize,
        beta: Self,
        c: &mut [Self],
        ldc: usize,
    ) -> Result<()>;

    /// `C := op(A) · B` where both `A` and `B` are sparse and `C` is dense.
    #[allow(clippy::too_many_arguments)]
    fn spmmd(
        op: Trans,
        a: sys::aoclsparse_matrix,
        b: sys::aoclsparse_matrix,
        layout: Order,
        c: &mut [Self],
        ldc: usize,
    ) -> Result<()>;

    /// `C := α · op_A(A) · op_B(B) + β · C` (sparse·sparse → dense).
    #[allow(clippy::too_many_arguments)]
    fn sp2md(
        op_a: Trans,
        descr_a: &MatDescr,
        a: sys::aoclsparse_matrix,
        op_b: Trans,
        descr_b: &MatDescr,
        b: sys::aoclsparse_matrix,
        alpha: Self,
        beta: Self,
        c: &mut [Self],
        layout: Order,
        ldc: usize,
    ) -> Result<()>;

    /// Sparse-sparse `C := α · op(A) + B`.
    ///
    /// # Safety
    /// Caller must adopt the resulting handle (e.g. via
    /// [`ComplexSparseMatrix::from_library_owned`]).
    unsafe fn add_ffi(
        op: sys::aoclsparse_operation,
        a: sys::aoclsparse_matrix,
        alpha: Self,
        b: sys::aoclsparse_matrix,
        out: *mut sys::aoclsparse_matrix,
    ) -> sys::aoclsparse_status;

    /// One step of (S)SOR / forward / backward Gauss-Seidel relaxation.
    #[allow(clippy::too_many_arguments)]
    fn sorv(
        sor_type: SorType,
        descr: &MatDescr,
        a: sys::aoclsparse_matrix,
        omega: Self,
        alpha: Self,
        x: &mut [Self],
        b: &[Self],
    ) -> Result<()>;

    /// Underlying real type — `f32` for `Complex32`, `f64` for `Complex64`.
    /// Used to type the `rinfo[100]` array returned by iterative solvers,
    /// which is real-valued even for complex problems.
    type Real: Copy + Default;
}

// =========================================================================
//   Complex32 (c32) impl
// =========================================================================

impl ComplexScalar for Complex32 {
    type Real = f32;

    fn axpyi(alpha: Self, x: &[Self], indx: &[sys::aoclsparse_int], y: &mut [Self]) -> Result<()> {
        let status = unsafe {
            sys::aoclsparse_caxpyi(
                x.len() as sys::aoclsparse_int,
                &alpha as *const _ as *const std::os::raw::c_void,
                x.as_ptr() as *const std::os::raw::c_void,
                indx.as_ptr(),
                y.as_mut_ptr() as *mut std::os::raw::c_void,
            )
        };
        check_status("sparse", status)
    }

    fn gthr(y: &[Self], indx: &[sys::aoclsparse_int], x: &mut [Self]) -> Result<()> {
        let status = unsafe {
            sys::aoclsparse_cgthr(
                x.len() as sys::aoclsparse_int,
                y.as_ptr() as *const std::os::raw::c_void,
                x.as_mut_ptr() as *mut std::os::raw::c_void,
                indx.as_ptr(),
            )
        };
        check_status("sparse", status)
    }

    fn sctr(x: &[Self], indx: &[sys::aoclsparse_int], y: &mut [Self]) -> Result<()> {
        let status = unsafe {
            sys::aoclsparse_csctr(
                x.len() as sys::aoclsparse_int,
                x.as_ptr() as *const std::os::raw::c_void,
                indx.as_ptr(),
                y.as_mut_ptr() as *mut std::os::raw::c_void,
            )
        };
        check_status("sparse", status)
    }

    fn create_csr(
        base: IndexBase, m: usize, n: usize, nnz: usize,
        row_ptr: *mut sys::aoclsparse_int,
        col_idx: *mut sys::aoclsparse_int,
        val: *mut Self,
    ) -> Result<sys::aoclsparse_matrix> {
        let mut raw: sys::aoclsparse_matrix = std::ptr::null_mut();
        let status = unsafe {
            sys::aoclsparse_create_ccsr(
                &mut raw,
                base.raw_for_complex(),
                m as sys::aoclsparse_int,
                n as sys::aoclsparse_int,
                nnz as sys::aoclsparse_int,
                row_ptr, col_idx,
                val as *mut sys::aoclsparse_float_complex,
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
        let mut val: *mut sys::aoclsparse_float_complex = std::ptr::null_mut();
        let status = unsafe {
            sys::aoclsparse_export_ccsr(
                mat, &mut base, &mut m, &mut n, &mut nnz,
                &mut row_ptr, &mut col_ind, &mut val,
            )
        };
        check_status("sparse", status)?;
        let base_e = if base == sys::aoclsparse_index_base__aoclsparse_index_base_one {
            IndexBase::One
        } else {
            IndexBase::Zero
        };
        Ok((base_e, m as usize, n as usize, nnz as usize,
            row_ptr, col_ind, val as *mut Self))
    }

    fn mv(
        op: Trans, alpha: Self, mat: sys::aoclsparse_matrix, descr: &MatDescr,
        x: &[Self], beta: Self, y: &mut [Self],
    ) -> Result<()> {
        let status = unsafe {
            sys::aoclsparse_cmv(
                trans_raw(op),
                &alpha as *const _ as *const sys::aoclsparse_float_complex,
                mat,
                descr.as_raw(),
                x.as_ptr() as *const sys::aoclsparse_float_complex,
                &beta as *const _ as *const sys::aoclsparse_float_complex,
                y.as_mut_ptr() as *mut sys::aoclsparse_float_complex,
            )
        };
        check_status("sparse", status)
    }

    fn trsv(
        op: Trans, alpha: Self, mat: sys::aoclsparse_matrix, descr: &MatDescr,
        b: &[Self], x: &mut [Self],
    ) -> Result<()> {
        // aoclsparse_ctrsv takes alpha by-value (struct), unlike mv.
        let alpha_raw = sys::aoclsparse_float_complex { real: alpha.re, imag: alpha.im };
        let status = unsafe {
            sys::aoclsparse_ctrsv(
                trans_raw(op),
                alpha_raw,
                mat,
                descr.as_raw(),
                b.as_ptr() as *const sys::aoclsparse_float_complex,
                x.as_mut_ptr() as *mut sys::aoclsparse_float_complex,
            )
        };
        check_status("sparse", status)
    }

    fn ilu_smoother(
        op: Trans, a: sys::aoclsparse_matrix, descr: &MatDescr,
        x: &mut [Self], b: &[Self],
    ) -> Result<()> {
        let mut precond_csr_val: *mut sys::aoclsparse_float_complex = std::ptr::null_mut();
        let status = unsafe {
            sys::aoclsparse_cilu_smoother(
                trans_raw(op), a, descr.as_raw(),
                &mut precond_csr_val,
                std::ptr::null(),
                x.as_mut_ptr() as *mut sys::aoclsparse_float_complex,
                b.as_ptr() as *const sys::aoclsparse_float_complex,
            )
        };
        check_status("sparse", status)
    }

    fn itsol_init(handle: &mut sys::aoclsparse_itsol_handle) -> Result<()> {
        let status = unsafe { sys::aoclsparse_itsol_c_init(handle) };
        check_status("sparse", status)
    }

    fn itsol_solve(
        handle: sys::aoclsparse_itsol_handle,
        n: usize,
        mat: sys::aoclsparse_matrix,
        descr: &MatDescr,
        b: &[Self], x: &mut [Self],
        rinfo: &mut [f32; 100],
    ) -> Result<()> {
        if b.len() < n || x.len() < n {
            return Err(Error::InvalidArgument(format!(
                "itsol_solve: b.len()={}, x.len()={}, n={n}", b.len(), x.len()
            )));
        }
        let status = unsafe {
            sys::aoclsparse_itsol_c_solve(
                handle,
                n as sys::aoclsparse_int,
                mat,
                descr.as_raw(),
                b.as_ptr() as *const sys::aoclsparse_float_complex,
                x.as_mut_ptr() as *mut sys::aoclsparse_float_complex,
                rinfo.as_mut_ptr(),
                None, None,
                std::ptr::null_mut(),
            )
        };
        check_status("sparse", status)
    }

    fn csrmm(
        op: Trans, alpha: Self, a: sys::aoclsparse_matrix, descr: &MatDescr,
        order: Order, b: &[Self], n: usize, ldb: usize,
        beta: Self, c: &mut [Self], ldc: usize,
    ) -> Result<()> {
        let alpha_raw = sys::aoclsparse_float_complex { real: alpha.re, imag: alpha.im };
        let beta_raw = sys::aoclsparse_float_complex { real: beta.re, imag: beta.im };
        let status = unsafe {
            sys::aoclsparse_ccsrmm(
                trans_raw(op), alpha_raw, a, descr.as_raw(), order.raw(),
                b.as_ptr() as *const sys::aoclsparse_float_complex,
                n as sys::aoclsparse_int, ldb as sys::aoclsparse_int,
                beta_raw,
                c.as_mut_ptr() as *mut sys::aoclsparse_float_complex,
                ldc as sys::aoclsparse_int,
            )
        };
        check_status("sparse", status)
    }

    fn spmmd(
        op: Trans, a: sys::aoclsparse_matrix, b: sys::aoclsparse_matrix,
        layout: Order, c: &mut [Self], ldc: usize,
    ) -> Result<()> {
        let status = unsafe {
            sys::aoclsparse_cspmmd(
                trans_raw(op), a, b, layout.raw(),
                c.as_mut_ptr() as *mut sys::aoclsparse_float_complex,
                ldc as sys::aoclsparse_int,
            )
        };
        check_status("sparse", status)
    }

    fn sp2md(
        op_a: Trans, descr_a: &MatDescr, a: sys::aoclsparse_matrix,
        op_b: Trans, descr_b: &MatDescr, b: sys::aoclsparse_matrix,
        alpha: Self, beta: Self, c: &mut [Self], layout: Order, ldc: usize,
    ) -> Result<()> {
        let alpha_raw = sys::aoclsparse_float_complex { real: alpha.re, imag: alpha.im };
        let beta_raw = sys::aoclsparse_float_complex { real: beta.re, imag: beta.im };
        let status = unsafe {
            sys::aoclsparse_csp2md(
                trans_raw(op_a), descr_a.as_raw(), a,
                trans_raw(op_b), descr_b.as_raw(), b,
                alpha_raw, beta_raw,
                c.as_mut_ptr() as *mut sys::aoclsparse_float_complex,
                layout.raw(), ldc as sys::aoclsparse_int,
            )
        };
        check_status("sparse", status)
    }

    unsafe fn add_ffi(
        op: sys::aoclsparse_operation,
        a: sys::aoclsparse_matrix,
        alpha: Self,
        b: sys::aoclsparse_matrix,
        out: *mut sys::aoclsparse_matrix,
    ) -> sys::aoclsparse_status {
        let alpha_raw = sys::aoclsparse_float_complex { real: alpha.re, imag: alpha.im };
        sys::aoclsparse_cadd(op, a, alpha_raw, b, out)
    }

    fn sorv(
        sor_type: SorType, descr: &MatDescr, a: sys::aoclsparse_matrix,
        omega: Self, alpha: Self, x: &mut [Self], b: &[Self],
    ) -> Result<()> {
        let omega_raw = sys::aoclsparse_float_complex { real: omega.re, imag: omega.im };
        let alpha_raw = sys::aoclsparse_float_complex { real: alpha.re, imag: alpha.im };
        let status = unsafe {
            sys::aoclsparse_csorv(
                sor_type.raw(), descr.as_raw(), a, omega_raw, alpha_raw,
                x.as_mut_ptr() as *mut sys::aoclsparse_float_complex,
                b.as_ptr() as *const sys::aoclsparse_float_complex,
            )
        };
        check_status("sparse", status)
    }
}

// =========================================================================
//   Complex64 (c64) impl
// =========================================================================

impl ComplexScalar for Complex64 {
    type Real = f64;

    fn axpyi(alpha: Self, x: &[Self], indx: &[sys::aoclsparse_int], y: &mut [Self]) -> Result<()> {
        let status = unsafe {
            sys::aoclsparse_zaxpyi(
                x.len() as sys::aoclsparse_int,
                &alpha as *const _ as *const std::os::raw::c_void,
                x.as_ptr() as *const std::os::raw::c_void,
                indx.as_ptr(),
                y.as_mut_ptr() as *mut std::os::raw::c_void,
            )
        };
        check_status("sparse", status)
    }

    fn gthr(y: &[Self], indx: &[sys::aoclsparse_int], x: &mut [Self]) -> Result<()> {
        let status = unsafe {
            sys::aoclsparse_zgthr(
                x.len() as sys::aoclsparse_int,
                y.as_ptr() as *const std::os::raw::c_void,
                x.as_mut_ptr() as *mut std::os::raw::c_void,
                indx.as_ptr(),
            )
        };
        check_status("sparse", status)
    }

    fn sctr(x: &[Self], indx: &[sys::aoclsparse_int], y: &mut [Self]) -> Result<()> {
        let status = unsafe {
            sys::aoclsparse_zsctr(
                x.len() as sys::aoclsparse_int,
                x.as_ptr() as *const std::os::raw::c_void,
                indx.as_ptr(),
                y.as_mut_ptr() as *mut std::os::raw::c_void,
            )
        };
        check_status("sparse", status)
    }

    fn create_csr(
        base: IndexBase, m: usize, n: usize, nnz: usize,
        row_ptr: *mut sys::aoclsparse_int,
        col_idx: *mut sys::aoclsparse_int,
        val: *mut Self,
    ) -> Result<sys::aoclsparse_matrix> {
        let mut raw: sys::aoclsparse_matrix = std::ptr::null_mut();
        let status = unsafe {
            sys::aoclsparse_create_zcsr(
                &mut raw,
                base.raw_for_complex(),
                m as sys::aoclsparse_int,
                n as sys::aoclsparse_int,
                nnz as sys::aoclsparse_int,
                row_ptr, col_idx,
                val as *mut sys::aoclsparse_double_complex,
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
        let mut val: *mut sys::aoclsparse_double_complex = std::ptr::null_mut();
        let status = unsafe {
            sys::aoclsparse_export_zcsr(
                mat, &mut base, &mut m, &mut n, &mut nnz,
                &mut row_ptr, &mut col_ind, &mut val,
            )
        };
        check_status("sparse", status)?;
        let base_e = if base == sys::aoclsparse_index_base__aoclsparse_index_base_one {
            IndexBase::One
        } else {
            IndexBase::Zero
        };
        Ok((base_e, m as usize, n as usize, nnz as usize,
            row_ptr, col_ind, val as *mut Self))
    }

    fn mv(
        op: Trans, alpha: Self, mat: sys::aoclsparse_matrix, descr: &MatDescr,
        x: &[Self], beta: Self, y: &mut [Self],
    ) -> Result<()> {
        let status = unsafe {
            sys::aoclsparse_zmv(
                trans_raw(op),
                &alpha as *const _ as *const sys::aoclsparse_double_complex,
                mat,
                descr.as_raw(),
                x.as_ptr() as *const sys::aoclsparse_double_complex,
                &beta as *const _ as *const sys::aoclsparse_double_complex,
                y.as_mut_ptr() as *mut sys::aoclsparse_double_complex,
            )
        };
        check_status("sparse", status)
    }

    fn trsv(
        op: Trans, alpha: Self, mat: sys::aoclsparse_matrix, descr: &MatDescr,
        b: &[Self], x: &mut [Self],
    ) -> Result<()> {
        let alpha_raw = sys::aoclsparse_double_complex_ { real: alpha.re, imag: alpha.im };
        let status = unsafe {
            sys::aoclsparse_ztrsv(
                trans_raw(op),
                alpha_raw,
                mat,
                descr.as_raw(),
                b.as_ptr() as *const sys::aoclsparse_double_complex,
                x.as_mut_ptr() as *mut sys::aoclsparse_double_complex,
            )
        };
        check_status("sparse", status)
    }

    fn ilu_smoother(
        op: Trans, a: sys::aoclsparse_matrix, descr: &MatDescr,
        x: &mut [Self], b: &[Self],
    ) -> Result<()> {
        let mut precond_csr_val: *mut sys::aoclsparse_double_complex = std::ptr::null_mut();
        let status = unsafe {
            sys::aoclsparse_zilu_smoother(
                trans_raw(op), a, descr.as_raw(),
                &mut precond_csr_val,
                std::ptr::null(),
                x.as_mut_ptr() as *mut sys::aoclsparse_double_complex,
                b.as_ptr() as *const sys::aoclsparse_double_complex,
            )
        };
        check_status("sparse", status)
    }

    fn itsol_init(handle: &mut sys::aoclsparse_itsol_handle) -> Result<()> {
        let status = unsafe { sys::aoclsparse_itsol_z_init(handle) };
        check_status("sparse", status)
    }

    fn itsol_solve(
        handle: sys::aoclsparse_itsol_handle,
        n: usize,
        mat: sys::aoclsparse_matrix,
        descr: &MatDescr,
        b: &[Self], x: &mut [Self],
        rinfo: &mut [f64; 100],
    ) -> Result<()> {
        if b.len() < n || x.len() < n {
            return Err(Error::InvalidArgument(format!(
                "itsol_solve: b.len()={}, x.len()={}, n={n}", b.len(), x.len()
            )));
        }
        let status = unsafe {
            sys::aoclsparse_itsol_z_solve(
                handle,
                n as sys::aoclsparse_int,
                mat,
                descr.as_raw(),
                b.as_ptr() as *const sys::aoclsparse_double_complex,
                x.as_mut_ptr() as *mut sys::aoclsparse_double_complex,
                rinfo.as_mut_ptr(),
                None, None,
                std::ptr::null_mut(),
            )
        };
        check_status("sparse", status)
    }

    fn csrmm(
        op: Trans, alpha: Self, a: sys::aoclsparse_matrix, descr: &MatDescr,
        order: Order, b: &[Self], n: usize, ldb: usize,
        beta: Self, c: &mut [Self], ldc: usize,
    ) -> Result<()> {
        let alpha_raw = sys::aoclsparse_double_complex_ { real: alpha.re, imag: alpha.im };
        let beta_raw = sys::aoclsparse_double_complex_ { real: beta.re, imag: beta.im };
        let status = unsafe {
            sys::aoclsparse_zcsrmm(
                trans_raw(op), alpha_raw, a, descr.as_raw(), order.raw(),
                b.as_ptr() as *const sys::aoclsparse_double_complex,
                n as sys::aoclsparse_int, ldb as sys::aoclsparse_int,
                beta_raw,
                c.as_mut_ptr() as *mut sys::aoclsparse_double_complex,
                ldc as sys::aoclsparse_int,
            )
        };
        check_status("sparse", status)
    }

    fn spmmd(
        op: Trans, a: sys::aoclsparse_matrix, b: sys::aoclsparse_matrix,
        layout: Order, c: &mut [Self], ldc: usize,
    ) -> Result<()> {
        let status = unsafe {
            sys::aoclsparse_zspmmd(
                trans_raw(op), a, b, layout.raw(),
                c.as_mut_ptr() as *mut sys::aoclsparse_double_complex,
                ldc as sys::aoclsparse_int,
            )
        };
        check_status("sparse", status)
    }

    fn sp2md(
        op_a: Trans, descr_a: &MatDescr, a: sys::aoclsparse_matrix,
        op_b: Trans, descr_b: &MatDescr, b: sys::aoclsparse_matrix,
        alpha: Self, beta: Self, c: &mut [Self], layout: Order, ldc: usize,
    ) -> Result<()> {
        let alpha_raw = sys::aoclsparse_double_complex_ { real: alpha.re, imag: alpha.im };
        let beta_raw = sys::aoclsparse_double_complex_ { real: beta.re, imag: beta.im };
        let status = unsafe {
            sys::aoclsparse_zsp2md(
                trans_raw(op_a), descr_a.as_raw(), a,
                trans_raw(op_b), descr_b.as_raw(), b,
                alpha_raw, beta_raw,
                c.as_mut_ptr() as *mut sys::aoclsparse_double_complex,
                layout.raw(), ldc as sys::aoclsparse_int,
            )
        };
        check_status("sparse", status)
    }

    unsafe fn add_ffi(
        op: sys::aoclsparse_operation,
        a: sys::aoclsparse_matrix,
        alpha: Self,
        b: sys::aoclsparse_matrix,
        out: *mut sys::aoclsparse_matrix,
    ) -> sys::aoclsparse_status {
        let alpha_raw = sys::aoclsparse_double_complex_ { real: alpha.re, imag: alpha.im };
        sys::aoclsparse_zadd(op, a, alpha_raw, b, out)
    }

    fn sorv(
        sor_type: SorType, descr: &MatDescr, a: sys::aoclsparse_matrix,
        omega: Self, alpha: Self, x: &mut [Self], b: &[Self],
    ) -> Result<()> {
        let omega_raw = sys::aoclsparse_double_complex_ { real: omega.re, imag: omega.im };
        let alpha_raw = sys::aoclsparse_double_complex_ { real: alpha.re, imag: alpha.im };
        let status = unsafe {
            sys::aoclsparse_zsorv(
                sor_type.raw(), descr.as_raw(), a, omega_raw, alpha_raw,
                x.as_mut_ptr() as *mut sys::aoclsparse_double_complex,
                b.as_ptr() as *const sys::aoclsparse_double_complex,
            )
        };
        check_status("sparse", status)
    }
}

// =========================================================================
//   Complex sparse matrix handle
// =========================================================================

enum CsrStorage<T: ComplexScalar> {
    Owned {
        _row_ptr: Vec<sys::aoclsparse_int>,
        _col_ind: Vec<sys::aoclsparse_int>,
        _val: Vec<T>,
    },
    LibraryOwned,
}

/// RAII wrapper for a complex `aoclsparse_matrix` in CSR format.
pub struct ComplexSparseMatrix<T: ComplexScalar> {
    raw: sys::aoclsparse_matrix,
    #[allow(dead_code)]
    storage: CsrStorage<T>,
    base: IndexBase,
    m: usize,
    n: usize,
    nnz: usize,
}

impl<T: ComplexScalar> ComplexSparseMatrix<T> {
    /// Build from CSR arrays. Values are copied into the wrapper.
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
                "complex from_csr: row_ptr length {} != m+1 = {}",
                row_ptr.len(), m + 1
            )));
        }
        let nnz = val.len();
        if col_ind.len() != nnz {
            return Err(Error::InvalidArgument(format!(
                "complex from_csr: col_ind length {} != val length {nnz}",
                col_ind.len()
            )));
        }
        let mut row_ptr = row_ptr.to_vec();
        let mut col_ind = col_ind.to_vec();
        let mut val = val.to_vec();
        let raw = T::create_csr(
            base, m, n, nnz,
            row_ptr.as_mut_ptr(),
            col_ind.as_mut_ptr(),
            val.as_mut_ptr(),
        )?;
        Ok(Self {
            raw,
            storage: CsrStorage::Owned { _row_ptr: row_ptr, _col_ind: col_ind, _val: val },
            base, m, n, nnz,
        })
    }

    /// Adopt a library-allocated matrix handle (e.g. the result of
    /// [`csr2m`]).
    ///
    /// # Safety
    /// `raw` must be a valid `aoclsparse_matrix` whose precision matches
    /// `T` and whose internal storage is library-owned.
    pub unsafe fn from_library_owned(raw: sys::aoclsparse_matrix) -> Result<Self> {
        if raw.is_null() {
            return Err(Error::AllocationFailed("sparse"));
        }
        let (base, m, n, nnz, _, _, _) = T::export_csr(raw)?;
        Ok(Self { raw, storage: CsrStorage::LibraryOwned, base, m, n, nnz })
    }

    /// `(m, n)` dimensions.
    pub fn dims(&self) -> (usize, usize) { (self.m, self.n) }
    /// Number of explicitly stored non-zeros.
    pub fn nnz(&self) -> usize { self.nnz }
    /// Index base for row-pointer / column-index arrays.
    pub fn base(&self) -> IndexBase { self.base }

    /// Borrow the raw `aoclsparse_matrix` handle. Do not call
    /// `aoclsparse_destroy` on it; the wrapper does.
    pub fn as_raw(&self) -> sys::aoclsparse_matrix { self.raw }

    /// Read out the CSR contents as freshly allocated `Vec`s.
    pub fn export_csr(&self) -> Result<(IndexBase, Vec<sys::aoclsparse_int>, Vec<sys::aoclsparse_int>, Vec<T>)> {
        let (base, m, _, nnz, row_ptr, col_ind, val) = T::export_csr(self.raw)?;
        let row_ptr = unsafe { std::slice::from_raw_parts(row_ptr, m + 1).to_vec() };
        let col_ind = unsafe { std::slice::from_raw_parts(col_ind, nnz).to_vec() };
        let val = unsafe { std::slice::from_raw_parts(val, nnz).to_vec() };
        Ok((base, row_ptr, col_ind, val))
    }
}

impl<T: ComplexScalar> Drop for ComplexSparseMatrix<T> {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { let _ = sys::aoclsparse_destroy(&mut self.raw); }
            self.raw = std::ptr::null_mut();
        }
    }
}

impl<T: ComplexScalar> std::fmt::Debug for ComplexSparseMatrix<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ComplexSparseMatrix")
            .field("m", &self.m)
            .field("n", &self.n)
            .field("nnz", &self.nnz)
            .field("base", &self.base)
            .finish()
    }
}

// =========================================================================
//   Free functions
// =========================================================================

/// `y[indx] := α·x + y[indx]`. `x` and `indx` must have equal length.
pub fn axpyi<T: ComplexScalar>(
    alpha: T, x: &[T], indx: &[sys::aoclsparse_int], y: &mut [T],
) -> Result<()> {
    if x.len() != indx.len() {
        return Err(Error::InvalidArgument(format!(
            "axpyi: x.len()={}, indx.len()={}", x.len(), indx.len()
        )));
    }
    T::axpyi(alpha, x, indx, y)
}

/// `x[i] := y[indx[i]]` (sparse gather).
pub fn gthr<T: ComplexScalar>(
    y: &[T], indx: &[sys::aoclsparse_int], x: &mut [T],
) -> Result<()> {
    if x.len() != indx.len() {
        return Err(Error::InvalidArgument(format!(
            "gthr: x.len()={}, indx.len()={}", x.len(), indx.len()
        )));
    }
    T::gthr(y, indx, x)
}

/// `y[indx[i]] := x[i]` (sparse scatter).
pub fn sctr<T: ComplexScalar>(
    x: &[T], indx: &[sys::aoclsparse_int], y: &mut [T],
) -> Result<()> {
    if x.len() != indx.len() {
        return Err(Error::InvalidArgument(format!(
            "sctr: x.len()={}, indx.len()={}", x.len(), indx.len()
        )));
    }
    T::sctr(x, indx, y)
}

/// Compute `y := α · op(A) · x + β · y` for a complex CSR matrix `A`.
#[allow(clippy::too_many_arguments)]
pub fn mv<T: ComplexScalar>(
    op: Trans,
    alpha: T,
    a: &ComplexSparseMatrix<T>,
    descr: &MatDescr,
    x: &[T],
    beta: T,
    y: &mut [T],
) -> Result<()> {
    let (x_len, y_len) = match op {
        Trans::No => (a.n, a.m),
        Trans::T | Trans::C => (a.m, a.n),
    };
    if x.len() < x_len || y.len() < y_len {
        return Err(Error::InvalidArgument(format!(
            "mv: x.len()={}, y.len()={}, expected ({x_len}, {y_len})",
            x.len(), y.len()
        )));
    }
    T::mv(op, alpha, a.raw, descr, x, beta, y)
}

/// Solve `op(A) · x = α · b` where `A` is sparse triangular.
#[allow(clippy::too_many_arguments)]
pub fn trsv<T: ComplexScalar>(
    op: Trans,
    alpha: T,
    a: &ComplexSparseMatrix<T>,
    descr: &MatDescr,
    b: &[T],
    x: &mut [T],
) -> Result<()> {
    if b.len() < a.m || x.len() < a.m {
        return Err(Error::InvalidArgument(format!(
            "trsv: b.len()={}, x.len()={}, m={}", b.len(), x.len(), a.m
        )));
    }
    T::trsv(op, alpha, a.raw, descr, b, x)
}

/// Compute `C := α · op(A) · B + β · C` where `A` is sparse complex
/// (CSR) and `B`, `C` are dense complex matrices laid out per `order`.
#[allow(clippy::too_many_arguments)]
pub fn csrmm<T: ComplexScalar>(
    op: Trans,
    alpha: T,
    a: &ComplexSparseMatrix<T>,
    descr: &MatDescr,
    order: Order,
    b: &[T],
    n: usize,
    ldb: usize,
    beta: T,
    c: &mut [T],
    ldc: usize,
) -> Result<()> {
    T::csrmm(op, alpha, a.as_raw(), descr, order, b, n, ldb, beta, c, ldc)
}

/// Compute `C := op(A) · B` where both `A` and `B` are sparse complex
/// matrices and `C` is dense.
pub fn spmmd<T: ComplexScalar>(
    op: Trans,
    a: &ComplexSparseMatrix<T>,
    b: &ComplexSparseMatrix<T>,
    layout: Order,
    c: &mut [T],
    ldc: usize,
) -> Result<()> {
    T::spmmd(op, a.as_raw(), b.as_raw(), layout, c, ldc)
}

/// Compute `C := α · op_A(A) · op_B(B) + β · C` for complex sparse `A`
/// and `B`, dense complex `C`.
#[allow(clippy::too_many_arguments)]
pub fn sp2md<T: ComplexScalar>(
    op_a: Trans,
    descr_a: &MatDescr,
    a: &ComplexSparseMatrix<T>,
    op_b: Trans,
    descr_b: &MatDescr,
    b: &ComplexSparseMatrix<T>,
    alpha: T,
    beta: T,
    c: &mut [T],
    layout: Order,
    ldc: usize,
) -> Result<()> {
    T::sp2md(op_a, descr_a, a.as_raw(), op_b, descr_b, b.as_raw(),
             alpha, beta, c, layout, ldc)
}

/// Compute `C := α · op(A) + B` returning a fresh complex CSR matrix.
pub fn add<T: ComplexScalar>(
    op: Trans,
    a: &ComplexSparseMatrix<T>,
    alpha: T,
    b: &ComplexSparseMatrix<T>,
) -> Result<ComplexSparseMatrix<T>> {
    let mut c_raw: sys::aoclsparse_matrix = std::ptr::null_mut();
    let status = unsafe {
        T::add_ffi(trans_raw(op), a.as_raw(), alpha, b.as_raw(), &mut c_raw)
    };
    check_status("sparse", status)?;
    unsafe { ComplexSparseMatrix::from_library_owned(c_raw) }
}

/// One step of (S)SOR / forward / backward Gauss-Seidel relaxation for a
/// complex sparse matrix.
pub fn sorv<T: ComplexScalar>(
    sor_type: SorType,
    descr: &MatDescr,
    a: &ComplexSparseMatrix<T>,
    omega: T,
    alpha: T,
    x: &mut [T],
    b: &[T],
) -> Result<()> {
    if x.len() < a.dims().1 || b.len() < a.dims().0 {
        return Err(Error::InvalidArgument(format!(
            "sorv: x.len()={}, b.len()={}, dims=({}, {})",
            x.len(), b.len(), a.dims().0, a.dims().1
        )));
    }
    T::sorv(sor_type, descr, a.as_raw(), omega, alpha, x, b)
}

/// One ILU(0) smoothing step on a complex matrix.
pub fn ilu_smoother<T: ComplexScalar>(
    op: Trans,
    a: &ComplexSparseMatrix<T>,
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
//   Iterative solver (CG / GMRES) for complex
// =========================================================================

/// Iterative-solver handle for complex sparse systems.
pub struct ComplexIterSolver<T: ComplexScalar> {
    handle: sys::aoclsparse_itsol_handle,
    _marker: PhantomData<T>,
}

impl<T: ComplexScalar> ComplexIterSolver<T> {
    /// Initialise a new handle.
    pub fn new() -> Result<Self> {
        let mut handle: sys::aoclsparse_itsol_handle = std::ptr::null_mut();
        T::itsol_init(&mut handle)?;
        if handle.is_null() {
            return Err(Error::AllocationFailed("sparse"));
        }
        Ok(Self { handle, _marker: PhantomData })
    }

    /// Set a string-keyed solver option (see `aoclsparse_itsol_option_set`).
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

    /// Solve `A · x = b`. `rinfo[100]` of solver stats is real-typed.
    pub fn solve(
        &mut self,
        mat: &ComplexSparseMatrix<T>,
        descr: &MatDescr,
        b: &[T],
        x: &mut [T],
    ) -> Result<Box<[T::Real; 100]>> {
        let n = mat.n;
        if mat.m != mat.n {
            return Err(Error::InvalidArgument(format!(
                "iterative solve requires square matrix; got ({}, {})", mat.m, mat.n
            )));
        }
        let mut rinfo: Box<[T::Real; 100]> = Box::new([T::Real::default(); 100]);
        T::itsol_solve(self.handle, n, mat.raw, descr, b, x, &mut rinfo)?;
        Ok(rinfo)
    }
}

impl<T: ComplexScalar> Drop for ComplexIterSolver<T> {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { sys::aoclsparse_itsol_destroy(&mut self.handle); }
            self.handle = std::ptr::null_mut();
        }
    }
}

impl<T: ComplexScalar> std::fmt::Debug for ComplexIterSolver<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ComplexIterSolver").finish_non_exhaustive()
    }
}

// IndexBase helper used above. Mirrors the real API's internal `raw()`
// accessor without making the function public to outside crates.
impl IndexBase {
    pub(crate) fn raw_for_complex(self) -> sys::aoclsparse_index_base {
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
    fn axpyi_c64_round_trip() {
        // y = [1+0i, 2+0i, 3+0i, 4+0i]; x = [10+1i, 20+2i] at indx [0, 2]; α = (3+0i).
        // → y[0] += 3·(10+1i) = 30+3i; y[0] = 31+3i; y[2] += 3·(20+2i) = 60+6i; y[2] = 63+6i.
        let mut y = [
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
        ];
        let x = [Complex64::new(10.0, 1.0), Complex64::new(20.0, 2.0)];
        let indx: [sys::aoclsparse_int; 2] = [0, 2];
        axpyi(Complex64::new(3.0, 0.0), &x, &indx, &mut y).unwrap();
        assert!((y[0].re - 31.0).abs() < 1e-12);
        assert!((y[0].im - 3.0).abs() < 1e-12);
        assert!((y[2].re - 63.0).abs() < 1e-12);
        assert!((y[2].im - 6.0).abs() < 1e-12);
        assert!((y[1].re - 2.0).abs() < 1e-12);
        assert!((y[3].re - 4.0).abs() < 1e-12);
    }

    #[test]
    fn complex_sparse_matrix_round_trip_c64() {
        // 2×2 identity with imaginary-part = 0
        let val = [Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)];
        let col: [sys::aoclsparse_int; 2] = [0, 1];
        let rp: [sys::aoclsparse_int; 3] = [0, 1, 2];
        let mat = ComplexSparseMatrix::<Complex64>::from_csr(
            IndexBase::Zero, 2, 2, &rp, &col, &val).unwrap();
        assert_eq!(mat.dims(), (2, 2));
        assert_eq!(mat.nnz(), 2);
        let (base, rp2, col2, val2) = mat.export_csr().unwrap();
        assert_eq!(base, IndexBase::Zero);
        assert_eq!(rp2, [0, 1, 2]);
        assert_eq!(col2, [0, 1]);
        assert!((val2[0].re - 1.0).abs() < 1e-12);
    }

    #[test]
    fn complex_mv_identity_c64() {
        // 2×2 identity matrix; A·x should equal x.
        let val = [Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)];
        let col: [sys::aoclsparse_int; 2] = [0, 1];
        let rp: [sys::aoclsparse_int; 3] = [0, 1, 2];
        let mat = ComplexSparseMatrix::<Complex64>::from_csr(
            IndexBase::Zero, 2, 2, &rp, &col, &val).unwrap();
        let descr = MatDescr::new().unwrap();
        let x = [Complex64::new(3.0, 1.0), Complex64::new(4.0, -2.0)];
        let mut y = [Complex64::new(0.0, 0.0); 2];
        mv(Trans::No, Complex64::new(1.0, 0.0), &mat, &descr, &x, Complex64::new(0.0, 0.0), &mut y).unwrap();
        assert!((y[0].re - 3.0).abs() < 1e-12);
        assert!((y[0].im - 1.0).abs() < 1e-12);
        assert!((y[1].re - 4.0).abs() < 1e-12);
        assert!((y[1].im + 2.0).abs() < 1e-12);
    }

    #[test]
    fn complex_mv_identity_c32() {
        let val = [Complex32::new(1.0, 0.0), Complex32::new(1.0, 0.0)];
        let col: [sys::aoclsparse_int; 2] = [0, 1];
        let rp: [sys::aoclsparse_int; 3] = [0, 1, 2];
        let mat = ComplexSparseMatrix::<Complex32>::from_csr(
            IndexBase::Zero, 2, 2, &rp, &col, &val).unwrap();
        let descr = MatDescr::new().unwrap();
        let x = [Complex32::new(3.0, 1.0), Complex32::new(4.0, -2.0)];
        let mut y = [Complex32::new(0.0, 0.0); 2];
        mv(Trans::No, Complex32::new(1.0, 0.0), &mat, &descr,
           &x, Complex32::new(0.0, 0.0), &mut y).unwrap();
        assert!((y[0].re - 3.0).abs() < 1e-6);
        assert!((y[0].im - 1.0).abs() < 1e-6);
        assert!((y[1].re - 4.0).abs() < 1e-6);
        assert!((y[1].im + 2.0).abs() < 1e-6);
    }

    #[test]
    fn complex_csrmm_identity_c64() {
        // 2x2 identity sparse A, 2x2 dense B; C = A * B should equal B.
        let val = [Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)];
        let col: [sys::aoclsparse_int; 2] = [0, 1];
        let rp: [sys::aoclsparse_int; 3] = [0, 1, 2];
        let a = ComplexSparseMatrix::<Complex64>::from_csr(
            IndexBase::Zero, 2, 2, &rp, &col, &val).unwrap();
        let descr = MatDescr::new().unwrap();
        // 2x2 dense B (row-major, ldb = 2):
        // [(1, 1), (2, -1)]
        // [(3, 0), (4,  2)]
        let b = [
            Complex64::new(1.0, 1.0), Complex64::new(2.0, -1.0),
            Complex64::new(3.0, 0.0), Complex64::new(4.0,  2.0),
        ];
        let mut c = [Complex64::new(0.0, 0.0); 4];
        crate::complex::csrmm(
            Trans::No,
            Complex64::new(1.0, 0.0),
            &a, &descr, Order::RowMajor,
            &b, 2, 2,
            Complex64::new(0.0, 0.0),
            &mut c, 2,
        ).unwrap();
        for (got, want) in c.iter().zip(b.iter()) {
            assert!((got.re - want.re).abs() < 1e-12);
            assert!((got.im - want.im).abs() < 1e-12);
        }
    }

    #[test]
    fn complex_add_identity_plus_identity_c64() {
        let val = [Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)];
        let col: [sys::aoclsparse_int; 2] = [0, 1];
        let rp: [sys::aoclsparse_int; 3] = [0, 1, 2];
        let a = ComplexSparseMatrix::<Complex64>::from_csr(
            IndexBase::Zero, 2, 2, &rp, &col, &val).unwrap();
        let b = ComplexSparseMatrix::<Complex64>::from_csr(
            IndexBase::Zero, 2, 2, &rp, &col, &val).unwrap();
        let c = crate::complex::add(Trans::No, &a, Complex64::new(1.0, 0.0), &b).unwrap();
        let (_, _, _, val_c) = c.export_csr().unwrap();
        assert_eq!(val_c.len(), 2);
        for v in &val_c {
            assert!((v.re - 2.0).abs() < 1e-12);
            assert!(v.im.abs() < 1e-12);
        }
    }
}
