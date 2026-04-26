//! Hand-written FFI for AOCL-ScaLAPACK.
//!
//! AOCL ships no public C headers for ScaLAPACK; the declarations below
//! follow standard ScaLAPACK / BLACS conventions:
//!
//! - **BLACS C-API entry points** (`Cblacs_*`) take and return values via
//!   pointers but are exposed under their cdecl names with no trailing
//!   underscore.
//! - **ScaLAPACK Fortran routines** (`pdgesv`, `pdgemm`, etc.) follow the
//!   Fortran calling convention: trailing underscore in the symbol name,
//!   every argument by pointer, character arguments as `*const c_char`,
//!   integer arguments as `*const i32` (LP64) or `*const i64` (ILP64,
//!   not modelled here).
//! - **Distributed-matrix descriptors** are `[i32; 9]` arrays, populated
//!   by `descinit_`.
//!
//! At runtime the host program must be launched under an MPI runtime
//! (MS-MPI / MPICH / OpenMPI). MPI is **not** declared as a Cargo
//! dependency — link with the MPI library on your platform via your own
//! build script if you need additional MPI symbols.
//!
//! The signatures here cover the most-used distributed solve / BLAS-3
//! entry points across all four precisions; extend as needed.

#![allow(non_snake_case, non_camel_case_types, dead_code, clippy::all)]

use std::os::raw::{c_char, c_int};

// =========================================================================
//   BLACS lifecycle (C-API: prefixed `Cblacs_`, no trailing underscore)
// =========================================================================

extern "C" {
    /// Get this process's BLACS rank and total process count.
    /// Output: `*mypnum` and `*nprocs` are filled. If BLACS hasn't been
    /// initialised, on entry `*mypnum` should be -1; BLACS will
    /// initialise as a side effect.
    pub fn Cblacs_pinfo(mypnum: *mut c_int, nprocs: *mut c_int);

    /// Get a context value for the default system. `what = 0` returns
    /// the default system context in `*val`.
    pub fn Cblacs_get(icontxt: c_int, what: c_int, val: *mut c_int);

    /// Initialise an `nprow × npcol` BLACS process grid in `*context`.
    /// `order` is `b'R'` (row-major) or `b'C'` (column-major) for how
    /// MPI ranks are laid out across the grid.
    pub fn Cblacs_gridinit(context: *mut c_int, order: *const c_char,
                           nprow: c_int, npcol: c_int);

    /// Read back the grid shape and this process's coordinates.
    pub fn Cblacs_gridinfo(context: c_int, nprow: *mut c_int, npcol: *mut c_int,
                           myprow: *mut c_int, mypcol: *mut c_int);

    /// Release the grid context.
    pub fn Cblacs_gridexit(context: c_int);

    /// Tear down BLACS. Pass `continue_after = 0` for a normal shutdown.
    pub fn Cblacs_exit(continue_after: c_int);

    /// Synchronization barrier across the grid. `scope` is `b'A'` (all),
    /// `b'R'` (row), or `b'C'` (column).
    pub fn Cblacs_barrier(context: c_int, scope: *const c_char);
}

// =========================================================================
//   Distributed-matrix descriptor (Fortran)
// =========================================================================

/// Initialise a 9-element ScaLAPACK descriptor `desc` describing an
/// `m × n` block-cyclically distributed matrix with block size
/// `mb × nb`, root process at `(rsrc, csrc)`, attached to the BLACS
/// context `ctxt`, with local leading dimension `lld`. On exit `info`
/// is non-zero on error.
extern "C" {
    pub fn descinit_(
        desc: *mut c_int,
        m: *const c_int,
        n: *const c_int,
        mb: *const c_int,
        nb: *const c_int,
        rsrc: *const c_int,
        csrc: *const c_int,
        ictxt: *const c_int,
        lld: *const c_int,
        info: *mut c_int,
    );
}

/// Compute the local row count for a block-cyclically distributed array
/// with `n` global rows split into `nb`-row blocks across `nprocs`
/// processes, with this process's row coordinate `iproc` and root at
/// `isrcproc`.
extern "C" {
    pub fn numroc_(
        n: *const c_int,
        nb: *const c_int,
        iproc: *const c_int,
        isrcproc: *const c_int,
        nprocs: *const c_int,
    ) -> c_int;
}

// =========================================================================
//   ScaLAPACK Fortran routines (trailing underscore, all-by-pointer)
// =========================================================================

/// Distributed general solve `A·X = B` for double-precision `A`.
/// `n`, `nrhs` are global problem sizes. `a` and `b` are local
/// fragments described by `desca` and `descb`. `ipiv` holds local
/// pivots. On exit `info = 0` on success.
extern "C" {
    pub fn psgesv_(
        n: *const c_int, nrhs: *const c_int,
        a: *mut f32, ia: *const c_int, ja: *const c_int, desca: *const c_int,
        ipiv: *mut c_int,
        b: *mut f32, ib: *const c_int, jb: *const c_int, descb: *const c_int,
        info: *mut c_int,
    );

    pub fn pdgesv_(
        n: *const c_int, nrhs: *const c_int,
        a: *mut f64, ia: *const c_int, ja: *const c_int, desca: *const c_int,
        ipiv: *mut c_int,
        b: *mut f64, ib: *const c_int, jb: *const c_int, descb: *const c_int,
        info: *mut c_int,
    );

    /// Complex (single) variant. `a` and `b` are pointers to `[f32; 2]`
    /// pairs of `(re, im)`.
    pub fn pcgesv_(
        n: *const c_int, nrhs: *const c_int,
        a: *mut [f32; 2], ia: *const c_int, ja: *const c_int, desca: *const c_int,
        ipiv: *mut c_int,
        b: *mut [f32; 2], ib: *const c_int, jb: *const c_int, descb: *const c_int,
        info: *mut c_int,
    );

    pub fn pzgesv_(
        n: *const c_int, nrhs: *const c_int,
        a: *mut [f64; 2], ia: *const c_int, ja: *const c_int, desca: *const c_int,
        ipiv: *mut c_int,
        b: *mut [f64; 2], ib: *const c_int, jb: *const c_int, descb: *const c_int,
        info: *mut c_int,
    );
}

/// Distributed `C := α · op(A) · op(B) + β · C`. `transa`, `transb` are
/// 1-byte chars (`b'N'`, `b'T'`, `b'C'`).
extern "C" {
    pub fn psgemm_(
        transa: *const c_char, transb: *const c_char,
        m: *const c_int, n: *const c_int, k: *const c_int,
        alpha: *const f32,
        a: *const f32, ia: *const c_int, ja: *const c_int, desca: *const c_int,
        b: *const f32, ib: *const c_int, jb: *const c_int, descb: *const c_int,
        beta: *const f32,
        c: *mut f32, ic: *const c_int, jc: *const c_int, descc: *const c_int,
    );

    pub fn pdgemm_(
        transa: *const c_char, transb: *const c_char,
        m: *const c_int, n: *const c_int, k: *const c_int,
        alpha: *const f64,
        a: *const f64, ia: *const c_int, ja: *const c_int, desca: *const c_int,
        b: *const f64, ib: *const c_int, jb: *const c_int, descb: *const c_int,
        beta: *const f64,
        c: *mut f64, ic: *const c_int, jc: *const c_int, descc: *const c_int,
    );
}

/// Distributed Cholesky factorization (positive-definite).
extern "C" {
    pub fn pspotrf_(
        uplo: *const c_char, n: *const c_int,
        a: *mut f32, ia: *const c_int, ja: *const c_int, desca: *const c_int,
        info: *mut c_int,
    );

    pub fn pdpotrf_(
        uplo: *const c_char, n: *const c_int,
        a: *mut f64, ia: *const c_int, ja: *const c_int, desca: *const c_int,
        info: *mut c_int,
    );
}

/// Distributed LU factorization with partial pivoting.
extern "C" {
    pub fn psgetrf_(
        m: *const c_int, n: *const c_int,
        a: *mut f32, ia: *const c_int, ja: *const c_int, desca: *const c_int,
        ipiv: *mut c_int,
        info: *mut c_int,
    );

    pub fn pdgetrf_(
        m: *const c_int, n: *const c_int,
        a: *mut f64, ia: *const c_int, ja: *const c_int, desca: *const c_int,
        ipiv: *mut c_int,
        info: *mut c_int,
    );
}
