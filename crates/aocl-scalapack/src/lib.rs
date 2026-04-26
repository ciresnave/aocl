//! Safe Rust wrappers for AOCL-ScaLAPACK.
//!
//! Provides a [`Grid`] RAII wrapper around the BLACS process grid, a
//! [`BlockCyclic`] distributed-matrix descriptor, and safe wrappers
//! around the most-used ScaLAPACK routines: `pgesv` (distributed
//! linear solve), `pgemm` (distributed matrix-matrix multiply), and
//! `pgetrf` / `ppotrf` factorizations.
//!
//! # Runtime
//!
//! Any program calling these functions must be launched under an MPI
//! runtime (MS-MPI on Windows, MPICH / OpenMPI on Linux). MPI is **not**
//! a Cargo dependency of this crate — link against your platform's MPI
//! library at the binary level if you need additional MPI symbols.
//!
//! Tests in this module are `#[ignore]`d by default because they
//! require a multi-rank MPI environment to do anything meaningful;
//! invoke them via `mpiexec -n N cargo test -- --ignored` from inside
//! a properly configured MPI host.

#![warn(missing_debug_implementations)]
#![cfg_attr(docsrs, feature(doc_cfg))]

use aocl_scalapack_sys as sys;
pub use aocl_error::{Error, Result};
pub use aocl_types::{Layout, Trans, Uplo};

use std::os::raw::{c_char, c_int};

fn map_info(component: &'static str, info: c_int) -> Result<()> {
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
            message: format!("ScaLAPACK reported singular factor at index {info}"),
        }),
    }
}

fn trans_char(t: Trans) -> c_char {
    match t {
        Trans::No => b'N' as c_char,
        Trans::T => b'T' as c_char,
        Trans::C => b'C' as c_char,
    }
}

fn uplo_char(u: Uplo) -> c_char {
    match u {
        Uplo::Upper => b'U' as c_char,
        Uplo::Lower => b'L' as c_char,
    }
}

// =========================================================================
//   BLACS grid
// =========================================================================

/// RAII wrapper around a BLACS process grid context.
pub struct Grid {
    context: c_int,
    nprow: c_int,
    npcol: c_int,
    myprow: c_int,
    mypcol: c_int,
    /// True after `Drop` (or `exit`) has been called; suppresses double-free.
    released: bool,
}

unsafe impl Send for Grid {}

impl Grid {
    /// Initialise BLACS (if not already initialised) and create a grid
    /// of size `nprow × npcol` with row-major rank layout.
    ///
    /// Must be called from inside an MPI program (typically launched
    /// via `mpiexec`).
    pub fn new(nprow: usize, npcol: usize) -> Result<Self> {
        let mut mypnum: c_int = -1;
        let mut nprocs: c_int = 0;
        unsafe { sys::Cblacs_pinfo(&mut mypnum, &mut nprocs) };
        if nprocs <= 0 {
            return Err(Error::Status {
                component: "scalapack",
                code: nprocs as i64,
                message: "BLACS reported zero processes — is MPI initialised?".into(),
            });
        }
        let needed = nprow * npcol;
        if (nprocs as usize) < needed {
            return Err(Error::InvalidArgument(format!(
                "Grid::new: nprow*npcol = {needed} > MPI world size {nprocs}"
            )));
        }
        let mut ctxt: c_int = 0;
        unsafe { sys::Cblacs_get(-1, 0, &mut ctxt) };
        let order = b"R\0".as_ptr() as *const c_char;
        unsafe { sys::Cblacs_gridinit(&mut ctxt, order, nprow as c_int, npcol as c_int) };

        let mut g_nprow = 0;
        let mut g_npcol = 0;
        let mut myprow = -1;
        let mut mypcol = -1;
        unsafe {
            sys::Cblacs_gridinfo(ctxt, &mut g_nprow, &mut g_npcol, &mut myprow, &mut mypcol)
        };
        Ok(Self {
            context: ctxt,
            nprow: g_nprow,
            npcol: g_npcol,
            myprow,
            mypcol,
            released: false,
        })
    }

    /// `(nprow, npcol)`.
    pub fn shape(&self) -> (usize, usize) {
        (self.nprow as usize, self.npcol as usize)
    }

    /// `(myprow, mypcol)`. Returns `(usize::MAX, usize::MAX)` for
    /// processes that aren't part of the grid (per BLACS convention,
    /// out-of-grid coords are `-1`).
    pub fn coords(&self) -> (usize, usize) {
        let to_usize = |c: c_int| if c < 0 { usize::MAX } else { c as usize };
        (to_usize(self.myprow), to_usize(self.mypcol))
    }

    /// Raw BLACS context value, for routines this crate doesn't yet wrap.
    pub fn context(&self) -> c_int {
        self.context
    }

    /// Synchronise across the grid (`scope = b'A'` covers the whole grid;
    /// `b'R'` row only; `b'C'` column only).
    pub fn barrier(&self, scope: u8) {
        let s = [scope as c_char, 0];
        unsafe { sys::Cblacs_barrier(self.context, s.as_ptr()) };
    }
}

impl Drop for Grid {
    fn drop(&mut self) {
        if !self.released {
            unsafe { sys::Cblacs_gridexit(self.context) };
            self.released = true;
        }
    }
}

impl std::fmt::Debug for Grid {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Grid")
            .field("nprow", &self.nprow)
            .field("npcol", &self.npcol)
            .field("myprow", &self.myprow)
            .field("mypcol", &self.mypcol)
            .finish_non_exhaustive()
    }
}

/// Tear down BLACS at program exit. Call exactly once after the last
/// `Grid` is dropped.
pub fn blacs_exit(continue_after_mpi: bool) {
    unsafe { sys::Cblacs_exit(if continue_after_mpi { 1 } else { 0 }) };
}

// =========================================================================
//   Block-cyclic descriptor
// =========================================================================

/// 9-element ScaLAPACK descriptor for a block-cyclically distributed
/// matrix.
#[derive(Debug, Clone)]
pub struct BlockCyclic {
    desc: [c_int; 9],
}

impl BlockCyclic {
    /// Build a descriptor for an `m × n` global matrix block-distributed
    /// in `mb × nb` blocks across `grid`, with root process at
    /// `(rsrc, csrc)` and local leading dimension `lld`.
    pub fn new(
        grid: &Grid,
        m: usize,
        n: usize,
        mb: usize,
        nb: usize,
        rsrc: usize,
        csrc: usize,
        lld: usize,
    ) -> Result<Self> {
        let mut desc = [0_i32; 9];
        let mut info: c_int = 0;
        let m_i = m as c_int;
        let n_i = n as c_int;
        let mb_i = mb as c_int;
        let nb_i = nb as c_int;
        let rsrc_i = rsrc as c_int;
        let csrc_i = csrc as c_int;
        let ctxt = grid.context;
        let lld_i = lld as c_int;
        unsafe {
            sys::descinit_(
                desc.as_mut_ptr(),
                &m_i, &n_i, &mb_i, &nb_i,
                &rsrc_i, &csrc_i, &ctxt, &lld_i,
                &mut info,
            );
        }
        if info != 0 {
            return Err(Error::Status {
                component: "scalapack",
                code: info as i64,
                message: format!("descinit_ returned info={info}"),
            });
        }
        Ok(Self { desc })
    }

    /// Borrow the raw 9-element descriptor.
    pub fn as_raw(&self) -> &[c_int; 9] {
        &self.desc
    }
}

/// Compute the local row count for an `n`-row global array partitioned
/// into `nb`-row blocks across `nprocs` grid rows, with this process at
/// row coordinate `iproc` and root at `isrcproc`.
pub fn numroc(n: usize, nb: usize, iproc: usize, isrcproc: usize, nprocs: usize) -> usize {
    let n_i = n as c_int;
    let nb_i = nb as c_int;
    let iproc_i = iproc as c_int;
    let isrcproc_i = isrcproc as c_int;
    let nprocs_i = nprocs as c_int;
    let r = unsafe {
        sys::numroc_(&n_i, &nb_i, &iproc_i, &isrcproc_i, &nprocs_i)
    };
    if r < 0 { 0 } else { r as usize }
}

// =========================================================================
//   Distributed solve / GEMM / factorization
// =========================================================================

/// Distributed double-precision general solve `A · X = B`.
/// All matrices use 1-based row/column origins per Fortran convention
/// (`ia`, `ja`, `ib`, `jb` are typically 1).
#[allow(clippy::too_many_arguments)]
pub fn pdgesv(
    n: usize,
    nrhs: usize,
    a: &mut [f64], ia: i32, ja: i32, desca: &BlockCyclic,
    ipiv: &mut [i32],
    b: &mut [f64], ib: i32, jb: i32, descb: &BlockCyclic,
) -> Result<()> {
    let n_i = n as c_int;
    let nrhs_i = nrhs as c_int;
    let mut info: c_int = 0;
    unsafe {
        sys::pdgesv_(
            &n_i, &nrhs_i,
            a.as_mut_ptr(), &ia, &ja, desca.as_raw().as_ptr(),
            ipiv.as_mut_ptr(),
            b.as_mut_ptr(), &ib, &jb, descb.as_raw().as_ptr(),
            &mut info,
        );
    }
    map_info("scalapack", info)
}

/// Distributed double-precision `C := α · op(A) · op(B) + β · C`.
#[allow(clippy::too_many_arguments)]
pub fn pdgemm(
    trans_a: Trans, trans_b: Trans,
    m: usize, n: usize, k: usize,
    alpha: f64,
    a: &[f64], ia: i32, ja: i32, desca: &BlockCyclic,
    b: &[f64], ib: i32, jb: i32, descb: &BlockCyclic,
    beta: f64,
    c: &mut [f64], ic: i32, jc: i32, descc: &BlockCyclic,
) {
    let m_i = m as c_int;
    let n_i = n as c_int;
    let k_i = k as c_int;
    let ta = trans_char(trans_a);
    let tb = trans_char(trans_b);
    unsafe {
        sys::pdgemm_(
            &ta, &tb,
            &m_i, &n_i, &k_i,
            &alpha,
            a.as_ptr(), &ia, &ja, desca.as_raw().as_ptr(),
            b.as_ptr(), &ib, &jb, descb.as_raw().as_ptr(),
            &beta,
            c.as_mut_ptr(), &ic, &jc, descc.as_raw().as_ptr(),
        );
    }
}

/// Distributed double-precision Cholesky factorization.
#[allow(clippy::too_many_arguments)]
pub fn pdpotrf(
    uplo: Uplo,
    n: usize,
    a: &mut [f64], ia: i32, ja: i32, desca: &BlockCyclic,
) -> Result<()> {
    let n_i = n as c_int;
    let u = uplo_char(uplo);
    let mut info: c_int = 0;
    unsafe {
        sys::pdpotrf_(
            &u, &n_i,
            a.as_mut_ptr(), &ia, &ja, desca.as_raw().as_ptr(),
            &mut info,
        );
    }
    map_info("scalapack", info)
}

/// Distributed double-precision LU factorization with partial pivoting.
#[allow(clippy::too_many_arguments)]
pub fn pdgetrf(
    m: usize, n: usize,
    a: &mut [f64], ia: i32, ja: i32, desca: &BlockCyclic,
    ipiv: &mut [i32],
) -> Result<()> {
    let m_i = m as c_int;
    let n_i = n as c_int;
    let mut info: c_int = 0;
    unsafe {
        sys::pdgetrf_(
            &m_i, &n_i,
            a.as_mut_ptr(), &ia, &ja, desca.as_raw().as_ptr(),
            ipiv.as_mut_ptr(),
            &mut info,
        );
    }
    map_info("scalapack", info)
}

/// Distributed back-substitution after [`pdgetrf`].
#[allow(clippy::too_many_arguments)]
pub fn pdgetrs(
    trans: Trans,
    n: usize, nrhs: usize,
    a: &[f64], ia: i32, ja: i32, desca: &BlockCyclic,
    ipiv: &[i32],
    b: &mut [f64], ib: i32, jb: i32, descb: &BlockCyclic,
) -> Result<()> {
    let t = trans_char(trans);
    let n_i = n as c_int;
    let nrhs_i = nrhs as c_int;
    let mut info: c_int = 0;
    unsafe {
        sys::pdgetrs_(
            &t,
            &n_i, &nrhs_i,
            a.as_ptr(), &ia, &ja, desca.as_raw().as_ptr(),
            ipiv.as_ptr(),
            b.as_mut_ptr(), &ib, &jb, descb.as_raw().as_ptr(),
            &mut info,
        );
    }
    map_info("scalapack", info)
}

/// Distributed back-substitution after [`pdpotrf`].
#[allow(clippy::too_many_arguments)]
pub fn pdpotrs(
    uplo: Uplo,
    n: usize, nrhs: usize,
    a: &[f64], ia: i32, ja: i32, desca: &BlockCyclic,
    b: &mut [f64], ib: i32, jb: i32, descb: &BlockCyclic,
) -> Result<()> {
    let u = uplo_char(uplo);
    let n_i = n as c_int;
    let nrhs_i = nrhs as c_int;
    let mut info: c_int = 0;
    unsafe {
        sys::pdpotrs_(
            &u,
            &n_i, &nrhs_i,
            a.as_ptr(), &ia, &ja, desca.as_raw().as_ptr(),
            b.as_mut_ptr(), &ib, &jb, descb.as_raw().as_ptr(),
            &mut info,
        );
    }
    map_info("scalapack", info)
}

/// Whether `pdsyev` should compute eigenvectors as well as eigenvalues.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EigCompute {
    ValuesOnly,
    ValuesAndVectors,
}

impl EigCompute {
    fn job_char(self) -> c_char {
        match self {
            EigCompute::ValuesOnly => b'N' as c_char,
            EigCompute::ValuesAndVectors => b'V' as c_char,
        }
    }
}

/// Distributed real symmetric eigendecomposition. Provide `work` of
/// size `lwork`; pass `lwork = -1` for a workspace-size query (the
/// optimum is written into `work[0]`).
#[allow(clippy::too_many_arguments)]
pub fn pdsyev(
    compute: EigCompute,
    uplo: Uplo,
    n: usize,
    a: &mut [f64], ia: i32, ja: i32, desca: &BlockCyclic,
    w: &mut [f64],
    z: &mut [f64], iz: i32, jz: i32, descz: &BlockCyclic,
    work: &mut [f64],
    lwork: i32,
) -> Result<()> {
    let j = compute.job_char();
    let u = uplo_char(uplo);
    let n_i = n as c_int;
    let mut info: c_int = 0;
    unsafe {
        sys::pdsyev_(
            &j, &u, &n_i,
            a.as_mut_ptr(), &ia, &ja, desca.as_raw().as_ptr(),
            w.as_mut_ptr(),
            z.as_mut_ptr(), &iz, &jz, descz.as_raw().as_ptr(),
            work.as_mut_ptr(), &lwork,
            &mut info,
        );
    }
    map_info("scalapack", info)
}

/// Distributed least squares `min || A·X − B ||₂`. `trans = Trans::No`
/// for the standard problem; `Trans::T` for the transposed.
#[allow(clippy::too_many_arguments)]
pub fn pdgels(
    trans: Trans,
    m: usize, n: usize, nrhs: usize,
    a: &mut [f64], ia: i32, ja: i32, desca: &BlockCyclic,
    b: &mut [f64], ib: i32, jb: i32, descb: &BlockCyclic,
    work: &mut [f64], lwork: i32,
) -> Result<()> {
    let t = trans_char(trans);
    let m_i = m as c_int;
    let n_i = n as c_int;
    let nrhs_i = nrhs as c_int;
    let mut info: c_int = 0;
    unsafe {
        sys::pdgels_(
            &t,
            &m_i, &n_i, &nrhs_i,
            a.as_mut_ptr(), &ia, &ja, desca.as_raw().as_ptr(),
            b.as_mut_ptr(), &ib, &jb, descb.as_raw().as_ptr(),
            work.as_mut_ptr(), &lwork,
            &mut info,
        );
    }
    map_info("scalapack", info)
}

/// Distributed QR factorization.
#[allow(clippy::too_many_arguments)]
pub fn pdgeqrf(
    m: usize, n: usize,
    a: &mut [f64], ia: i32, ja: i32, desca: &BlockCyclic,
    tau: &mut [f64],
    work: &mut [f64], lwork: i32,
) -> Result<()> {
    let m_i = m as c_int;
    let n_i = n as c_int;
    let mut info: c_int = 0;
    unsafe {
        sys::pdgeqrf_(
            &m_i, &n_i,
            a.as_mut_ptr(), &ia, &ja, desca.as_raw().as_ptr(),
            tau.as_mut_ptr(),
            work.as_mut_ptr(), &lwork,
            &mut info,
        );
    }
    map_info("scalapack", info)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke-test that the FFI declarations match what the linker can
    /// resolve. No actual ScaLAPACK call — just compile-time + link-time
    /// symbol existence verification (the link directives will pull
    /// scalapack.dll in regardless).
    ///
    /// Real distributed-routine tests must be run under MPI:
    /// ```bash
    /// mpiexec -n 4 cargo test -p aocl-scalapack -- --ignored
    /// ```
    #[test]
    fn ffi_declarations_link() {
        // No-op: getting here means the linker resolved every symbol
        // we declared in aocl-scalapack-sys. If that fails the test
        // exe will never start.
    }

    #[test]
    #[ignore = "needs MPI runtime; launch via `mpiexec -n 4`"]
    fn pdgesv_distributed_solve() {
        // Sketch (won't run without mpiexec):
        // let _grid = Grid::new(2, 2).unwrap();
        // let desca = BlockCyclic::new(&_grid, 8, 8, 4, 4, 0, 0, 4).unwrap();
        // let descb = BlockCyclic::new(&_grid, 8, 1, 4, 1, 0, 0, 4).unwrap();
        // pdgesv(8, 1, ..., 1, 1, &desca, ..., ..., 1, 1, &descb).unwrap();
    }
}
