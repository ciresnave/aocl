//! Safe wrappers for AOCL-FFTW.
//!
//! Provides:
//! - One-shot 1-D / 2-D / 3-D complex DFTs (in-place).
//! - One-shot 1-D real-to-complex (`r2c`) and complex-to-real (`c2r`)
//!   transforms.
//! - A reusable [`Plan`] type that caches plan creation and supports
//!   the FFTW "new-array execute" routines for repeated transforms over
//!   the same shape.
//!
//! Plan creation/destruction is serialized through a process-wide
//! mutex because FFTW's planner is not internally thread-safe in the
//! single-threaded build we link against. Plan **execution** is
//! thread-safe and not held under that lock.

#![warn(missing_debug_implementations)]
#![cfg_attr(docsrs, feature(doc_cfg))]

use aocl_fft_sys as sys;
pub use aocl_error::{Error, Result};
use std::sync::Mutex;

/// FFTW's plan-creation and plan-destruction routines mutate global state.
static PLANNER_LOCK: Mutex<()> = Mutex::new(());

/// Direction of a complex DFT.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Direction {
    /// Forward transform (FFTW_FORWARD = -1).
    Forward,
    /// Backward / inverse transform (FFTW_BACKWARD = +1). FFTW does
    /// **not** divide by `n` — call sites that want a strict inverse
    /// must scale the output themselves.
    Backward,
}

impl Direction {
    fn raw(self) -> std::os::raw::c_int {
        match self {
            Direction::Forward => sys::FFTW_FORWARD,
            Direction::Backward => sys::FFTW_BACKWARD as i32,
        }
    }
}

fn check_n(name: &str, n: usize) -> Result<i32> {
    if n > i32::MAX as usize {
        return Err(Error::InvalidArgument(format!(
            "{name}: dimension {n} exceeds i32::MAX"
        )));
    }
    Ok(n as i32)
}

// =========================================================================
//   One-shot complex DFTs (1-D / 2-D / 3-D)
// =========================================================================

/// Compute a 1-D complex DFT in place. `data` is treated as `n = data.len()`
/// complex samples in `[real, imag]` order.
pub fn dft_1d_inplace(direction: Direction, data: &mut [[f64; 2]]) -> Result<()> {
    if data.is_empty() {
        return Ok(());
    }
    let n = check_n("dft_1d_inplace", data.len())?;
    let ptr = data.as_mut_ptr() as *mut sys::fftw_complex;
    let plan = {
        let _g = PLANNER_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        unsafe { sys::fftw_plan_dft_1d(n, ptr, ptr, direction.raw(), sys::FFTW_ESTIMATE) }
    };
    if plan.is_null() {
        return Err(Error::AllocationFailed("fft"));
    }
    unsafe { sys::fftw_execute(plan) };
    let _g = PLANNER_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    unsafe { sys::fftw_destroy_plan(plan) };
    Ok(())
}

/// Compute a 2-D complex DFT in place. `data` is `n0 × n1` complex samples
/// in row-major order (length `n0 · n1`).
pub fn dft_2d_inplace(
    direction: Direction,
    n0: usize,
    n1: usize,
    data: &mut [[f64; 2]],
) -> Result<()> {
    if n0 == 0 || n1 == 0 {
        return Ok(());
    }
    let need = n0 * n1;
    if data.len() < need {
        return Err(Error::InvalidArgument(format!(
            "dft_2d_inplace: data length {} < n0·n1 = {need}",
            data.len()
        )));
    }
    let n0 = check_n("dft_2d_inplace: n0", n0)?;
    let n1 = check_n("dft_2d_inplace: n1", n1)?;
    let ptr = data.as_mut_ptr() as *mut sys::fftw_complex;
    let plan = {
        let _g = PLANNER_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        unsafe { sys::fftw_plan_dft_2d(n0, n1, ptr, ptr, direction.raw(), sys::FFTW_ESTIMATE) }
    };
    if plan.is_null() {
        return Err(Error::AllocationFailed("fft"));
    }
    unsafe { sys::fftw_execute(plan) };
    let _g = PLANNER_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    unsafe { sys::fftw_destroy_plan(plan) };
    Ok(())
}

/// Compute a 3-D complex DFT in place.
pub fn dft_3d_inplace(
    direction: Direction,
    n0: usize,
    n1: usize,
    n2: usize,
    data: &mut [[f64; 2]],
) -> Result<()> {
    if n0 == 0 || n1 == 0 || n2 == 0 {
        return Ok(());
    }
    let need = n0 * n1 * n2;
    if data.len() < need {
        return Err(Error::InvalidArgument(format!(
            "dft_3d_inplace: data length {} < n0·n1·n2 = {need}",
            data.len()
        )));
    }
    let n0 = check_n("dft_3d_inplace: n0", n0)?;
    let n1 = check_n("dft_3d_inplace: n1", n1)?;
    let n2 = check_n("dft_3d_inplace: n2", n2)?;
    let ptr = data.as_mut_ptr() as *mut sys::fftw_complex;
    let plan = {
        let _g = PLANNER_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        unsafe {
            sys::fftw_plan_dft_3d(n0, n1, n2, ptr, ptr, direction.raw(), sys::FFTW_ESTIMATE)
        }
    };
    if plan.is_null() {
        return Err(Error::AllocationFailed("fft"));
    }
    unsafe { sys::fftw_execute(plan) };
    let _g = PLANNER_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    unsafe { sys::fftw_destroy_plan(plan) };
    Ok(())
}

/// Convenience: forward 1-D DFT in place.
pub fn forward_inplace(data: &mut [[f64; 2]]) -> Result<()> {
    dft_1d_inplace(Direction::Forward, data)
}

/// Convenience: backward (unscaled) 1-D DFT in place.
pub fn backward_inplace(data: &mut [[f64; 2]]) -> Result<()> {
    dft_1d_inplace(Direction::Backward, data)
}

// =========================================================================
//   Real ↔ complex one-shot (1-D)
// =========================================================================

/// Compute a 1-D real-to-complex (forward) DFT.
///
/// `out` must hold at least `n/2 + 1` complex samples (the second half of
/// the spectrum is implied by Hermitian symmetry).
pub fn r2c_1d(input: &mut [f64], output: &mut [[f64; 2]]) -> Result<()> {
    let n = input.len();
    if n == 0 {
        return Ok(());
    }
    let need_out = n / 2 + 1;
    if output.len() < need_out {
        return Err(Error::InvalidArgument(format!(
            "r2c_1d: output length {} < n/2+1 = {need_out}",
            output.len()
        )));
    }
    let n_i = check_n("r2c_1d", n)?;
    let in_p = input.as_mut_ptr();
    let out_p = output.as_mut_ptr() as *mut sys::fftw_complex;
    let plan = {
        let _g = PLANNER_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        unsafe { sys::fftw_plan_dft_r2c_1d(n_i, in_p, out_p, sys::FFTW_ESTIMATE) }
    };
    if plan.is_null() {
        return Err(Error::AllocationFailed("fft"));
    }
    unsafe { sys::fftw_execute(plan) };
    let _g = PLANNER_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    unsafe { sys::fftw_destroy_plan(plan) };
    Ok(())
}

/// Compute a 1-D complex-to-real (backward) DFT.
///
/// `n` is the size of the *output* (the real time-domain signal). `input`
/// must hold at least `n/2 + 1` complex samples. Note FFTW does **not**
/// normalize — divide by `n` to recover the original time-domain signal
/// after a forward+backward round-trip.
pub fn c2r_1d(n: usize, input: &mut [[f64; 2]], output: &mut [f64]) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    let need_in = n / 2 + 1;
    if input.len() < need_in {
        return Err(Error::InvalidArgument(format!(
            "c2r_1d: input length {} < n/2+1 = {need_in}",
            input.len()
        )));
    }
    if output.len() < n {
        return Err(Error::InvalidArgument(format!(
            "c2r_1d: output length {} < n = {n}",
            output.len()
        )));
    }
    let n_i = check_n("c2r_1d", n)?;
    let in_p = input.as_mut_ptr() as *mut sys::fftw_complex;
    let out_p = output.as_mut_ptr();
    let plan = {
        let _g = PLANNER_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        unsafe { sys::fftw_plan_dft_c2r_1d(n_i, in_p, out_p, sys::FFTW_ESTIMATE) }
    };
    if plan.is_null() {
        return Err(Error::AllocationFailed("fft"));
    }
    unsafe { sys::fftw_execute(plan) };
    let _g = PLANNER_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    unsafe { sys::fftw_destroy_plan(plan) };
    Ok(())
}

// =========================================================================
//   Reusable Plan
// =========================================================================

/// Kind of transform a [`Plan`] performs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum PlanKind {
    Complex,
    R2C,
    C2R,
}

/// A reusable FFTW plan over a fixed shape and transform kind.
///
/// Construct once with [`Plan::dft_1d`] / [`Plan::dft_2d`] / [`Plan::dft_3d`]
/// / [`Plan::r2c_1d`] / [`Plan::c2r_1d`], then call
/// [`Plan::execute_dft`] / [`Plan::execute_r2c`] / [`Plan::execute_c2r`]
/// repeatedly with new buffers of the matching shape. The buffers used at
/// execute time must have the same alignment as the buffers used at plan
/// creation; plain `Vec<…>` and `[…; N]` storage on x86_64 satisfy this.
pub struct Plan {
    plan: sys::fftw_plan,
    kind: PlanKind,
    /// Total number of complex elements (for dft) or real elements (for r2c/c2r).
    n_total: usize,
}

unsafe impl Send for Plan {}

impl Plan {
    /// Build a 1-D complex DFT plan over `n` samples in the given direction.
    pub fn dft_1d(n: usize, direction: Direction) -> Result<Self> {
        if n == 0 {
            return Err(Error::InvalidArgument("dft_1d: n must be positive".into()));
        }
        let n_i = check_n("dft_1d", n)?;
        // Create a scratch buffer for plan-time pointer requirements;
        // FFTW_ESTIMATE does not actually run the transform, so the
        // contents are irrelevant.
        let mut scratch = vec![[0.0_f64, 0.0_f64]; n];
        let p = scratch.as_mut_ptr() as *mut sys::fftw_complex;
        let plan = {
            let _g = PLANNER_LOCK.lock().unwrap_or_else(|e| e.into_inner());
            unsafe { sys::fftw_plan_dft_1d(n_i, p, p, direction.raw(), sys::FFTW_ESTIMATE) }
        };
        if plan.is_null() {
            return Err(Error::AllocationFailed("fft"));
        }
        Ok(Self { plan, kind: PlanKind::Complex, n_total: n })
    }

    /// Build a 2-D complex DFT plan over `n0 × n1` samples.
    pub fn dft_2d(n0: usize, n1: usize, direction: Direction) -> Result<Self> {
        if n0 == 0 || n1 == 0 {
            return Err(Error::InvalidArgument("dft_2d: dimensions must be positive".into()));
        }
        let n0_i = check_n("dft_2d: n0", n0)?;
        let n1_i = check_n("dft_2d: n1", n1)?;
        let mut scratch = vec![[0.0_f64, 0.0_f64]; n0 * n1];
        let p = scratch.as_mut_ptr() as *mut sys::fftw_complex;
        let plan = {
            let _g = PLANNER_LOCK.lock().unwrap_or_else(|e| e.into_inner());
            unsafe {
                sys::fftw_plan_dft_2d(n0_i, n1_i, p, p, direction.raw(), sys::FFTW_ESTIMATE)
            }
        };
        if plan.is_null() {
            return Err(Error::AllocationFailed("fft"));
        }
        Ok(Self { plan, kind: PlanKind::Complex, n_total: n0 * n1 })
    }

    /// Build a 3-D complex DFT plan.
    pub fn dft_3d(n0: usize, n1: usize, n2: usize, direction: Direction) -> Result<Self> {
        if n0 == 0 || n1 == 0 || n2 == 0 {
            return Err(Error::InvalidArgument("dft_3d: dimensions must be positive".into()));
        }
        let n0_i = check_n("dft_3d: n0", n0)?;
        let n1_i = check_n("dft_3d: n1", n1)?;
        let n2_i = check_n("dft_3d: n2", n2)?;
        let mut scratch = vec![[0.0_f64, 0.0_f64]; n0 * n1 * n2];
        let p = scratch.as_mut_ptr() as *mut sys::fftw_complex;
        let plan = {
            let _g = PLANNER_LOCK.lock().unwrap_or_else(|e| e.into_inner());
            unsafe {
                sys::fftw_plan_dft_3d(n0_i, n1_i, n2_i, p, p, direction.raw(), sys::FFTW_ESTIMATE)
            }
        };
        if plan.is_null() {
            return Err(Error::AllocationFailed("fft"));
        }
        Ok(Self { plan, kind: PlanKind::Complex, n_total: n0 * n1 * n2 })
    }

    /// Build a 1-D real-to-complex forward plan.
    pub fn r2c_1d(n: usize) -> Result<Self> {
        if n == 0 {
            return Err(Error::InvalidArgument("r2c_1d: n must be positive".into()));
        }
        let n_i = check_n("r2c_1d", n)?;
        let mut in_buf = vec![0.0_f64; n];
        let mut out_buf = vec![[0.0_f64, 0.0_f64]; n / 2 + 1];
        let plan = {
            let _g = PLANNER_LOCK.lock().unwrap_or_else(|e| e.into_inner());
            unsafe {
                sys::fftw_plan_dft_r2c_1d(
                    n_i,
                    in_buf.as_mut_ptr(),
                    out_buf.as_mut_ptr() as *mut sys::fftw_complex,
                    sys::FFTW_ESTIMATE,
                )
            }
        };
        if plan.is_null() {
            return Err(Error::AllocationFailed("fft"));
        }
        Ok(Self { plan, kind: PlanKind::R2C, n_total: n })
    }

    /// Build a 1-D complex-to-real backward plan.
    pub fn c2r_1d(n: usize) -> Result<Self> {
        if n == 0 {
            return Err(Error::InvalidArgument("c2r_1d: n must be positive".into()));
        }
        let n_i = check_n("c2r_1d", n)?;
        let mut in_buf = vec![[0.0_f64, 0.0_f64]; n / 2 + 1];
        let mut out_buf = vec![0.0_f64; n];
        let plan = {
            let _g = PLANNER_LOCK.lock().unwrap_or_else(|e| e.into_inner());
            unsafe {
                sys::fftw_plan_dft_c2r_1d(
                    n_i,
                    in_buf.as_mut_ptr() as *mut sys::fftw_complex,
                    out_buf.as_mut_ptr(),
                    sys::FFTW_ESTIMATE,
                )
            }
        };
        if plan.is_null() {
            return Err(Error::AllocationFailed("fft"));
        }
        Ok(Self { plan, kind: PlanKind::C2R, n_total: n })
    }

    /// Execute a complex DFT plan against new buffers of the matching shape.
    pub fn execute_dft(
        &self,
        input: &mut [[f64; 2]],
        output: &mut [[f64; 2]],
    ) -> Result<()> {
        if self.kind != PlanKind::Complex {
            return Err(Error::InvalidArgument(
                "execute_dft: plan is not a complex DFT".into(),
            ));
        }
        if input.len() < self.n_total || output.len() < self.n_total {
            return Err(Error::InvalidArgument(format!(
                "execute_dft: buffers ({}, {}) smaller than plan size {}",
                input.len(),
                output.len(),
                self.n_total
            )));
        }
        unsafe {
            sys::fftw_execute_dft(
                self.plan,
                input.as_mut_ptr() as *mut sys::fftw_complex,
                output.as_mut_ptr() as *mut sys::fftw_complex,
            );
        }
        Ok(())
    }

    /// Execute a real-to-complex plan against new buffers.
    pub fn execute_r2c(
        &self,
        input: &mut [f64],
        output: &mut [[f64; 2]],
    ) -> Result<()> {
        if self.kind != PlanKind::R2C {
            return Err(Error::InvalidArgument(
                "execute_r2c: plan is not an r2c plan".into(),
            ));
        }
        let need_out = self.n_total / 2 + 1;
        if input.len() < self.n_total || output.len() < need_out {
            return Err(Error::InvalidArgument(format!(
                "execute_r2c: input {} < n={}; output {} < n/2+1={}",
                input.len(),
                self.n_total,
                output.len(),
                need_out
            )));
        }
        unsafe {
            sys::fftw_execute_dft_r2c(
                self.plan,
                input.as_mut_ptr(),
                output.as_mut_ptr() as *mut sys::fftw_complex,
            );
        }
        Ok(())
    }

    /// Execute a complex-to-real plan against new buffers.
    pub fn execute_c2r(
        &self,
        input: &mut [[f64; 2]],
        output: &mut [f64],
    ) -> Result<()> {
        if self.kind != PlanKind::C2R {
            return Err(Error::InvalidArgument(
                "execute_c2r: plan is not a c2r plan".into(),
            ));
        }
        let need_in = self.n_total / 2 + 1;
        if input.len() < need_in || output.len() < self.n_total {
            return Err(Error::InvalidArgument(format!(
                "execute_c2r: input {} < n/2+1={}; output {} < n={}",
                input.len(),
                need_in,
                output.len(),
                self.n_total
            )));
        }
        unsafe {
            sys::fftw_execute_dft_c2r(
                self.plan,
                input.as_mut_ptr() as *mut sys::fftw_complex,
                output.as_mut_ptr(),
            );
        }
        Ok(())
    }
}

impl Drop for Plan {
    fn drop(&mut self) {
        if !self.plan.is_null() {
            let _g = PLANNER_LOCK.lock().unwrap_or_else(|e| e.into_inner());
            unsafe { sys::fftw_destroy_plan(self.plan) };
            self.plan = std::ptr::null_mut();
        }
    }
}

impl std::fmt::Debug for Plan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Plan")
            .field("kind", &self.kind)
            .field("n_total", &self.n_total)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dft_inverse_recovers_input() {
        let original: [[f64; 2]; 4] = [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]];
        let mut buf = original;
        forward_inplace(&mut buf).unwrap();
        backward_inplace(&mut buf).unwrap();
        let n = original.len() as f64;
        for (got, orig) in buf.iter().zip(original.iter()) {
            assert!((got[0] / n - orig[0]).abs() < 1e-12);
            assert!((got[1] / n - orig[1]).abs() < 1e-12);
        }
    }

    #[test]
    fn dc_signal_concentrates_at_dc() {
        let n = 8;
        let c = 1.5_f64;
        let mut buf: Vec<[f64; 2]> = (0..n).map(|_| [c, 0.0]).collect();
        forward_inplace(&mut buf).unwrap();
        let n_f = n as f64;
        assert!((buf[0][0] - c * n_f).abs() < 1e-12);
        assert!(buf[0][1].abs() < 1e-12);
        for v in &buf[1..] {
            assert!(v[0].abs() < 1e-12);
            assert!(v[1].abs() < 1e-12);
        }
    }

    #[test]
    fn empty_input_is_ok() {
        let mut empty: [[f64; 2]; 0] = [];
        forward_inplace(&mut empty).unwrap();
    }

    #[test]
    fn dft_2d_dc() {
        // 4×4 DC signal → spectrum has all energy at (0,0).
        let n0 = 4; let n1 = 4;
        let mut buf: Vec<[f64; 2]> = vec![[1.0, 0.0]; n0 * n1];
        dft_2d_inplace(Direction::Forward, n0, n1, &mut buf).unwrap();
        let total = (n0 * n1) as f64;
        assert!((buf[0][0] - total).abs() < 1e-9);
        assert!(buf[0][1].abs() < 1e-9);
        for v in &buf[1..] {
            assert!(v[0].abs() < 1e-9);
            assert!(v[1].abs() < 1e-9);
        }
    }

    #[test]
    fn r2c_then_c2r_round_trip() {
        let original = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let n = original.len();
        let mut input = original;
        let mut spectrum = vec![[0.0_f64, 0.0]; n / 2 + 1];
        r2c_1d(&mut input, &mut spectrum).unwrap();
        let mut recovered = vec![0.0_f64; n];
        c2r_1d(n, &mut spectrum, &mut recovered).unwrap();
        // FFTW does not normalize; round-trip multiplies by n.
        for (got, orig) in recovered.iter().zip(original.iter()) {
            assert!((got / n as f64 - orig).abs() < 1e-10);
        }
    }

    #[test]
    fn plan_executes_repeatedly() {
        let plan = Plan::dft_1d(4, Direction::Forward).unwrap();
        let mut a: Vec<[f64; 2]> = vec![[1.0, 0.0]; 4];
        let mut out_a = vec![[0.0, 0.0]; 4];
        plan.execute_dft(&mut a, &mut out_a).unwrap();
        assert!((out_a[0][0] - 4.0).abs() < 1e-9);

        let mut b: Vec<[f64; 2]> = vec![[2.0, 0.0]; 4];
        let mut out_b = vec![[0.0, 0.0]; 4];
        plan.execute_dft(&mut b, &mut out_b).unwrap();
        assert!((out_b[0][0] - 8.0).abs() < 1e-9);
    }

    #[test]
    fn plan_kind_mismatch_is_error() {
        let plan = Plan::r2c_1d(8).unwrap();
        let mut input: Vec<[f64; 2]> = vec![[0.0, 0.0]; 8];
        let mut output: Vec<[f64; 2]> = vec![[0.0, 0.0]; 8];
        let err = plan.execute_dft(&mut input, &mut output).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)));
    }
}
