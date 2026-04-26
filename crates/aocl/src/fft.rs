//! Safe wrappers for AOCL-FFTW.
//!
//! Currently exposes a one-shot 1-D complex DFT. The function creates a
//! plan, executes it, and destroys it on each call — convenient and safe,
//! but you'll want to drop down to [`aocl_sys::fftw`] for repeated
//! transforms that benefit from cached plans.

use crate::error::{Error, Result};
use aocl_sys::fftw as sys;
use std::sync::Mutex;

/// FFTW's plan-creation and plan-destruction routines mutate global state
/// and are *not* internally thread-safe in the single-threaded library
/// variant we link against. We serialize them with a Rust-side mutex so
/// concurrent callers cannot race.
///
/// Note: `fftw_execute` *is* thread-safe on a constructed plan (provided
/// the plan was built with `FFTW_ESTIMATE` over the buffers being
/// executed) and is therefore not held under this lock.
static PLANNER_LOCK: Mutex<()> = Mutex::new(());

/// Direction of a 1-D DFT.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Direction {
    /// Forward transform (FFTW_FORWARD = -1).
    Forward,
    /// Backward / inverse transform (FFTW_BACKWARD = +1). Note that FFTW
    /// does **not** divide by `n` — call sites that want a strict inverse
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

/// Compute a 1-D complex DFT, overwriting `data` with the result.
///
/// `data` is treated as `n = data.len()` complex samples in `[real, imag]`
/// pairs. The transform is computed using `FFTW_ESTIMATE` (no plan
/// benchmarking, suitable for one-off use).
pub fn dft_1d_inplace(direction: Direction, data: &mut [[f64; 2]]) -> Result<()> {
    if data.is_empty() {
        return Ok(());
    }
    let n = data.len();
    if n > i32::MAX as usize {
        return Err(Error::InvalidArgument(format!(
            "dft_1d_inplace: n={n} exceeds i32::MAX"
        )));
    }
    // FFTW's `fftw_complex` is `[f64; 2]`; the layout matches our slice
    // exactly, so we can pass it through directly.
    //
    // SAFETY: `data` is a valid mutable slice; FFTW reads and writes only
    // its bounds. The plan we create here borrows the pointer for the
    // duration of `fftw_execute` and is destroyed before this function
    // returns, so the captured pointer never outlives `data`.
    let ptr = data.as_mut_ptr() as *mut sys::fftw_complex;
    let plan = {
        let _guard = PLANNER_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        unsafe {
            sys::fftw_plan_dft_1d(n as i32, ptr, ptr, direction.raw(), sys::FFTW_ESTIMATE)
        }
    };
    if plan.is_null() {
        return Err(Error::AllocationFailed("fftw"));
    }
    unsafe { sys::fftw_execute(plan) };
    let _guard = PLANNER_LOCK.lock().unwrap_or_else(|e| e.into_inner());
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dft_inverse_recovers_input() {
        // Pick a simple signal: [1+0i, 2+0i, 3+0i, 4+0i].
        let original: [[f64; 2]; 4] = [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]];
        let mut buf = original;
        forward_inplace(&mut buf).unwrap();
        backward_inplace(&mut buf).unwrap();
        // FFTW does not normalize; the round-trip multiplies by n=4.
        let n = original.len() as f64;
        for (got, orig) in buf.iter().zip(original.iter()) {
            assert!(
                (got[0] / n - orig[0]).abs() < 1e-12,
                "real mismatch: got {} vs orig {}",
                got[0] / n,
                orig[0]
            );
            assert!(
                (got[1] / n - orig[1]).abs() < 1e-12,
                "imag mismatch: got {} vs orig {}",
                got[1] / n,
                orig[1]
            );
        }
    }

    #[test]
    fn dc_signal_concentrates_at_dc() {
        // A constant real signal of value `c` and length `n` has DFT
        // [c·n, 0, 0, ..., 0].
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
}
