//! AOCL-LibM — vectorized scalar math functions, AMD-tuned.
//!
//! Each function maps to a single `amd_*` entry point from `amdlibm.h`. The
//! `f32` variants forward to the corresponding `amd_*f` symbol.
//!
//! This module currently exposes only the scalar API. The vector API in
//! `amdlibm_vec.h` is gated by AMD as experimental (requires
//! `AMD_LIBM_VEC_EXPERIMENTAL`) and will be added behind a sub-feature
//! once stable.

use aocl_sys::libm as sys;

// All amd_* scalar math entry points are pure functions of their arguments
// with no side effects or pointer dereferences (except sincos*, which take
// out-pointers). We mark the wrappers `#[inline]` so this layer compiles to
// a direct call.

/// Compute e^x.
#[inline]
pub fn exp(x: f64) -> f64 {
    unsafe { sys::amd_exp(x) }
}
/// Compute e^x in single precision.
#[inline]
pub fn expf(x: f32) -> f32 {
    unsafe { sys::amd_expf(x) }
}

/// Compute the natural logarithm of `x`.
#[inline]
pub fn ln(x: f64) -> f64 {
    unsafe { sys::amd_log(x) }
}
/// Compute the natural logarithm of `x` in single precision.
#[inline]
pub fn lnf(x: f32) -> f32 {
    unsafe { sys::amd_logf(x) }
}

/// Compute `x` raised to the power `y`.
#[inline]
pub fn pow(x: f64, y: f64) -> f64 {
    unsafe { sys::amd_pow(x, y) }
}
/// Compute `x` raised to the power `y` in single precision.
#[inline]
pub fn powf(x: f32, y: f32) -> f32 {
    unsafe { sys::amd_powf(x, y) }
}

/// Compute sin(x) with `x` in radians.
#[inline]
pub fn sin(x: f64) -> f64 {
    unsafe { sys::amd_sin(x) }
}
/// Compute sin(x) in single precision.
#[inline]
pub fn sinf(x: f32) -> f32 {
    unsafe { sys::amd_sinf(x) }
}

/// Compute cos(x) with `x` in radians.
#[inline]
pub fn cos(x: f64) -> f64 {
    unsafe { sys::amd_cos(x) }
}
/// Compute cos(x) in single precision.
#[inline]
pub fn cosf(x: f32) -> f32 {
    unsafe { sys::amd_cosf(x) }
}

/// Compute tan(x) with `x` in radians.
#[inline]
pub fn tan(x: f64) -> f64 {
    unsafe { sys::amd_tan(x) }
}

/// Compute sin and cos simultaneously (faster than calling each separately).
#[inline]
pub fn sincos(x: f64) -> (f64, f64) {
    let mut s = 0.0_f64;
    let mut c = 0.0_f64;
    // SAFETY: We pass valid pointers into stack-allocated locals.
    unsafe { sys::amd_sincos(x, &mut s, &mut c) };
    (s, c)
}
/// Compute sin and cos simultaneously in single precision.
#[inline]
pub fn sincosf(x: f32) -> (f32, f32) {
    let mut s = 0.0_f32;
    let mut c = 0.0_f32;
    unsafe { sys::amd_sincosf(x, &mut s, &mut c) };
    (s, c)
}

/// Compute the principal square root of `x`.
#[inline]
pub fn sqrt(x: f64) -> f64 {
    unsafe { sys::amd_sqrt(x) }
}
/// Compute the principal square root in single precision.
#[inline]
pub fn sqrtf(x: f32) -> f32 {
    unsafe { sys::amd_sqrtf(x) }
}

/// Compute the cube root of `x`.
#[inline]
pub fn cbrt(x: f64) -> f64 {
    unsafe { sys::amd_cbrt(x) }
}

/// Compute sqrt(x*x + y*y) without intermediate overflow / underflow.
#[inline]
pub fn hypot(x: f64, y: f64) -> f64 {
    unsafe { sys::amd_hypot(x, y) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, eps: f64) {
        assert!((a - b).abs() < eps, "|{a} - {b}| > {eps}");
    }

    #[test]
    fn scalar_math_matches_libstd() {
        let eps = 1e-12;
        approx(exp(1.0), std::f64::consts::E, eps);
        approx(ln(std::f64::consts::E), 1.0, eps);
        approx(pow(2.0, 10.0), 1024.0, eps);
        approx(sqrt(2.0), std::f64::consts::SQRT_2, eps);
        approx(hypot(3.0, 4.0), 5.0, eps);

        // sincos consistency.
        let x = 0.7;
        let (s, c) = sincos(x);
        approx(s * s + c * c, 1.0, eps);
        approx(s, x.sin(), 1e-9);
        approx(c, x.cos(), 1e-9);
    }

    #[test]
    fn f32_variants_match() {
        let eps = 1e-5_f32;
        assert!((expf(1.0) - std::f32::consts::E).abs() < eps);
        assert!((sqrtf(2.0) - std::f32::consts::SQRT_2).abs() < eps);
    }
}
