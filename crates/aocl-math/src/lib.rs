//! Safe Rust wrappers for AOCL-LibM — comprehensive scalar math, AMD-tuned.
//!
//! Covers the entire AOCL-LibM scalar API: trig + inverse trig + atan2;
//! hyperbolic + inverse hyperbolic; exp / log family (exp, exp2, exp10,
//! expm1, log, log2, log10, log1p, logb, ilogb); power family (pow, sqrt,
//! cbrt, hypot); floor/ceil/round/trunc/rint/nearbyint and their integer
//! variants (lrint, llrint, lround, llround); fmod / remainder / remquo;
//! frexp / ldexp / scalbn / scalbln / modf; nextafter / copysign /
//! fdim / fmin / fmax; finite check; erf; complex cexp / cpow / clog
//! (via aocl_types::Complex32 / Complex64).
//!
//! Each `f32` entry point forwards to the corresponding `amd_*f` symbol;
//! complex routines accept and return our [`aocl_types::Complex32`] /
//! [`aocl_types::Complex64`] (ABI-equivalent to AOCL's `fc32_t` / `fc64_t`
//! storage layout `{ _Val: [f, 2] }`).

#![warn(missing_debug_implementations)]
#![cfg_attr(docsrs, feature(doc_cfg))]

pub use aocl_error::{Error, Result};
use aocl_math_sys as sys;
pub use aocl_types::{Complex32, Complex64};

use std::ffi::CString;

// All amd_* scalar entry points are pure functions of their arguments
// (the few with out-pointers — sincos, modf, frexp, remquo — read/write
// only what they're explicitly handed). We mark wrappers `#[inline]` so
// they compile to a direct call.

// =========================================================================
//   Trigonometric
// =========================================================================

/// Cosine, `x` in radians.
#[inline]
pub fn cos(x: f64) -> f64 {
    unsafe { sys::amd_cos(x) }
}
/// Cosine in single precision.
#[inline]
pub fn cosf(x: f32) -> f32 {
    unsafe { sys::amd_cosf(x) }
}

/// Sine, `x` in radians.
#[inline]
pub fn sin(x: f64) -> f64 {
    unsafe { sys::amd_sin(x) }
}
/// Sine in single precision.
#[inline]
pub fn sinf(x: f32) -> f32 {
    unsafe { sys::amd_sinf(x) }
}

/// Tangent, `x` in radians.
#[inline]
pub fn tan(x: f64) -> f64 {
    unsafe { sys::amd_tan(x) }
}
/// Tangent in single precision.
#[inline]
pub fn tanf(x: f32) -> f32 {
    unsafe { sys::amd_tanf(x) }
}

/// Arc cosine.
#[inline]
pub fn acos(x: f64) -> f64 {
    unsafe { sys::amd_acos(x) }
}
#[inline]
pub fn acosf(x: f32) -> f32 {
    unsafe { sys::amd_acosf(x) }
}

/// Arc sine.
#[inline]
pub fn asin(x: f64) -> f64 {
    unsafe { sys::amd_asin(x) }
}
#[inline]
pub fn asinf(x: f32) -> f32 {
    unsafe { sys::amd_asinf(x) }
}

/// Arc tangent.
#[inline]
pub fn atan(x: f64) -> f64 {
    unsafe { sys::amd_atan(x) }
}
#[inline]
pub fn atanf(x: f32) -> f32 {
    unsafe { sys::amd_atanf(x) }
}

/// Two-argument arc tangent.
#[inline]
pub fn atan2(y: f64, x: f64) -> f64 {
    unsafe { sys::amd_atan2(y, x) }
}
#[inline]
pub fn atan2f(y: f32, x: f32) -> f32 {
    unsafe { sys::amd_atan2f(y, x) }
}

/// Compute sin and cos simultaneously.
#[inline]
pub fn sincos(x: f64) -> (f64, f64) {
    let mut s = 0.0_f64;
    let mut c = 0.0_f64;
    unsafe { sys::amd_sincos(x, &mut s, &mut c) };
    (s, c)
}
#[inline]
pub fn sincosf(x: f32) -> (f32, f32) {
    let mut s = 0.0_f32;
    let mut c = 0.0_f32;
    unsafe { sys::amd_sincosf(x, &mut s, &mut c) };
    (s, c)
}

// =========================================================================
//   Hyperbolic
// =========================================================================

#[inline]
pub fn cosh(x: f64) -> f64 {
    unsafe { sys::amd_cosh(x) }
}
#[inline]
pub fn coshf(x: f32) -> f32 {
    unsafe { sys::amd_coshf(x) }
}
#[inline]
pub fn sinh(x: f64) -> f64 {
    unsafe { sys::amd_sinh(x) }
}
#[inline]
pub fn sinhf(x: f32) -> f32 {
    unsafe { sys::amd_sinhf(x) }
}
#[inline]
pub fn tanh(x: f64) -> f64 {
    unsafe { sys::amd_tanh(x) }
}
#[inline]
pub fn tanhf(x: f32) -> f32 {
    unsafe { sys::amd_tanhf(x) }
}
#[inline]
pub fn acosh(x: f64) -> f64 {
    unsafe { sys::amd_acosh(x) }
}
#[inline]
pub fn acoshf(x: f32) -> f32 {
    unsafe { sys::amd_acoshf(x) }
}
#[inline]
pub fn asinh(x: f64) -> f64 {
    unsafe { sys::amd_asinh(x) }
}
#[inline]
pub fn asinhf(x: f32) -> f32 {
    unsafe { sys::amd_asinhf(x) }
}
#[inline]
pub fn atanh(x: f64) -> f64 {
    unsafe { sys::amd_atanh(x) }
}
#[inline]
pub fn atanhf(x: f32) -> f32 {
    unsafe { sys::amd_atanhf(x) }
}

// =========================================================================
//   Exponential / logarithm
// =========================================================================

/// `eˣ`.
#[inline]
pub fn exp(x: f64) -> f64 {
    unsafe { sys::amd_exp(x) }
}
#[inline]
pub fn expf(x: f32) -> f32 {
    unsafe { sys::amd_expf(x) }
}
/// `2ˣ`.
#[inline]
pub fn exp2(x: f64) -> f64 {
    unsafe { sys::amd_exp2(x) }
}
#[inline]
pub fn exp2f(x: f32) -> f32 {
    unsafe { sys::amd_exp2f(x) }
}
/// `10ˣ`.
#[inline]
pub fn exp10(x: f64) -> f64 {
    unsafe { sys::amd_exp10(x) }
}
#[inline]
pub fn exp10f(x: f32) -> f32 {
    unsafe { sys::amd_exp10f(x) }
}
/// `eˣ - 1` (accurate for small x).
#[inline]
pub fn expm1(x: f64) -> f64 {
    unsafe { sys::amd_expm1(x) }
}
#[inline]
pub fn expm1f(x: f32) -> f32 {
    unsafe { sys::amd_expm1f(x) }
}

/// Natural logarithm.
#[inline]
pub fn ln(x: f64) -> f64 {
    unsafe { sys::amd_log(x) }
}
#[inline]
pub fn lnf(x: f32) -> f32 {
    unsafe { sys::amd_logf(x) }
}
/// Log base 2.
#[inline]
pub fn log2(x: f64) -> f64 {
    unsafe { sys::amd_log2(x) }
}
#[inline]
pub fn log2f(x: f32) -> f32 {
    unsafe { sys::amd_log2f(x) }
}
/// Log base 10.
#[inline]
pub fn log10(x: f64) -> f64 {
    unsafe { sys::amd_log10(x) }
}
#[inline]
pub fn log10f(x: f32) -> f32 {
    unsafe { sys::amd_log10f(x) }
}
/// `ln(1 + x)` (accurate for small x).
#[inline]
pub fn ln_1p(x: f64) -> f64 {
    unsafe { sys::amd_log1p(x) }
}
#[inline]
pub fn ln_1pf(x: f32) -> f32 {
    unsafe { sys::amd_log1pf(x) }
}
/// Unbiased exponent of `x`.
#[inline]
pub fn logb(x: f64) -> f64 {
    unsafe { sys::amd_logb(x) }
}
#[inline]
pub fn logbf(x: f32) -> f32 {
    unsafe { sys::amd_logbf(x) }
}
/// Unbiased exponent of `x` as an integer.
#[inline]
pub fn ilogb(x: f64) -> i32 {
    unsafe { sys::amd_ilogb(x) }
}
#[inline]
pub fn ilogbf(x: f32) -> i32 {
    unsafe { sys::amd_ilogbf(x) }
}

// =========================================================================
//   Power / root
// =========================================================================

/// `x^y`.
#[inline]
pub fn pow(x: f64, y: f64) -> f64 {
    unsafe { sys::amd_pow(x, y) }
}
#[inline]
pub fn powf(x: f32, y: f32) -> f32 {
    unsafe { sys::amd_powf(x, y) }
}
/// √x.
#[inline]
pub fn sqrt(x: f64) -> f64 {
    unsafe { sys::amd_sqrt(x) }
}
#[inline]
pub fn sqrtf(x: f32) -> f32 {
    unsafe { sys::amd_sqrtf(x) }
}
/// ∛x.
#[inline]
pub fn cbrt(x: f64) -> f64 {
    unsafe { sys::amd_cbrt(x) }
}
#[inline]
pub fn cbrtf(x: f32) -> f32 {
    unsafe { sys::amd_cbrtf(x) }
}
/// √(x² + y²) without overflow.
#[inline]
pub fn hypot(x: f64, y: f64) -> f64 {
    unsafe { sys::amd_hypot(x, y) }
}
#[inline]
pub fn hypotf(x: f32, y: f32) -> f32 {
    unsafe { sys::amd_hypotf(x, y) }
}

// =========================================================================
//   Absolute value, rounding, integer conversions
// =========================================================================

#[inline]
pub fn fabs(x: f64) -> f64 {
    unsafe { sys::amd_fabs(x) }
}
#[inline]
pub fn fabsf(x: f32) -> f32 {
    unsafe { sys::amd_fabsf(x) }
}
#[inline]
pub fn ceil(x: f64) -> f64 {
    unsafe { sys::amd_ceil(x) }
}
#[inline]
pub fn ceilf(x: f32) -> f32 {
    unsafe { sys::amd_ceilf(x) }
}
#[inline]
pub fn floor(x: f64) -> f64 {
    unsafe { sys::amd_floor(x) }
}
#[inline]
pub fn floorf(x: f32) -> f32 {
    unsafe { sys::amd_floorf(x) }
}
#[inline]
pub fn trunc(x: f64) -> f64 {
    unsafe { sys::amd_trunc(x) }
}
#[inline]
pub fn truncf(x: f32) -> f32 {
    unsafe { sys::amd_truncf(x) }
}
/// Round to nearest integer using current rounding mode (no inexact flag).
#[inline]
pub fn nearbyint(x: f64) -> f64 {
    unsafe { sys::amd_nearbyint(x) }
}
#[inline]
pub fn nearbyintf(x: f32) -> f32 {
    unsafe { sys::amd_nearbyintf(x) }
}
/// Round to nearest integer using current rounding mode (may set inexact).
#[inline]
pub fn rint(x: f64) -> f64 {
    unsafe { sys::amd_rint(x) }
}
#[inline]
pub fn rintf(x: f32) -> f32 {
    unsafe { sys::amd_rintf(x) }
}
/// Round to nearest integer (ties away from zero).
#[inline]
pub fn round(x: f64) -> f64 {
    unsafe { sys::amd_round(x) }
}
#[inline]
pub fn roundf(x: f32) -> f32 {
    unsafe { sys::amd_roundf(x) }
}

/// Round to nearest `i64`-sized integer.
#[inline]
pub fn lrint(x: f64) -> i64 {
    unsafe { sys::amd_lrint(x) as i64 }
}
#[inline]
pub fn lrintf(x: f32) -> i64 {
    unsafe { sys::amd_lrintf(x) as i64 }
}
#[inline]
pub fn llrint(x: f64) -> i64 {
    unsafe { sys::amd_llrint(x) as i64 }
}
#[inline]
pub fn llrintf(x: f32) -> i64 {
    unsafe { sys::amd_llrintf(x) as i64 }
}
#[inline]
pub fn lround(x: f64) -> i64 {
    unsafe { sys::amd_lround(x) as i64 }
}
#[inline]
pub fn lroundf(x: f32) -> i64 {
    unsafe { sys::amd_lroundf(x) as i64 }
}
#[inline]
pub fn llround(x: f64) -> i64 {
    unsafe { sys::amd_llround(x) as i64 }
}
#[inline]
pub fn llroundf(x: f32) -> i64 {
    unsafe { sys::amd_llroundf(x) as i64 }
}

// =========================================================================
//   Modulo / remainder
// =========================================================================

/// `x mod y` with sign matching `x`.
#[inline]
pub fn fmod(x: f64, y: f64) -> f64 {
    unsafe { sys::amd_fmod(x, y) }
}
#[inline]
pub fn fmodf(x: f32, y: f32) -> f32 {
    unsafe { sys::amd_fmodf(x, y) }
}
/// IEEE remainder.
#[inline]
pub fn remainder(x: f64, y: f64) -> f64 {
    unsafe { sys::amd_remainder(x, y) }
}
#[inline]
pub fn remainderf(x: f32, y: f32) -> f32 {
    unsafe { sys::amd_remainderf(x, y) }
}
/// IEEE remainder + low bits of quotient.
#[inline]
pub fn remquo(x: f64, y: f64) -> (f64, i32) {
    let mut q: i32 = 0;
    let r = unsafe { sys::amd_remquo(x, y, &mut q) };
    (r, q)
}
#[inline]
pub fn remquof(x: f32, y: f32) -> (f32, i32) {
    let mut q: i32 = 0;
    let r = unsafe { sys::amd_remquof(x, y, &mut q) };
    (r, q)
}

// =========================================================================
//   Decompose / scale
// =========================================================================

/// Split `x` into integer and fractional parts. Returns `(fractional, integer)`.
#[inline]
pub fn modf(x: f64) -> (f64, f64) {
    let mut i = 0.0_f64;
    let frac = unsafe { sys::amd_modf(x, &mut i) };
    (frac, i)
}
#[inline]
pub fn modff(x: f32) -> (f32, f32) {
    let mut i = 0.0_f32;
    let frac = unsafe { sys::amd_modff(x, &mut i) };
    (frac, i)
}

/// Decompose `x = m · 2ᵉ` with `m ∈ [0.5, 1.0)` (or 0). Returns `(m, e)`.
#[inline]
pub fn frexp(x: f64) -> (f64, i32) {
    let mut e: i32 = 0;
    let m = unsafe { sys::amd_frexp(x, &mut e) };
    (m, e)
}
#[inline]
pub fn frexpf(x: f32) -> (f32, i32) {
    let mut e: i32 = 0;
    let m = unsafe { sys::amd_frexpf(x, &mut e) };
    (m, e)
}

/// `x · 2ⁿ`.
#[inline]
pub fn ldexp(x: f64, n: i32) -> f64 {
    unsafe { sys::amd_ldexp(x, n) }
}
#[inline]
pub fn ldexpf(x: f32, n: i32) -> f32 {
    unsafe { sys::amd_ldexpf(x, n) }
}
#[inline]
pub fn scalbn(x: f64, n: i32) -> f64 {
    unsafe { sys::amd_scalbn(x, n) }
}
#[inline]
pub fn scalbnf(x: f32, n: i32) -> f32 {
    unsafe { sys::amd_scalbnf(x, n) }
}
#[inline]
pub fn scalbln(x: f64, n: i64) -> f64 {
    unsafe { sys::amd_scalbln(x, n as std::os::raw::c_long) }
}
#[inline]
pub fn scalblnf(x: f32, n: i64) -> f32 {
    unsafe { sys::amd_scalblnf(x, n as std::os::raw::c_long) }
}

// =========================================================================
//   Float ops (sign manipulation, comparison, etc.)
// =========================================================================

#[inline]
pub fn copysign(magnitude: f64, sign: f64) -> f64 {
    unsafe { sys::amd_copysign(magnitude, sign) }
}
#[inline]
pub fn copysignf(magnitude: f32, sign: f32) -> f32 {
    unsafe { sys::amd_copysignf(magnitude, sign) }
}

/// Next representable value after `x` toward `y`.
#[inline]
pub fn nextafter(x: f64, y: f64) -> f64 {
    unsafe { sys::amd_nextafter(x, y) }
}
#[inline]
pub fn nextafterf(x: f32, y: f32) -> f32 {
    unsafe { sys::amd_nextafterf(x, y) }
}
/// `nexttoward` — `y` is f64 even for f32 input.
#[inline]
pub fn nexttoward(x: f64, y: f64) -> f64 {
    unsafe { sys::amd_nexttoward(x, y) }
}
#[inline]
pub fn nexttowardf(x: f32, y: f64) -> f32 {
    unsafe { sys::amd_nexttowardf(x, y) }
}

/// `max(x − y, 0)`.
#[inline]
pub fn fdim(x: f64, y: f64) -> f64 {
    unsafe { sys::amd_fdim(x, y) }
}
#[inline]
pub fn fdimf(x: f32, y: f32) -> f32 {
    unsafe { sys::amd_fdimf(x, y) }
}
#[inline]
pub fn fmax(x: f64, y: f64) -> f64 {
    unsafe { sys::amd_fmax(x, y) }
}
#[inline]
pub fn fmaxf(x: f32, y: f32) -> f32 {
    unsafe { sys::amd_fmaxf(x, y) }
}
#[inline]
pub fn fmin(x: f64, y: f64) -> f64 {
    unsafe { sys::amd_fmin(x, y) }
}
#[inline]
pub fn fminf(x: f32, y: f32) -> f32 {
    unsafe { sys::amd_fminf(x, y) }
}

/// Returns `true` if `x` is finite (neither infinite nor NaN).
#[inline]
pub fn finite(x: f64) -> bool {
    unsafe { sys::amd_finite(x) != 0 }
}
#[inline]
pub fn finitef(x: f32) -> bool {
    unsafe { sys::amd_finitef(x) != 0 }
}

/// Construct a NaN value, optionally with a payload tag.
pub fn nan(tag: &str) -> Result<f64> {
    let cs = CString::new(tag)
        .map_err(|e| Error::InvalidArgument(format!("nan: tag contained interior NUL: {e}")))?;
    Ok(unsafe { sys::amd_nan(cs.as_ptr()) })
}
pub fn nanf(tag: &str) -> Result<f32> {
    let cs = CString::new(tag)
        .map_err(|e| Error::InvalidArgument(format!("nanf: tag contained interior NUL: {e}")))?;
    Ok(unsafe { sys::amd_nanf(cs.as_ptr()) })
}

/// Error function `erf(x) = (2/√π) ∫₀ˣ e^(-t²) dt`.
#[inline]
pub fn erf(x: f64) -> f64 {
    unsafe { sys::amd_erf(x) }
}
#[inline]
pub fn erff(x: f32) -> f32 {
    unsafe { sys::amd_erff(x) }
}

// =========================================================================
//   Complex math (cexp, cpow, clog)
// =========================================================================

// AOCL-LibM's fc32_t/fc64_t are MSVC's `_Fcomplex` / `_Dcomplex`, which are
// repr(C) structs `{ _Val: [f32; 2] }` / `{ _Val: [f64; 2] }`. Our
// Complex32/Complex64 are repr(C) `{ re: f, im: f }`. Same byte layout
// — we transmute via raw cast at the boundary.

#[inline]
fn to_fc32(c: Complex32) -> sys::fc32_t {
    sys::fc32_t { _Val: [c.re, c.im] }
}
#[inline]
fn from_fc32(c: sys::fc32_t) -> Complex32 {
    Complex32::new(c._Val[0], c._Val[1])
}
#[inline]
fn to_fc64(c: Complex64) -> sys::fc64_t {
    sys::fc64_t { _Val: [c.re, c.im] }
}
#[inline]
fn from_fc64(c: sys::fc64_t) -> Complex64 {
    Complex64::new(c._Val[0], c._Val[1])
}

/// Complex exponential `e^z`.
#[inline]
pub fn cexp(z: Complex64) -> Complex64 {
    from_fc64(unsafe { sys::amd_cexp(to_fc64(z)) })
}
#[inline]
pub fn cexpf(z: Complex32) -> Complex32 {
    from_fc32(unsafe { sys::amd_cexpf(to_fc32(z)) })
}

/// Complex power `z^w`.
#[inline]
pub fn cpow(z: Complex64, w: Complex64) -> Complex64 {
    from_fc64(unsafe { sys::amd_cpow(to_fc64(z), to_fc64(w)) })
}
#[inline]
pub fn cpowf(z: Complex32, w: Complex32) -> Complex32 {
    from_fc32(unsafe { sys::amd_cpowf(to_fc32(z), to_fc32(w)) })
}

/// Complex natural logarithm.
#[inline]
pub fn cln(z: Complex64) -> Complex64 {
    from_fc64(unsafe { sys::amd_clog(to_fc64(z)) })
}
#[inline]
pub fn clnf(z: Complex32) -> Complex32 {
    from_fc32(unsafe { sys::amd_clogf(to_fc32(z)) })
}

// =========================================================================
//   Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, eps: f64) {
        assert!((a - b).abs() < eps, "|{a} - {b}| > {eps}");
    }

    #[test]
    fn trig_round_trips() {
        approx(asin(sin(0.7)), 0.7, 1e-12);
        approx(acos(cos(0.7)), 0.7, 1e-12);
        approx(atan(tan(0.5)), 0.5, 1e-12);
        approx(atan2(1.0, 1.0), std::f64::consts::FRAC_PI_4, 1e-12);
        let (s, c) = sincos(0.7);
        approx(s * s + c * c, 1.0, 1e-12);
    }

    #[test]
    fn hyperbolic_consistency() {
        let x = 0.5_f64;
        approx(asinh(sinh(x)), x, 1e-12);
        approx(acosh(cosh(x)), x, 1e-12);
        approx(atanh(tanh(x)), x, 1e-12);
        approx(cosh(x) * cosh(x) - sinh(x) * sinh(x), 1.0, 1e-12);
    }

    #[test]
    fn exp_log_family() {
        approx(exp(1.0), std::f64::consts::E, 1e-12);
        approx(exp2(10.0), 1024.0, 1e-12);
        approx(exp10(3.0), 1000.0, 1e-12);
        approx(expm1(0.0), 0.0, 1e-12);
        approx(ln(std::f64::consts::E), 1.0, 1e-12);
        approx(log2(1024.0), 10.0, 1e-12);
        approx(log10(1000.0), 3.0, 1e-12);
        approx(ln_1p(0.0), 0.0, 1e-12);
        approx(logb(8.0), 3.0, 1e-12);
        assert_eq!(ilogb(8.0), 3);
    }

    #[test]
    fn power_root_family() {
        approx(pow(2.0, 10.0), 1024.0, 1e-12);
        approx(sqrt(2.0), std::f64::consts::SQRT_2, 1e-12);
        approx(cbrt(27.0), 3.0, 1e-12);
        approx(hypot(3.0, 4.0), 5.0, 1e-12);
    }

    #[test]
    fn rounding_family() {
        approx(fabs(-3.0), 3.0, 1e-12);
        approx(ceil(2.3), 3.0, 1e-12);
        approx(floor(2.7), 2.0, 1e-12);
        approx(trunc(-2.7), -2.0, 1e-12);
        approx(round(2.5), 3.0, 1e-12);
        assert_eq!(lround(2.7), 3);
        assert_eq!(llround(-2.7), -3);
    }

    #[test]
    fn modulo_remainder() {
        approx(fmod(5.0, 3.0), 2.0, 1e-12);
        approx(remainder(5.0, 3.0), -1.0, 1e-12);
        let (r, q) = remquo(5.0, 3.0);
        approx(r, -1.0, 1e-12);
        assert_eq!(q, 2);
    }

    #[test]
    fn frexp_modf() {
        let (m, e) = frexp(8.0);
        approx(m, 0.5, 1e-12);
        assert_eq!(e, 4);
        let (frac, int) = modf(3.7);
        approx(frac, 0.7, 1e-12);
        approx(int, 3.0, 1e-12);
        approx(ldexp(1.0, 10), 1024.0, 1e-12);
        approx(scalbn(1.0, 5), 32.0, 1e-12);
    }

    #[test]
    fn float_ops() {
        approx(copysign(3.0, -1.0), -3.0, 1e-12);
        approx(fdim(5.0, 3.0), 2.0, 1e-12);
        approx(fdim(3.0, 5.0), 0.0, 1e-12);
        approx(fmax(2.0, 3.0), 3.0, 1e-12);
        approx(fmin(2.0, 3.0), 2.0, 1e-12);
        assert!(finite(1.0));
        assert!(!finite(f64::INFINITY));
        assert!(!finite(f64::NAN));
        assert!(nextafter(1.0, 2.0) > 1.0);
    }

    #[test]
    fn erf_known_values() {
        approx(erf(0.0), 0.0, 1e-12);
        // erf(1) ≈ 0.8427007929497149
        approx(erf(1.0), 0.842_700_792_949_715, 1e-12);
    }

    #[test]
    fn complex_exp_cpow_clog_round_trip() {
        // e^(iπ) ≈ -1
        let z = Complex64::new(0.0, std::f64::consts::PI);
        let r = cexp(z);
        assert!((r.re + 1.0).abs() < 1e-10, "got {r:?}");
        assert!(r.im.abs() < 1e-10);

        // cpow(e, ln(2)) ≈ 2
        let e = Complex64::new(std::f64::consts::E, 0.0);
        let ln2 = Complex64::new(std::f64::consts::LN_2, 0.0);
        let two = cpow(e, ln2);
        assert!((two.re - 2.0).abs() < 1e-10);

        // clog(e) ≈ 1
        let one = cln(e);
        assert!((one.re - 1.0).abs() < 1e-10);
    }

    #[test]
    fn nan_with_tag_round_trips() {
        let n = nan("123").unwrap();
        assert!(n.is_nan());
    }

    #[test]
    fn f32_variants_match() {
        assert!((expf(1.0_f32) - std::f32::consts::E).abs() < 1e-5);
        assert!((sqrtf(2.0_f32) - std::f32::consts::SQRT_2).abs() < 1e-5);
        assert!((sinhf(0.5_f32) - 0.5_f32.sinh()).abs() < 1e-5);
    }
}
