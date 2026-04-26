//! Safe wrappers for AOCL-RNG.
//!
//! Provides a [`Rng`] type holding the internal state of an AOCL pseudo-
//! random generator, plus methods to fill slices with samples from common
//! distributions. Currently exposes uniform, Gaussian, and exponential
//! sampling in `f64`.
//!
//! For generators not yet wrapped, drop down to [`aocl_sys::rng`].

use crate::error::{Error, Result};
use aocl_sys::rng as sys;

/// Base generator algorithm. The numeric values follow the NAG-style
/// `genid` convention used by AOCL-RNG.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum BaseGenerator {
    /// Linear-congruential generator.
    LinearCongruential = 1,
    /// Wichmann–Hill 1.
    WichmannHill = 2,
    /// Mersenne Twister.
    MersenneTwister = 3,
    /// Wichmann–Hill 2 (parallel-friendly).
    WichmannHillParallel = 4,
    /// Modified L'Ecuyer combined recursive generator.
    Mlfg = 5,
    /// Marsaglia's xorshift / multiply-with-carry family.
    Mwc = 6,
}

/// State buffer size, sized for the worst case (Mersenne Twister) so any
/// generator we expose fits.
const STATE_LEN: usize = 633;
/// Default seed buffer length (one i32 is enough for the generators we
/// expose; MT and friends derive their full internal state from a single
/// 32-bit seed).
const SEED_LEN: usize = 1;

/// AOCL pseudo-random number generator.
///
/// Wraps an AOCL state buffer initialized via `drandinitialize`. Samples
/// are produced into caller-provided slices to avoid heap allocation in
/// hot paths.
pub struct Rng {
    state: Box<[sys::rng_int_t; STATE_LEN]>,
}

impl Rng {
    /// Initialize a new generator.
    ///
    /// `seed` is a single 32-bit integer; AOCL expands it into the
    /// generator's full internal state.
    ///
    /// **Caveat:** several AOCL generators (notably the Mersenne Twister)
    /// reject `seed = 0`; use any non-zero seed.
    pub fn new(generator: BaseGenerator, seed: i32) -> Result<Self> {
        Self::with_subid(generator, 1, seed)
    }

    /// Initialize with an explicit `subid`. Most users want [`Self::new`];
    /// `subid` selects an independent stream within the generator family
    /// (typically `1..=NUM_STREAMS`).
    pub fn with_subid(generator: BaseGenerator, subid: i32, seed: i32) -> Result<Self> {
        let mut state = Box::new([0 as sys::rng_int_t; STATE_LEN]);
        let mut seed_buf = [seed as sys::rng_int_t; SEED_LEN];
        let mut lseed: sys::rng_int_t = SEED_LEN as sys::rng_int_t;
        let mut lstate: sys::rng_int_t = STATE_LEN as sys::rng_int_t;
        let mut info: sys::rng_int_t = 0;
        let genid: sys::rng_int_t = generator as sys::rng_int_t;
        let subid_raw: sys::rng_int_t = subid as sys::rng_int_t;

        // SAFETY: All pointers reference live local/heap allocations. The
        // routine only reads `genid`, `subid`, `seed`, `lseed`, and writes
        // into `state` (up to `lstate`) and `info`. The chosen lstate of
        // 633 is sufficient for every generator AOCL exposes.
        unsafe {
            sys::drandinitialize(
                genid,
                subid_raw,
                seed_buf.as_mut_ptr(),
                &mut lseed,
                state.as_mut_ptr(),
                &mut lstate,
                &mut info,
            );
        }

        if info != 0 {
            return Err(Error::Status {
                component: "rng",
                code: info as i64,
                message: format!("drandinitialize returned info={info}"),
            });
        }
        Ok(Rng { state })
    }

    /// Fill `out` with samples from a uniform `[a, b)` distribution.
    pub fn uniform(&mut self, a: f64, b: f64, out: &mut [f64]) -> Result<()> {
        if !(a < b) {
            return Err(Error::InvalidArgument(format!(
                "uniform: require a < b, got a={a} b={b}"
            )));
        }
        let n = out.len();
        if n == 0 {
            return Ok(());
        }
        let n_int: sys::rng_int_t = n.try_into().map_err(|_| {
            Error::InvalidArgument(format!(
                "uniform: n={n} exceeds rng_int_t range"
            ))
        })?;
        let mut info: sys::rng_int_t = 0;
        // SAFETY: state is owned + valid; out is a valid mutable slice of length n.
        unsafe {
            sys::dranduniform(
                n_int,
                a,
                b,
                self.state.as_mut_ptr(),
                out.as_mut_ptr(),
                &mut info,
            );
        }
        if info != 0 {
            return Err(Error::Status {
                component: "rng",
                code: info as i64,
                message: format!("dranduniform returned info={info}"),
            });
        }
        Ok(())
    }

    /// Fill `out` with samples from a Gaussian / normal `N(mean, variance)`
    /// distribution. (Note: AOCL takes the *variance*, not the standard
    /// deviation.)
    pub fn gaussian(&mut self, mean: f64, variance: f64, out: &mut [f64]) -> Result<()> {
        if variance < 0.0 {
            return Err(Error::InvalidArgument(format!(
                "gaussian: variance must be non-negative, got {variance}"
            )));
        }
        let n = out.len();
        if n == 0 {
            return Ok(());
        }
        let n_int: sys::rng_int_t = n.try_into().map_err(|_| {
            Error::InvalidArgument(format!(
                "gaussian: n={n} exceeds rng_int_t range"
            ))
        })?;
        let mut info: sys::rng_int_t = 0;
        unsafe {
            sys::drandgaussian(
                n_int,
                mean,
                variance,
                self.state.as_mut_ptr(),
                out.as_mut_ptr(),
                &mut info,
            );
        }
        if info != 0 {
            return Err(Error::Status {
                component: "rng",
                code: info as i64,
                message: format!("drandgaussian returned info={info}"),
            });
        }
        Ok(())
    }

    /// Fill `out` with samples from an exponential distribution with
    /// the given rate `λ` (mean = 1/λ).
    pub fn exponential(&mut self, mean: f64, out: &mut [f64]) -> Result<()> {
        if mean <= 0.0 {
            return Err(Error::InvalidArgument(format!(
                "exponential: mean must be positive, got {mean}"
            )));
        }
        let n = out.len();
        if n == 0 {
            return Ok(());
        }
        let n_int: sys::rng_int_t = n.try_into().map_err(|_| {
            Error::InvalidArgument(format!(
                "exponential: n={n} exceeds rng_int_t range"
            ))
        })?;
        let mut info: sys::rng_int_t = 0;
        unsafe {
            sys::drandexponential(
                n_int,
                mean,
                self.state.as_mut_ptr(),
                out.as_mut_ptr(),
                &mut info,
            );
        }
        if info != 0 {
            return Err(Error::Status {
                component: "rng",
                code: info as i64,
                message: format!("drandexponential returned info={info}"),
            });
        }
        Ok(())
    }
}

impl std::fmt::Debug for Rng {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Rng").finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mean(xs: &[f64]) -> f64 {
        xs.iter().sum::<f64>() / xs.len() as f64
    }

    fn variance(xs: &[f64], mu: f64) -> f64 {
        xs.iter().map(|&x| (x - mu).powi(2)).sum::<f64>() / xs.len() as f64
    }

    #[test]
    fn uniform_in_range() {
        let mut rng = Rng::new(BaseGenerator::MersenneTwister, 12345).unwrap();
        let mut out = vec![0.0_f64; 10_000];
        rng.uniform(0.0, 1.0, &mut out).unwrap();
        assert!(out.iter().all(|&x| (0.0..=1.0).contains(&x)));
        // Mean of U(0,1) is ~0.5; with n=10k, error well under 0.05.
        let mu = mean(&out);
        assert!((mu - 0.5).abs() < 0.05, "mean={mu}");
    }

    #[test]
    fn gaussian_stats_close() {
        let mut rng = Rng::new(BaseGenerator::MersenneTwister, 67890).unwrap();
        let mut out = vec![0.0_f64; 50_000];
        rng.gaussian(2.0, 4.0, &mut out).unwrap(); // mean=2, variance=4
        let mu = mean(&out);
        let var = variance(&out, mu);
        assert!((mu - 2.0).abs() < 0.1, "mean={mu}");
        assert!((var - 4.0).abs() < 0.2, "variance={var}");
    }

    #[test]
    fn reproducible_with_same_seed() {
        let mut a = Rng::new(BaseGenerator::MersenneTwister, 1).unwrap();
        let mut b = Rng::new(BaseGenerator::MersenneTwister, 1).unwrap();
        let mut va = vec![0.0_f64; 100];
        let mut vb = vec![0.0_f64; 100];
        a.uniform(0.0, 1.0, &mut va).unwrap();
        b.uniform(0.0, 1.0, &mut vb).unwrap();
        assert_eq!(va, vb);
    }

    #[test]
    fn invalid_uniform_range_is_error() {
        // seed must be non-zero for the Mersenne Twister generator;
        // a zero seed causes drandinitialize to return info=-3.
        let mut rng = Rng::new(BaseGenerator::MersenneTwister, 42).unwrap();
        let mut out = [0.0_f64; 4];
        let err = rng.uniform(1.0, 0.5, &mut out).unwrap_err();
        matches!(err, Error::InvalidArgument(_));
    }
}
