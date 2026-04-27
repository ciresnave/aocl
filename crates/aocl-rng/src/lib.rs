//! Safe wrappers for AOCL-RNG.

#![warn(missing_debug_implementations)]
#![cfg_attr(docsrs, feature(doc_cfg))]

use aocl_rng_sys as sys;
pub use aocl_error::{Error, Result};

/// Integer width used by AOCL-RNG for size parameters and integer-valued
/// distribution outputs. `i32` in the default LP64 build, `i64` in the
/// `ilp64` Cargo-feature build.
pub type RngInt = sys::rng_int_t;

/// Base generator algorithm. The numeric values follow the NAG-style
/// `genid` convention used by AOCL-RNG.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum BaseGenerator {
    LinearCongruential = 1,
    WichmannHill = 2,
    MersenneTwister = 3,
    WichmannHillParallel = 4,
    Mlfg = 5,
    Mwc = 6,
}

const STATE_LEN: usize = 633;
const SEED_LEN: usize = 1;

/// AOCL pseudo-random number generator.
pub struct Rng {
    state: Box<[sys::rng_int_t; STATE_LEN]>,
}

impl Rng {
    /// Initialize a new generator.
    ///
    /// **Caveat:** several AOCL generators (notably the Mersenne Twister)
    /// reject `seed = 0`; use any non-zero seed.
    pub fn new(generator: BaseGenerator, seed: i32) -> Result<Self> {
        Self::with_subid(generator, 1, seed)
    }

    /// Initialize with an explicit `subid`.
    pub fn with_subid(generator: BaseGenerator, subid: i32, seed: i32) -> Result<Self> {
        let mut state = Box::new([0 as sys::rng_int_t; STATE_LEN]);
        let mut seed_buf = [seed as sys::rng_int_t; SEED_LEN];
        let mut lseed: sys::rng_int_t = SEED_LEN as sys::rng_int_t;
        let mut lstate: sys::rng_int_t = STATE_LEN as sys::rng_int_t;
        let mut info: sys::rng_int_t = 0;
        let genid: sys::rng_int_t = generator as sys::rng_int_t;
        let subid_raw: sys::rng_int_t = subid as sys::rng_int_t;

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
    /// distribution. Note: AOCL takes the *variance*, not the standard
    /// deviation.
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
    /// the given mean.
    pub fn exponential(&mut self, mean: f64, out: &mut [f64]) -> Result<()> {
        if mean <= 0.0 {
            return Err(Error::InvalidArgument(format!(
                "exponential: mean must be positive, got {mean}"
            )));
        }
        self.fill_real("exponential", out, |n, st, p, info| unsafe {
            sys::drandexponential(n, mean, st, p, info)
        })
    }

    /// Beta distribution `B(a, b)`. Both shape parameters must be positive.
    pub fn beta(&mut self, a: f64, b: f64, out: &mut [f64]) -> Result<()> {
        self.fill_real("beta", out, |n, st, p, info| unsafe {
            sys::drandbeta(n, a, b, st, p, info)
        })
    }

    /// Cauchy distribution with median `med` and scale `scale`.
    pub fn cauchy(&mut self, med: f64, scale: f64, out: &mut [f64]) -> Result<()> {
        self.fill_real("cauchy", out, |n, st, p, info| unsafe {
            sys::drandcauchy(n, med, scale, st, p, info)
        })
    }

    /// χ² distribution with `df` degrees of freedom.
    pub fn chi_squared(&mut self, df: i32, out: &mut [f64]) -> Result<()> {
        self.fill_real("chi_squared", out, |n, st, p, info| unsafe {
            sys::drandchisquared(n, df as sys::rng_int_t, st, p, info)
        })
    }

    /// F distribution with `df1`, `df2` degrees of freedom.
    pub fn f_dist(&mut self, df1: i32, df2: i32, out: &mut [f64]) -> Result<()> {
        self.fill_real("f_dist", out, |n, st, p, info| unsafe {
            sys::drandf(n, df1 as sys::rng_int_t, df2 as sys::rng_int_t, st, p, info)
        })
    }

    /// Gamma distribution with shape `a` and scale `b`.
    pub fn gamma(&mut self, a: f64, b: f64, out: &mut [f64]) -> Result<()> {
        self.fill_real("gamma", out, |n, st, p, info| unsafe {
            sys::drandgamma(n, a, b, st, p, info)
        })
    }

    /// Logistic distribution with location `a` and scale `b`.
    pub fn logistic(&mut self, a: f64, b: f64, out: &mut [f64]) -> Result<()> {
        self.fill_real("logistic", out, |n, st, p, info| unsafe {
            sys::drandlogistic(n, a, b, st, p, info)
        })
    }

    /// Lognormal with underlying-normal mean `mean_ln` and variance `var_ln`.
    pub fn lognormal(&mut self, mean_ln: f64, var_ln: f64, out: &mut [f64]) -> Result<()> {
        self.fill_real("lognormal", out, |n, st, p, info| unsafe {
            sys::drandlognormal(n, mean_ln, var_ln, st, p, info)
        })
    }

    /// Student's t distribution with `df` degrees of freedom.
    pub fn students_t(&mut self, df: i32, out: &mut [f64]) -> Result<()> {
        self.fill_real("students_t", out, |n, st, p, info| unsafe {
            sys::drandstudentst(n, df as sys::rng_int_t, st, p, info)
        })
    }

    /// Triangular distribution on `[xmin, xmax]` with mode `xmed`.
    pub fn triangular(&mut self, xmin: f64, xmed: f64, xmax: f64, out: &mut [f64]) -> Result<()> {
        if !(xmin <= xmed && xmed <= xmax) {
            return Err(Error::InvalidArgument(format!(
                "triangular: require xmin <= xmed <= xmax, got {xmin}, {xmed}, {xmax}"
            )));
        }
        self.fill_real("triangular", out, |n, st, p, info| unsafe {
            sys::drandtriangular(n, xmin, xmed, xmax, st, p, info)
        })
    }

    /// Von Mises (circular normal) distribution with concentration `kappa`.
    pub fn von_mises(&mut self, kappa: f64, out: &mut [f64]) -> Result<()> {
        self.fill_real("von_mises", out, |n, st, p, info| unsafe {
            sys::drandvonmises(n, kappa, st, p, info)
        })
    }

    /// Weibull distribution with shape `a` and scale `b`.
    pub fn weibull(&mut self, a: f64, b: f64, out: &mut [f64]) -> Result<()> {
        self.fill_real("weibull", out, |n, st, p, info| unsafe {
            sys::drandweibull(n, a, b, st, p, info)
        })
    }

    // ---- Discrete distributions (integer-valued) -----------------------

    /// Binomial(`m`, `p`).
    pub fn binomial(&mut self, m: RngInt, p: f64, out: &mut [RngInt]) -> Result<()> {
        self.fill_int("binomial", out, |n, st, ptr, info| unsafe {
            sys::drandbinomial(n, m, p, st, ptr, info)
        })
    }

    /// Geometric distribution with success probability `p`.
    pub fn geometric(&mut self, p: f64, out: &mut [RngInt]) -> Result<()> {
        self.fill_int("geometric", out, |n, st, ptr, info| unsafe {
            sys::drandgeometric(n, p, st, ptr, info)
        })
    }

    /// Negative-binomial: number of failures before `m` successes, each
    /// with success probability `p`.
    pub fn negative_binomial(&mut self, m: RngInt, p: f64, out: &mut [RngInt]) -> Result<()> {
        self.fill_int("negative_binomial", out, |n, st, ptr, info| unsafe {
            sys::drandnegativebinomial(n, m, p, st, ptr, info)
        })
    }

    /// Poisson with mean `lambda`.
    pub fn poisson(&mut self, lambda: f64, out: &mut [RngInt]) -> Result<()> {
        self.fill_int("poisson", out, |n, st, ptr, info| unsafe {
            sys::drandpoisson(n, lambda, st, ptr, info)
        })
    }

    /// Discrete uniform on the inclusive integer range `[a, b]`.
    pub fn discrete_uniform(&mut self, a: RngInt, b: RngInt, out: &mut [RngInt]) -> Result<()> {
        if a > b {
            return Err(Error::InvalidArgument(format!(
                "discrete_uniform: require a <= b, got a={a} b={b}"
            )));
        }
        self.fill_int("discrete_uniform", out, |n, st, ptr, info| unsafe {
            sys::dranddiscreteuniform(n, a, b, st, ptr, info)
        })
    }

    /// Hypergeometric distribution: number of successes in `m` draws
    /// without replacement from a population of `np` items containing
    /// `ns` successes.
    pub fn hypergeometric(&mut self, np: RngInt, ns: RngInt, m: RngInt, out: &mut [RngInt]) -> Result<()> {
        self.fill_int("hypergeometric", out, |n, st, ptr, info| unsafe {
            sys::drandhypergeometric(n, np, ns, m, st, ptr, info)
        })
    }

    /// General discrete distribution. `ref_dist` is the cumulative
    /// probability vector (sorted, with `ref_dist[k-1] == 1.0`); `out`
    /// receives integer indices into it.
    pub fn general_discrete(&mut self, ref_dist: &mut [f64], out: &mut [RngInt]) -> Result<()> {
        let n = out.len();
        if n == 0 {
            return Ok(());
        }
        let n_int: RngInt = n.try_into().map_err(|_| {
            Error::InvalidArgument(format!("general_discrete: n={n} exceeds rng_int_t range"))
        })?;
        let mut info: RngInt = 0;
        unsafe {
            sys::drandgeneraldiscrete(
                n_int,
                ref_dist.as_mut_ptr(),
                self.state.as_mut_ptr(),
                out.as_mut_ptr(),
                &mut info,
            );
        }
        if info != 0 {
            return Err(Error::Status {
                component: "rng",
                code: info as i64,
                message: format!("general_discrete returned info={info}"),
            });
        }
        Ok(())
    }

    /// Multinomial distribution: each of `n` samples is the count
    /// vector (length `k`) from `m` trials with category probabilities
    /// `p` (length `k`). `out` is laid out as `n × k` with leading
    /// dimension `ldx`.
    #[allow(clippy::too_many_arguments)]
    pub fn multinomial(
        &mut self,
        m: RngInt,
        p: &mut [f64],
        k: RngInt,
        out: &mut [RngInt],
        ldx: RngInt,
    ) -> Result<()> {
        let n_samples = if k > 0 { out.len() / k as usize } else { 0 };
        if n_samples == 0 {
            return Ok(());
        }
        let n_int: RngInt = n_samples.try_into().map_err(|_| {
            Error::InvalidArgument(format!("multinomial: n={n_samples} exceeds rng_int_t range"))
        })?;
        let mut info: RngInt = 0;
        unsafe {
            sys::drandmultinomial(
                n_int, m, p.as_mut_ptr(), k,
                self.state.as_mut_ptr(),
                out.as_mut_ptr(), ldx,
                &mut info,
            );
        }
        if info != 0 {
            return Err(Error::Status {
                component: "rng",
                code: info as i64,
                message: format!("multinomial returned info={info}"),
            });
        }
        Ok(())
    }

    /// Multivariate normal: `n` samples each drawn from `N(xmu, C)`
    /// where `C` is an `m × m` covariance matrix with leading
    /// dimension `ldc`. `out` is laid out as `n × m` with leading
    /// dimension `ldx`.
    #[allow(clippy::too_many_arguments)]
    pub fn multinormal(
        &mut self,
        m: RngInt,
        xmu: &mut [f64],
        c: &mut [f64], ldc: RngInt,
        out: &mut [f64], ldx: RngInt,
    ) -> Result<()> {
        let n_samples = if m > 0 { out.len() / m as usize } else { 0 };
        if n_samples == 0 {
            return Ok(());
        }
        let n_int: RngInt = n_samples.try_into().map_err(|_| {
            Error::InvalidArgument(format!("multinormal: n={n_samples} exceeds rng_int_t range"))
        })?;
        let mut info: RngInt = 0;
        unsafe {
            sys::drandmultinormal(
                n_int, m,
                xmu.as_mut_ptr(),
                c.as_mut_ptr(), ldc,
                self.state.as_mut_ptr(),
                out.as_mut_ptr(), ldx,
                &mut info,
            );
        }
        if info != 0 {
            return Err(Error::Status {
                component: "rng",
                code: info as i64,
                message: format!("multinormal returned info={info}"),
            });
        }
        Ok(())
    }

    /// Skip-ahead: advance the state as if `n_skip` samples had been
    /// drawn. Useful for splitting a single seed into independent
    /// substreams.
    pub fn skip_ahead(&mut self, n_skip: u64) -> Result<()> {
        let n_int: RngInt = n_skip.try_into().map_err(|_| {
            Error::InvalidArgument(format!("skip_ahead: n={n_skip} exceeds rng_int_t range"))
        })?;
        let mut info: RngInt = 0;
        unsafe { sys::drandskipahead(n_int, self.state.as_mut_ptr(), &mut info); }
        if info != 0 {
            return Err(Error::Status {
                component: "rng",
                code: info as i64,
                message: format!("skip_ahead returned info={info}"),
            });
        }
        Ok(())
    }

    /// Leapfrog: split this stream into `n` substreams and advance
    /// this state to substream `k` (0-based). Each `leapfrog(n, _)`
    /// call subselects every `n`-th sample.
    pub fn leapfrog(&mut self, n: RngInt, k: RngInt) -> Result<()> {
        let mut info: RngInt = 0;
        unsafe { sys::drandleapfrog(n, k, self.state.as_mut_ptr(), &mut info); }
        if info != 0 {
            return Err(Error::Status {
                component: "rng",
                code: info as i64,
                message: format!("leapfrog returned info={info}"),
            });
        }
        Ok(())
    }

    // ---- internal helpers ----------------------------------------------

    fn fill_real<F>(&mut self, op: &'static str, out: &mut [f64], call: F) -> Result<()>
    where
        F: FnOnce(sys::rng_int_t, *mut sys::rng_int_t, *mut f64, *mut sys::rng_int_t),
    {
        let n = out.len();
        if n == 0 {
            return Ok(());
        }
        let n_int: sys::rng_int_t = n.try_into().map_err(|_| {
            Error::InvalidArgument(format!("{op}: n={n} exceeds rng_int_t range"))
        })?;
        let mut info: sys::rng_int_t = 0;
        call(n_int, self.state.as_mut_ptr(), out.as_mut_ptr(), &mut info);
        if info != 0 {
            return Err(Error::Status {
                component: "rng",
                code: info as i64,
                message: format!("{op} returned info={info}"),
            });
        }
        Ok(())
    }

    fn fill_int<F>(&mut self, op: &'static str, out: &mut [RngInt], call: F) -> Result<()>
    where
        F: FnOnce(sys::rng_int_t, *mut sys::rng_int_t, *mut sys::rng_int_t, *mut sys::rng_int_t),
    {
        let n = out.len();
        if n == 0 {
            return Ok(());
        }
        let n_int: sys::rng_int_t = n.try_into().map_err(|_| {
            Error::InvalidArgument(format!("{op}: n={n} exceeds rng_int_t range"))
        })?;
        let mut info: sys::rng_int_t = 0;
        call(n_int, self.state.as_mut_ptr(), out.as_mut_ptr(), &mut info);
        if info != 0 {
            return Err(Error::Status {
                component: "rng",
                code: info as i64,
                message: format!("{op} returned info={info}"),
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
        let mu = mean(&out);
        assert!((mu - 0.5).abs() < 0.05, "mean={mu}");
    }

    #[test]
    fn gaussian_stats_close() {
        let mut rng = Rng::new(BaseGenerator::MersenneTwister, 67890).unwrap();
        let mut out = vec![0.0_f64; 50_000];
        rng.gaussian(2.0, 4.0, &mut out).unwrap();
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
    fn beta_in_unit_interval() {
        let mut rng = Rng::new(BaseGenerator::MersenneTwister, 13).unwrap();
        let mut out = vec![0.0_f64; 5_000];
        rng.beta(2.0, 5.0, &mut out).unwrap();
        assert!(out.iter().all(|&x| (0.0..=1.0).contains(&x)));
        // Beta(2,5) mean = 2/(2+5) ≈ 0.2857
        let mu = mean(&out);
        assert!((mu - 2.0 / 7.0).abs() < 0.05, "mean={mu}");
    }

    #[test]
    fn gamma_positive_and_mean() {
        let mut rng = Rng::new(BaseGenerator::MersenneTwister, 17).unwrap();
        let mut out = vec![0.0_f64; 5_000];
        rng.gamma(3.0, 2.0, &mut out).unwrap();
        // Gamma(shape=3, scale=2) is positive, mean = 6.
        assert!(out.iter().all(|&x| x > 0.0));
        let mu = mean(&out);
        assert!((mu - 6.0).abs() < 0.5, "mean={mu}");
    }

    #[test]
    fn weibull_positive() {
        let mut rng = Rng::new(BaseGenerator::MersenneTwister, 23).unwrap();
        let mut out = vec![0.0_f64; 1_000];
        rng.weibull(1.5, 1.0, &mut out).unwrap();
        assert!(out.iter().all(|&x| x >= 0.0));
    }

    #[test]
    fn poisson_non_negative_with_mean() {
        let mut rng = Rng::new(BaseGenerator::MersenneTwister, 29).unwrap();
        let mut out = vec![0 as RngInt; 5_000];
        rng.poisson(4.5, &mut out).unwrap();
        assert!(out.iter().all(|&x| x >= 0));
        let mu = out.iter().map(|&x| x as f64).sum::<f64>() / out.len() as f64;
        assert!((mu - 4.5).abs() < 0.2, "mean={mu}");
    }

    #[test]
    fn discrete_uniform_in_range() {
        let mut rng = Rng::new(BaseGenerator::MersenneTwister, 31).unwrap();
        let mut out = vec![0 as RngInt; 1_000];
        rng.discrete_uniform(10, 20, &mut out).unwrap();
        assert!(out.iter().all(|&x| (10..=20).contains(&x)));
    }

    #[test]
    fn binomial_in_range() {
        let mut rng = Rng::new(BaseGenerator::MersenneTwister, 41).unwrap();
        let mut out = vec![0 as RngInt; 2_000];
        rng.binomial(10, 0.3, &mut out).unwrap();
        assert!(out.iter().all(|&x| (0..=10).contains(&x)));
        // Mean = 10·0.3 = 3
        let mu = out.iter().map(|&x| x as f64).sum::<f64>() / out.len() as f64;
        assert!((mu - 3.0).abs() < 0.2, "mean={mu}");
    }

    #[test]
    fn triangular_in_range() {
        let mut rng = Rng::new(BaseGenerator::MersenneTwister, 47).unwrap();
        let mut out = vec![0.0_f64; 2_000];
        rng.triangular(-1.0, 0.0, 1.0, &mut out).unwrap();
        assert!(out.iter().all(|&x| (-1.0..=1.0).contains(&x)));
    }

    #[test]
    fn invalid_uniform_range_is_error() {
        let mut rng = Rng::new(BaseGenerator::MersenneTwister, 42).unwrap();
        let mut out = [0.0_f64; 4];
        let err = rng.uniform(1.0, 0.5, &mut out).unwrap_err();
        matches!(err, Error::InvalidArgument(_));
    }
}
