//! Safe wrappers for AOCL-DA (Data Analytics).
//!
//! Currently exposes basic-statistics primitives (column / row / global
//! mean and variance) for `f32` and `f64` matrices. The broader DA API
//! (linear models, k-means, PCA, decision forests, etc.) requires the
//! `da_handle` machinery and will be wrapped incrementally.
//!
//! For routines not yet wrapped here, drop down to [`aocl_sys::data_analytics`].

use crate::error::{Error, Result};
use aocl_sys::data_analytics as sys;

/// Memory layout of an input matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Order {
    RowMajor,
    ColMajor,
}

impl Order {
    fn raw(self) -> sys::da_order {
        match self {
            Order::RowMajor => sys::da_order__row_major,
            Order::ColMajor => sys::da_order__column_major,
        }
    }
}

/// Reduction axis: per-column, per-row, or global.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Axis {
    /// Reduce each column down to a single value (output length = n_cols).
    Col,
    /// Reduce each row down to a single value (output length = n_rows).
    Row,
    /// Reduce all elements to a single value (output length = 1).
    All,
}

impl Axis {
    fn raw(self) -> sys::da_axis {
        match self {
            Axis::Col => sys::da_axis__da_axis_col,
            Axis::Row => sys::da_axis__da_axis_row,
            Axis::All => sys::da_axis__da_axis_all,
        }
    }

    fn output_len(self, n_rows: usize, n_cols: usize) -> usize {
        match self {
            Axis::Col => n_cols,
            Axis::Row => n_rows,
            Axis::All => 1,
        }
    }
}

fn check_status(component: &'static str, status: sys::da_status) -> Result<()> {
    if status == sys::da_status__da_status_success {
        return Ok(());
    }
    let message = match status {
        s if s == sys::da_status__da_status_internal_error => "internal error",
        s if s == sys::da_status__da_status_memory_error => "memory error",
        s if s == sys::da_status__da_status_invalid_pointer => "invalid pointer",
        s if s == sys::da_status__da_status_invalid_input => "invalid input",
        s if s == sys::da_status__da_status_not_implemented => "not implemented",
        s if s == sys::da_status__da_status_overflow => "overflow",
        s if s == sys::da_status__da_status_invalid_leading_dimension => {
            "invalid leading dimension"
        }
        s if s == sys::da_status__da_status_negative_data => "negative data",
        s if s == sys::da_status__da_status_invalid_array_dimension => {
            "invalid array dimension"
        }
        s if s == sys::da_status__da_status_no_data => "no data",
        _ => "unknown DA status",
    }
    .to_string();
    Err(Error::Status {
        component,
        code: status as i64,
        message,
    })
}

fn check_matrix(
    name: &str,
    order: Order,
    n_rows: usize,
    n_cols: usize,
    ldx: usize,
    x_len: usize,
) -> Result<()> {
    if n_rows == 0 || n_cols == 0 {
        return Ok(());
    }
    let (lead, trail) = match order {
        Order::RowMajor => (n_rows, n_cols),
        Order::ColMajor => (n_cols, n_rows),
    };
    if ldx < trail {
        return Err(Error::InvalidArgument(format!(
            "{name}: ldx={ldx} < {trail}"
        )));
    }
    let needed = (lead - 1) * ldx + trail;
    if x_len < needed {
        return Err(Error::InvalidArgument(format!(
            "{name}: slice length {x_len} too small (need {needed})"
        )));
    }
    Ok(())
}

/// Scalar element type usable with the wrapped DA routines.
pub trait Scalar: Copy + Sized + private::Sealed {
    /// Compute means along `axis` for an `n_rows × n_cols` matrix.
    #[allow(clippy::too_many_arguments)]
    fn mean(
        order: Order,
        axis: Axis,
        n_rows: usize,
        n_cols: usize,
        x: &[Self],
        ldx: usize,
        out: &mut [Self],
    ) -> Result<()>;

    /// Compute means and variances along `axis`.
    ///
    /// `dof` selects the divisor used in the variance:
    /// - `dof < 0` — divide by `n` (the number of observations).
    /// - `dof == 0` — divide by `n - 1` (Bessel-corrected, unbiased estimator).
    /// - `dof > 0` — divide by the literal value of `dof`.
    #[allow(clippy::too_many_arguments)]
    fn variance(
        order: Order,
        axis: Axis,
        n_rows: usize,
        n_cols: usize,
        x: &[Self],
        ldx: usize,
        dof: i64,
        mean_out: &mut [Self],
        variance_out: &mut [Self],
    ) -> Result<()>;
}

mod private {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

macro_rules! impl_scalar {
    ($t:ty, $mean:ident, $variance:ident) => {
        impl Scalar for $t {
            fn mean(
                order: Order,
                axis: Axis,
                n_rows: usize,
                n_cols: usize,
                x: &[Self],
                ldx: usize,
                out: &mut [Self],
            ) -> Result<()> {
                check_matrix("mean: X", order, n_rows, n_cols, ldx, x.len())?;
                let need = axis.output_len(n_rows, n_cols);
                if out.len() < need {
                    return Err(Error::InvalidArgument(format!(
                        "mean: out length {} < required {need}",
                        out.len()
                    )));
                }
                // SAFETY: pointers come from valid slices; integer casts
                // bounded by check_matrix above.
                let status = unsafe {
                    sys::$mean(
                        order.raw(),
                        axis.raw(),
                        n_rows as sys::da_int,
                        n_cols as sys::da_int,
                        x.as_ptr(),
                        ldx as sys::da_int,
                        out.as_mut_ptr(),
                    )
                };
                check_status("data-analytics", status)
            }

            fn variance(
                order: Order,
                axis: Axis,
                n_rows: usize,
                n_cols: usize,
                x: &[Self],
                ldx: usize,
                dof: i64,
                mean_out: &mut [Self],
                variance_out: &mut [Self],
            ) -> Result<()> {
                check_matrix("variance: X", order, n_rows, n_cols, ldx, x.len())?;
                let need = axis.output_len(n_rows, n_cols);
                if mean_out.len() < need || variance_out.len() < need {
                    return Err(Error::InvalidArgument(format!(
                        "variance: outputs must hold {need} elements"
                    )));
                }
                let status = unsafe {
                    sys::$variance(
                        order.raw(),
                        axis.raw(),
                        n_rows as sys::da_int,
                        n_cols as sys::da_int,
                        x.as_ptr(),
                        ldx as sys::da_int,
                        dof as sys::da_int,
                        mean_out.as_mut_ptr(),
                        variance_out.as_mut_ptr(),
                    )
                };
                check_status("data-analytics", status)
            }
        }
    };
}

impl_scalar!(f32, da_mean_s, da_variance_s);
impl_scalar!(f64, da_mean_d, da_variance_d);

/// Compute the mean of a tightly-packed row-major matrix along `axis`.
pub fn mean<T: Scalar>(
    axis: Axis,
    n_rows: usize,
    n_cols: usize,
    x: &[T],
    out: &mut [T],
) -> Result<()> {
    T::mean(Order::RowMajor, axis, n_rows, n_cols, x, n_cols, out)
}

/// Compute mean and variance of a tightly-packed row-major matrix along `axis`.
pub fn variance<T: Scalar>(
    axis: Axis,
    n_rows: usize,
    n_cols: usize,
    x: &[T],
    dof: i64,
    mean_out: &mut [T],
    variance_out: &mut [T],
) -> Result<()> {
    T::variance(
        Order::RowMajor,
        axis,
        n_rows,
        n_cols,
        x,
        n_cols,
        dof,
        mean_out,
        variance_out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn column_means_2x3() {
        // Row-major 2x3:
        //   [[1, 2, 3],
        //    [4, 5, 6]]
        // Column means: [2.5, 3.5, 4.5]
        let x = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut out = [0.0_f64; 3];
        mean(Axis::Col, 2, 3, &x, &mut out).unwrap();
        assert!((out[0] - 2.5).abs() < 1e-12);
        assert!((out[1] - 3.5).abs() < 1e-12);
        assert!((out[2] - 4.5).abs() < 1e-12);
    }

    #[test]
    fn row_means_2x3() {
        // Same matrix; row means: [2.0, 5.0]
        let x = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut out = [0.0_f64; 2];
        mean(Axis::Row, 2, 3, &x, &mut out).unwrap();
        assert!((out[0] - 2.0).abs() < 1e-12);
        assert!((out[1] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn variance_dof_conventions() {
        // Vector [1, 2, 3, 4, 5] as a 1xN matrix; mean = 3, Σ(xᵢ-μ)² = 10.
        let x = [1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let mut mu = [0.0_f64; 1];
        let mut var = [0.0_f64; 1];

        // dof < 0 → divide by n=5 → variance = 10/5 = 2.0
        variance(Axis::All, 1, 5, &x, -1, &mut mu, &mut var).unwrap();
        assert!((mu[0] - 3.0).abs() < 1e-12);
        assert!((var[0] - 2.0).abs() < 1e-12, "n-divisor var = {}", var[0]);

        // dof == 0 → divide by n-1=4 → variance = 10/4 = 2.5 (unbiased)
        variance(Axis::All, 1, 5, &x, 0, &mut mu, &mut var).unwrap();
        assert!((var[0] - 2.5).abs() < 1e-12, "(n-1)-divisor var = {}", var[0]);

        // dof == 2 → literal divisor → variance = 10/2 = 5.0
        variance(Axis::All, 1, 5, &x, 2, &mut mu, &mut var).unwrap();
        assert!((var[0] - 5.0).abs() < 1e-12, "literal-divisor var = {}", var[0]);
    }

    #[test]
    fn dim_mismatch_is_error() {
        let x = [1.0_f64; 4]; // claim 2x3 but only have 4 elements
        let mut out = [0.0_f64; 3];
        let err = mean(Axis::Col, 2, 3, &x, &mut out).unwrap_err();
        matches!(err, Error::InvalidArgument(_));
    }
}
