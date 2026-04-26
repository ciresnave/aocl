//! Safe wrappers for AOCL-DA (Data Analytics).

#![warn(missing_debug_implementations)]
#![cfg_attr(docsrs, feature(doc_cfg))]

use aocl_data_analytics_sys as sys;
pub use aocl_error::{Error, Result};
pub use aocl_types::Layout;
use aocl_types::sealed::Sealed;

fn layout_raw(l: Layout) -> sys::da_order {
    match l {
        Layout::RowMajor => sys::da_order__row_major,
        Layout::ColMajor => sys::da_order__column_major,
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
    layout: Layout,
    n_rows: usize,
    n_cols: usize,
    ldx: usize,
    x_len: usize,
) -> Result<()> {
    if n_rows == 0 || n_cols == 0 {
        return Ok(());
    }
    let (lead, trail) = match layout {
        Layout::RowMajor => (n_rows, n_cols),
        Layout::ColMajor => (n_cols, n_rows),
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
pub trait Scalar: Copy + Sized + Sealed {
    /// Compute means along `axis` for an `n_rows × n_cols` matrix.
    #[allow(clippy::too_many_arguments)]
    fn mean(
        layout: Layout,
        axis: Axis,
        n_rows: usize,
        n_cols: usize,
        x: &[Self],
        ldx: usize,
        out: &mut [Self],
    ) -> Result<()>;

    /// Compute means and variances along `axis`.
    ///
    /// `dof` selects the variance divisor:
    /// - `dof < 0` — divide by `n` (number of observations).
    /// - `dof == 0` — divide by `n - 1` (Bessel-corrected, unbiased).
    /// - `dof > 0` — divide by the literal value of `dof`.
    #[allow(clippy::too_many_arguments)]
    fn variance(
        layout: Layout,
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

macro_rules! impl_scalar {
    ($t:ty, $mean:ident, $variance:ident) => {
        impl Scalar for $t {
            fn mean(
                layout: Layout,
                axis: Axis,
                n_rows: usize,
                n_cols: usize,
                x: &[Self],
                ldx: usize,
                out: &mut [Self],
            ) -> Result<()> {
                check_matrix("mean: X", layout, n_rows, n_cols, ldx, x.len())?;
                let need = axis.output_len(n_rows, n_cols);
                if out.len() < need {
                    return Err(Error::InvalidArgument(format!(
                        "mean: out length {} < required {need}",
                        out.len()
                    )));
                }
                let status = unsafe {
                    sys::$mean(
                        layout_raw(layout),
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
                layout: Layout,
                axis: Axis,
                n_rows: usize,
                n_cols: usize,
                x: &[Self],
                ldx: usize,
                dof: i64,
                mean_out: &mut [Self],
                variance_out: &mut [Self],
            ) -> Result<()> {
                check_matrix("variance: X", layout, n_rows, n_cols, ldx, x.len())?;
                let need = axis.output_len(n_rows, n_cols);
                if mean_out.len() < need || variance_out.len() < need {
                    return Err(Error::InvalidArgument(format!(
                        "variance: outputs must hold {need} elements"
                    )));
                }
                let status = unsafe {
                    sys::$variance(
                        layout_raw(layout),
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
    T::mean(Layout::RowMajor, axis, n_rows, n_cols, x, n_cols, out)
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
        Layout::RowMajor,
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
        let x = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut out = [0.0_f64; 3];
        mean(Axis::Col, 2, 3, &x, &mut out).unwrap();
        assert!((out[0] - 2.5).abs() < 1e-12);
        assert!((out[1] - 3.5).abs() < 1e-12);
        assert!((out[2] - 4.5).abs() < 1e-12);
    }

    #[test]
    fn row_means_2x3() {
        let x = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut out = [0.0_f64; 2];
        mean(Axis::Row, 2, 3, &x, &mut out).unwrap();
        assert!((out[0] - 2.0).abs() < 1e-12);
        assert!((out[1] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn variance_dof_conventions() {
        let x = [1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let mut mu = [0.0_f64; 1];
        let mut var = [0.0_f64; 1];

        variance(Axis::All, 1, 5, &x, -1, &mut mu, &mut var).unwrap();
        assert!((mu[0] - 3.0).abs() < 1e-12);
        assert!((var[0] - 2.0).abs() < 1e-12, "n-divisor var = {}", var[0]);

        variance(Axis::All, 1, 5, &x, 0, &mut mu, &mut var).unwrap();
        assert!((var[0] - 2.5).abs() < 1e-12, "(n-1)-divisor var = {}", var[0]);

        variance(Axis::All, 1, 5, &x, 2, &mut mu, &mut var).unwrap();
        assert!((var[0] - 5.0).abs() < 1e-12, "literal-divisor var = {}", var[0]);
    }

    #[test]
    fn dim_mismatch_is_error() {
        let x = [1.0_f64; 4];
        let mut out = [0.0_f64; 3];
        let err = mean(Axis::Col, 2, 3, &x, &mut out).unwrap_err();
        matches!(err, Error::InvalidArgument(_));
    }
}
