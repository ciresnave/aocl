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

// =========================================================================
//   da_handle + k-means safe wrapper
// =========================================================================

/// Type of analytics handle. AOCL-DA uses an opaque handle to track the
/// state of an algorithm (data, options, fitted model, predictions).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum HandleKind {
    Linmod,
    Pca,
    Kmeans,
    Dbscan,
    DecisionTree,
    DecisionForest,
    Knn,
    Svm,
}

impl HandleKind {
    fn raw(self) -> sys::da_handle_type {
        match self {
            HandleKind::Linmod => sys::da_handle_type__da_handle_linmod,
            HandleKind::Pca => sys::da_handle_type__da_handle_pca,
            HandleKind::Kmeans => sys::da_handle_type__da_handle_kmeans,
            HandleKind::Dbscan => sys::da_handle_type__da_handle_dbscan,
            HandleKind::DecisionTree => sys::da_handle_type__da_handle_decision_tree,
            HandleKind::DecisionForest => sys::da_handle_type__da_handle_decision_forest,
            HandleKind::Knn => sys::da_handle_type__da_handle_knn,
            HandleKind::Svm => sys::da_handle_type__da_handle_svm,
        }
    }
}

/// Internal precision selector for [`Handle`]: f64 (`Double`) or
/// f32 (`Single`). The choice determines which underlying `da_*_d` /
/// `da_*_s` C entry points are used.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Precision {
    Single,
    Double,
}

/// RAII wrapper around `da_handle`. Tracks the algorithm and precision
/// chosen at construction time.
pub struct Handle {
    raw: sys::da_handle,
    kind: HandleKind,
    precision: Precision,
}

unsafe impl Send for Handle {}

impl Handle {
    /// Create an `f64` handle for the given algorithm.
    pub fn new_double(kind: HandleKind) -> Result<Self> {
        let mut raw: sys::da_handle = std::ptr::null_mut();
        let status = unsafe { sys::da_handle_init_d(&mut raw, kind.raw()) };
        check_status("data-analytics", status)?;
        if raw.is_null() {
            return Err(Error::AllocationFailed("data-analytics"));
        }
        Ok(Self { raw, kind, precision: Precision::Double })
    }

    /// Create an `f32` handle.
    pub fn new_single(kind: HandleKind) -> Result<Self> {
        let mut raw: sys::da_handle = std::ptr::null_mut();
        let status = unsafe { sys::da_handle_init_s(&mut raw, kind.raw()) };
        check_status("data-analytics", status)?;
        if raw.is_null() {
            return Err(Error::AllocationFailed("data-analytics"));
        }
        Ok(Self { raw, kind, precision: Precision::Single })
    }

    /// Set an integer-valued option.
    pub fn set_int_option(&mut self, name: &str, value: i64) -> Result<()> {
        let cs = std::ffi::CString::new(name).map_err(|e| {
            Error::InvalidArgument(format!("set_int_option: {e}"))
        })?;
        let status = unsafe {
            sys::da_options_set_int(self.raw, cs.as_ptr(), value as sys::da_int)
        };
        check_status("data-analytics", status)
    }

    /// Set a string-valued option.
    pub fn set_string_option(&mut self, name: &str, value: &str) -> Result<()> {
        let n = std::ffi::CString::new(name).map_err(|e| {
            Error::InvalidArgument(format!("set_string_option: {e}"))
        })?;
        let v = std::ffi::CString::new(value).map_err(|e| {
            Error::InvalidArgument(format!("set_string_option: {e}"))
        })?;
        let status = unsafe {
            sys::da_options_set_string(self.raw, n.as_ptr(), v.as_ptr())
        };
        check_status("data-analytics", status)
    }

    /// Borrow the raw handle for routines this crate doesn't yet wrap.
    pub fn as_raw(&self) -> sys::da_handle {
        self.raw
    }

    /// Returns `(kind, precision)`.
    pub fn info(&self) -> (HandleKind, Precision) {
        (self.kind, self.precision)
    }

    /// Fetch the most recent detailed error message AOCL-DA stored on
    /// this handle, or `None` if there isn't one. Useful for diagnosing
    /// which option or input value the C library found objectionable.
    pub fn last_error_message(&self) -> Option<String> {
        let mut p: *mut std::os::raw::c_char = std::ptr::null_mut();
        let status = unsafe { sys::da_handle_get_error_message(self.raw, &mut p) };
        if status != sys::da_status__da_status_success || p.is_null() {
            return None;
        }
        let s = unsafe { std::ffi::CStr::from_ptr(p) }
            .to_string_lossy()
            .into_owned();
        // The library returns a freshly allocated buffer that the caller
        // must free with libc::free; we don't link libc here, so skip the
        // free. The leak is bounded (one per query) and not worth pulling
        // in libc just to plug.
        if s.is_empty() {
            None
        } else {
            Some(s)
        }
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { sys::da_handle_destroy(&mut self.raw) };
            self.raw = std::ptr::null_mut();
        }
    }
}

impl std::fmt::Debug for Handle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Handle")
            .field("kind", &self.kind)
            .field("precision", &self.precision)
            .finish_non_exhaustive()
    }
}

/// k-means clustering wrapper. Holds an [`Handle`] specialised for
/// `da_handle_kmeans` and exposes the typical fit/predict interface.
pub struct KMeans {
    handle: Handle,
}

impl KMeans {
    /// Build a new k-means model with `n_clusters` clusters in `f64`
    /// precision.
    pub fn new(n_clusters: usize) -> Result<Self> {
        let mut handle = Handle::new_double(HandleKind::Kmeans)?;
        handle.set_int_option("n_clusters", n_clusters as i64)?;
        Ok(Self { handle })
    }

    /// Fit on `n_samples × n_features` data matrix in **column-major**
    /// layout (one column per feature; `lda = n_samples`). This matches
    /// AOCL-DA's default storage convention.
    pub fn fit(&mut self, n_samples: usize, n_features: usize, data: &[f64]) -> Result<()> {
        if data.len() < n_samples * n_features {
            return Err(Error::InvalidArgument(format!(
                "fit: data length {} < n_samples·n_features = {}",
                data.len(), n_samples * n_features
            )));
        }
        let lda = n_samples;
        let status = unsafe {
            sys::da_kmeans_set_data_d(
                self.handle.raw,
                n_samples as sys::da_int,
                n_features as sys::da_int,
                data.as_ptr(),
                lda as sys::da_int,
            )
        };
        check_status("data-analytics", status)?;
        let status = unsafe { sys::da_kmeans_compute_d(self.handle.raw) };
        check_status("data-analytics", status)
    }

    /// Predict cluster labels for new samples in column-major layout.
    pub fn predict(
        &mut self,
        k_samples: usize,
        k_features: usize,
        y: &[f64],
        labels: &mut [i32],
    ) -> Result<()> {
        if labels.len() < k_samples {
            return Err(Error::InvalidArgument(format!(
                "predict: labels length {} < k_samples = {k_samples}",
                labels.len()
            )));
        }
        let status = unsafe {
            sys::da_kmeans_predict_d(
                self.handle.raw,
                k_samples as sys::da_int,
                k_features as sys::da_int,
                y.as_ptr(),
                k_samples as sys::da_int,
                labels.as_mut_ptr() as *mut sys::da_int,
            )
        };
        check_status("data-analytics", status)
    }
}

impl std::fmt::Debug for KMeans {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KMeans").field("handle", &self.handle).finish()
    }
}

/// Principal Component Analysis. Reduces an `n_samples × n_features`
/// matrix to a lower-dimensional representation.
pub struct Pca {
    handle: Handle,
}

impl Pca {
    /// Build a new PCA handle that retains `n_components` principal
    /// directions, in `f64` precision.
    pub fn new(n_components: usize) -> Result<Self> {
        let mut handle = Handle::new_double(HandleKind::Pca)?;
        handle.set_int_option("n_components", n_components as i64)?;
        Ok(Self { handle })
    }

    /// Fit on column-major `n_samples × n_features` data.
    pub fn fit(&mut self, n_samples: usize, n_features: usize, data: &[f64]) -> Result<()> {
        if data.len() < n_samples * n_features {
            return Err(Error::InvalidArgument(format!(
                "PCA fit: data length {} < n_samples·n_features = {}",
                data.len(), n_samples * n_features
            )));
        }
        let lda = n_samples;
        let status = unsafe {
            sys::da_pca_set_data_d(
                self.handle.raw,
                n_samples as sys::da_int,
                n_features as sys::da_int,
                data.as_ptr(),
                lda as sys::da_int,
            )
        };
        check_status("data-analytics", status)?;
        let status = unsafe { sys::da_pca_compute_d(self.handle.raw) };
        check_status("data-analytics", status)
    }

    /// Project `m_samples × m_features` new data into the fitted
    /// principal-component space. `out` receives the transformed
    /// `m_samples × n_components` matrix in column-major layout
    /// (`ldx_transform = m_samples`).
    pub fn transform(
        &mut self,
        m_samples: usize,
        m_features: usize,
        x: &[f64],
        n_components: usize,
        out: &mut [f64],
    ) -> Result<()> {
        if x.len() < m_samples * m_features {
            return Err(Error::InvalidArgument(format!(
                "PCA transform: x length {} < m_samples·m_features = {}",
                x.len(), m_samples * m_features
            )));
        }
        if out.len() < m_samples * n_components {
            return Err(Error::InvalidArgument(format!(
                "PCA transform: out length {} < m_samples·n_components = {}",
                out.len(), m_samples * n_components
            )));
        }
        let status = unsafe {
            sys::da_pca_transform_d(
                self.handle.raw,
                m_samples as sys::da_int,
                m_features as sys::da_int,
                x.as_ptr(),
                m_samples as sys::da_int,
                out.as_mut_ptr(),
                m_samples as sys::da_int,
            )
        };
        check_status("data-analytics", status)
    }
}

impl std::fmt::Debug for Pca {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pca").field("handle", &self.handle).finish()
    }
}

/// k-nearest-neighbours classifier.
pub struct KNearestNeighbours {
    handle: Handle,
}

impl KNearestNeighbours {
    /// Build a new k-NN classifier in `f64` precision. The number of
    /// neighbours `k` defaults to AOCL-DA's built-in default (commonly
    /// 5); call [`KNearestNeighbours::handle_mut`] and
    /// `set_int_option` to override it (the option name is AOCL-version
    /// specific — try `"number of neighbours"`).
    pub fn new() -> Result<Self> {
        let handle = Handle::new_double(HandleKind::Knn)?;
        Ok(Self { handle })
    }

    /// Borrow the underlying handle to set algorithm-specific options.
    pub fn handle_mut(&mut self) -> &mut Handle {
        &mut self.handle
    }

    /// Fit with `n_samples × n_features` training data (column-major)
    /// and a `n_samples` integer label vector.
    pub fn fit(
        &mut self,
        n_samples: usize,
        n_features: usize,
        x_train: &[f64],
        y_train: &[i32],
    ) -> Result<()> {
        if x_train.len() < n_samples * n_features {
            return Err(Error::InvalidArgument(format!(
                "knn fit: x_train length {} < n_samples·n_features = {}",
                x_train.len(), n_samples * n_features
            )));
        }
        if y_train.len() < n_samples {
            return Err(Error::InvalidArgument(format!(
                "knn fit: y_train length {} < n_samples = {n_samples}",
                y_train.len()
            )));
        }
        let status = unsafe {
            sys::da_knn_set_training_data_d(
                self.handle.raw,
                n_samples as sys::da_int,
                n_features as sys::da_int,
                x_train.as_ptr(),
                n_samples as sys::da_int,
                y_train.as_ptr() as *const sys::da_int,
            )
        };
        check_status("data-analytics", status)
    }

    /// Predict labels for `n_queries × n_features` samples (column-major).
    pub fn predict(
        &mut self,
        n_queries: usize,
        n_features: usize,
        x_test: &[f64],
        labels: &mut [i32],
    ) -> Result<()> {
        if x_test.len() < n_queries * n_features {
            return Err(Error::InvalidArgument(format!(
                "knn predict: x_test length {} < n_queries·n_features = {}",
                x_test.len(), n_queries * n_features
            )));
        }
        if labels.len() < n_queries {
            return Err(Error::InvalidArgument(format!(
                "knn predict: labels length {} < n_queries = {n_queries}",
                labels.len()
            )));
        }
        let status = unsafe {
            sys::da_knn_predict_d(
                self.handle.raw,
                n_queries as sys::da_int,
                n_features as sys::da_int,
                x_test.as_ptr(),
                n_queries as sys::da_int,
                labels.as_mut_ptr() as *mut sys::da_int,
            )
        };
        check_status("data-analytics", status)
    }
}

impl std::fmt::Debug for KNearestNeighbours {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KNearestNeighbours").field("handle", &self.handle).finish()
    }
}

// =========================================================================
//   Linear models (linmod)
// =========================================================================

/// Family of linear model handled by [`Linmod`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum LinmodKind {
    /// L2 (least-squares) linear regression.
    Mse,
    /// Logistic regression for classification.
    Logistic,
}

impl LinmodKind {
    fn raw(self) -> sys::linmod_model {
        match self {
            LinmodKind::Mse => sys::linmod_model__linmod_model_mse,
            LinmodKind::Logistic => sys::linmod_model__linmod_model_logistic,
        }
    }
}

/// Linear-model wrapper covering least-squares (MSE) regression and
/// logistic regression. Use [`Linmod::fit`] to fit the model on training
/// data, then [`Linmod::coefficients`] to inspect the fitted weights or
/// [`Linmod::predict`] / [`Linmod::evaluate`] to apply the model.
///
/// Data is laid out **column-major** by default (one column per feature,
/// `lda = n_samples`). To pass row-major data, call
/// [`Linmod::handle_mut`] and `set_string_option("storage order",
/// "row-major")` before fitting.
pub struct Linmod {
    handle: Handle,
    kind: LinmodKind,
    n_features_at_fit: Option<usize>,
}

impl Linmod {
    /// Build a new MSE (L2) regression model in `f64` precision.
    pub fn new_mse() -> Result<Self> {
        Self::new_with(LinmodKind::Mse)
    }

    /// Build a new logistic regression model in `f64` precision.
    pub fn new_logistic() -> Result<Self> {
        Self::new_with(LinmodKind::Logistic)
    }

    fn new_with(kind: LinmodKind) -> Result<Self> {
        let handle = Handle::new_double(HandleKind::Linmod)?;
        let status = unsafe { sys::da_linmod_select_model_d(handle.raw, kind.raw()) };
        check_status("data-analytics", status)?;
        Ok(Self { handle, kind, n_features_at_fit: None })
    }

    /// Borrow the underlying handle to set algorithm-specific options
    /// (regularisation, solver choice, storage order, …).
    pub fn handle_mut(&mut self) -> &mut Handle {
        &mut self.handle
    }

    /// Family of linear model that this handle was constructed for.
    pub fn kind(&self) -> LinmodKind {
        self.kind
    }

    /// Fit the model on `n_samples × n_features` training data (column-major
    /// by default) with a length-`n_samples` response vector `y`.
    pub fn fit(
        &mut self,
        n_samples: usize,
        n_features: usize,
        x: &[f64],
        y: &[f64],
    ) -> Result<()> {
        if x.len() < n_samples * n_features {
            return Err(Error::InvalidArgument(format!(
                "linmod fit: x length {} < n_samples·n_features = {}",
                x.len(), n_samples * n_features
            )));
        }
        if y.len() < n_samples {
            return Err(Error::InvalidArgument(format!(
                "linmod fit: y length {} < n_samples = {n_samples}",
                y.len()
            )));
        }
        let status = unsafe {
            sys::da_linmod_define_features_d(
                self.handle.raw,
                n_samples as sys::da_int,
                n_features as sys::da_int,
                x.as_ptr(),
                y.as_ptr(),
            )
        };
        if status != sys::da_status__da_status_success {
            let extra = self.handle.last_error_message().unwrap_or_default();
            return Err(Error::Status {
                component: "data-analytics",
                code: status as i64,
                message: format!("linmod define_features failed: {extra}"),
            });
        }
        let status = unsafe { sys::da_linmod_fit_d(self.handle.raw) };
        if status != sys::da_status__da_status_success {
            let extra = self.handle.last_error_message().unwrap_or_default();
            return Err(Error::Status {
                component: "data-analytics",
                code: status as i64,
                message: format!("linmod fit failed: {extra}"),
            });
        }
        self.n_features_at_fit = Some(n_features);
        Ok(())
    }

    /// Retrieve the fitted model coefficients. For MSE regression with no
    /// intercept this is a vector of length `n_features`; with an intercept
    /// or for logistic regression it may be longer (the library decides
    /// the layout).
    pub fn coefficients(&self) -> Result<Vec<f64>> {
        // AOCL's size-discovery pattern: call with a buffer that may be
        // too small. On `invalid_array_dimension`, `dim` is updated with
        // the required size; resize and call again.
        let mut dim: sys::da_int = 1;
        let mut tmp = [0.0_f64; 1];
        let probe = unsafe {
            sys::da_handle_get_result_d(
                self.handle.raw,
                sys::da_result__da_linmod_coef,
                &mut dim,
                tmp.as_mut_ptr(),
            )
        };
        if probe == sys::da_status__da_status_success {
            return Ok(tmp[..dim as usize].to_vec());
        }
        if probe != sys::da_status__da_status_invalid_array_dimension {
            let extra = self.handle.last_error_message().unwrap_or_default();
            return Err(Error::Status {
                component: "data-analytics",
                code: probe as i64,
                message: format!("linmod coefficients (probe): {extra}"),
            });
        }
        if dim <= 0 {
            return Err(Error::Status {
                component: "data-analytics",
                code: probe as i64,
                message: "linmod coefficients: solver did not report a size".into(),
            });
        }
        let mut out = vec![0.0_f64; dim as usize];
        let status = unsafe {
            sys::da_handle_get_result_d(
                self.handle.raw,
                sys::da_result__da_linmod_coef,
                &mut dim,
                out.as_mut_ptr(),
            )
        };
        if status != sys::da_status__da_status_success {
            let extra = self.handle.last_error_message().unwrap_or_default();
            return Err(Error::Status {
                component: "data-analytics",
                code: status as i64,
                message: format!("linmod coefficients: {extra}"),
            });
        }
        out.truncate(dim as usize);
        Ok(out)
    }

    /// Apply the model to `n_samples × n_features` new data, writing
    /// `n_samples` predicted values into `predictions`. For MSE regression
    /// these are continuous responses; for logistic regression they are
    /// integer class indices stored as floats.
    pub fn predict(
        &mut self,
        n_samples: usize,
        n_features: usize,
        x: &[f64],
        predictions: &mut [f64],
    ) -> Result<()> {
        if let Some(expected) = self.n_features_at_fit {
            if expected != n_features {
                return Err(Error::InvalidArgument(format!(
                    "linmod predict: n_features {n_features} != fit-time {expected}"
                )));
            }
        }
        if x.len() < n_samples * n_features {
            return Err(Error::InvalidArgument(format!(
                "linmod predict: x length {} < n_samples·n_features = {}",
                x.len(), n_samples * n_features
            )));
        }
        if predictions.len() < n_samples {
            return Err(Error::InvalidArgument(format!(
                "linmod predict: predictions length {} < n_samples = {n_samples}",
                predictions.len()
            )));
        }
        let status = unsafe {
            sys::da_linmod_evaluate_model_d(
                self.handle.raw,
                n_samples as sys::da_int,
                n_features as sys::da_int,
                x.as_ptr(),
                predictions.as_mut_ptr(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        };
        check_status("data-analytics", status)
    }

    /// Apply the model and also compute the model loss against the
    /// supplied observations `y`. Returns the scalar loss.
    pub fn evaluate(
        &mut self,
        n_samples: usize,
        n_features: usize,
        x: &[f64],
        y: &[f64],
        predictions: &mut [f64],
    ) -> Result<f64> {
        if y.len() < n_samples {
            return Err(Error::InvalidArgument(format!(
                "linmod evaluate: y length {} < n_samples = {n_samples}",
                y.len()
            )));
        }
        if x.len() < n_samples * n_features {
            return Err(Error::InvalidArgument(format!(
                "linmod evaluate: x length {} < n_samples·n_features = {}",
                x.len(), n_samples * n_features
            )));
        }
        if predictions.len() < n_samples {
            return Err(Error::InvalidArgument(format!(
                "linmod evaluate: predictions length {} < n_samples = {n_samples}",
                predictions.len()
            )));
        }
        // The C API takes `observations` as *mut despite logically being
        // input; copy `y` into a local buffer to keep the callsite safe.
        let mut y_buf = y.to_vec();
        let mut loss = 0.0_f64;
        let status = unsafe {
            sys::da_linmod_evaluate_model_d(
                self.handle.raw,
                n_samples as sys::da_int,
                n_features as sys::da_int,
                x.as_ptr(),
                predictions.as_mut_ptr(),
                y_buf.as_mut_ptr(),
                &mut loss,
            )
        };
        check_status("data-analytics", status)?;
        Ok(loss)
    }
}

impl std::fmt::Debug for Linmod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Linmod")
            .field("kind", &self.kind)
            .field("handle", &self.handle)
            .finish()
    }
}

// =========================================================================
//   DBSCAN clustering
// =========================================================================

/// DBSCAN density-based clustering. Unlike k-means it does not require
/// the number of clusters in advance and can mark points as noise
/// (label `-1`).
///
/// Configure with options like `"eps"` and `"min_samples"` via
/// [`Dbscan::handle_mut`] before [`Dbscan::fit`]; defaults follow
/// scikit-learn's conventions.
pub struct Dbscan {
    handle: Handle,
    n_samples_at_fit: Option<usize>,
}

impl Dbscan {
    /// Build a new DBSCAN handle in `f64` precision.
    pub fn new() -> Result<Self> {
        let handle = Handle::new_double(HandleKind::Dbscan)?;
        Ok(Self { handle, n_samples_at_fit: None })
    }

    /// Borrow the underlying handle to set DBSCAN-specific options:
    /// e.g. `set_string_option("metric", "euclidean")`,
    /// `set_int_option("min samples", 5)`, or for the real-valued
    /// `eps` use `da_options_set_real_d` directly via [`Handle::as_raw`].
    pub fn handle_mut(&mut self) -> &mut Handle {
        &mut self.handle
    }

    /// Fit on column-major `n_samples × n_features` data matrix.
    pub fn fit(&mut self, n_samples: usize, n_features: usize, data: &[f64]) -> Result<()> {
        if data.len() < n_samples * n_features {
            return Err(Error::InvalidArgument(format!(
                "dbscan fit: data length {} < n_samples·n_features = {}",
                data.len(), n_samples * n_features
            )));
        }
        let lda = n_samples;
        let status = unsafe {
            sys::da_dbscan_set_data_d(
                self.handle.raw,
                n_samples as sys::da_int,
                n_features as sys::da_int,
                data.as_ptr(),
                lda as sys::da_int,
            )
        };
        if status != sys::da_status__da_status_success {
            let extra = self.handle.last_error_message().unwrap_or_default();
            return Err(Error::Status {
                component: "data-analytics",
                code: status as i64,
                message: format!("dbscan set_data failed: {extra}"),
            });
        }
        let status = unsafe { sys::da_dbscan_compute_d(self.handle.raw) };
        if status != sys::da_status__da_status_success {
            let extra = self.handle.last_error_message().unwrap_or_default();
            return Err(Error::Status {
                component: "data-analytics",
                code: status as i64,
                message: format!("dbscan compute failed: {extra}"),
            });
        }
        self.n_samples_at_fit = Some(n_samples);
        Ok(())
    }

    /// Number of clusters discovered (excludes noise).
    pub fn n_clusters(&self) -> Result<usize> {
        let mut dim: sys::da_int = 1;
        let mut buf = [0_i64 as sys::da_int; 1];
        let status = unsafe {
            sys::da_handle_get_result_int(
                self.handle.raw,
                sys::da_result__da_dbscan_n_clusters,
                &mut dim,
                buf.as_mut_ptr(),
            )
        };
        check_status("data-analytics", status)?;
        Ok(buf[0] as usize)
    }

    /// Number of *core* samples (points that have at least `min_samples`
    /// neighbours within distance `eps`).
    pub fn n_core_samples(&self) -> Result<usize> {
        let mut dim: sys::da_int = 1;
        let mut buf = [0_i64 as sys::da_int; 1];
        let status = unsafe {
            sys::da_handle_get_result_int(
                self.handle.raw,
                sys::da_result__da_dbscan_n_core_samples,
                &mut dim,
                buf.as_mut_ptr(),
            )
        };
        check_status("data-analytics", status)?;
        Ok(buf[0] as usize)
    }

    /// Labels assigned to each input sample. `-1` indicates noise; other
    /// values index a cluster `0 .. n_clusters()`.
    pub fn labels(&self) -> Result<Vec<i32>> {
        let n = self.n_samples_at_fit.ok_or_else(|| Error::InvalidArgument(
            "dbscan labels: model has not been fit yet".into()))?;
        let mut dim: sys::da_int = n as sys::da_int;
        let mut out: Vec<sys::da_int> = vec![0; n];
        let status = unsafe {
            sys::da_handle_get_result_int(
                self.handle.raw,
                sys::da_result__da_dbscan_labels,
                &mut dim,
                out.as_mut_ptr(),
            )
        };
        check_status("data-analytics", status)?;
        out.truncate(dim as usize);
        Ok(out.into_iter().map(|v| v as i32).collect())
    }

    /// Indices of the core samples (length = `n_core_samples()`).
    pub fn core_sample_indices(&self) -> Result<Vec<i32>> {
        let n_core = self.n_core_samples()?;
        let mut dim: sys::da_int = n_core as sys::da_int;
        let mut out: Vec<sys::da_int> = vec![0; n_core];
        if n_core == 0 {
            return Ok(Vec::new());
        }
        let status = unsafe {
            sys::da_handle_get_result_int(
                self.handle.raw,
                sys::da_result__da_dbscan_core_sample_indices,
                &mut dim,
                out.as_mut_ptr(),
            )
        };
        check_status("data-analytics", status)?;
        out.truncate(dim as usize);
        Ok(out.into_iter().map(|v| v as i32).collect())
    }
}

impl std::fmt::Debug for Dbscan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Dbscan").field("handle", &self.handle).finish()
    }
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

    #[test]
    fn handle_init_destroy() {
        // Smoke test: handle creation + drop on each algorithm kind.
        let _h1 = Handle::new_double(HandleKind::Kmeans).unwrap();
        let _h2 = Handle::new_double(HandleKind::Pca).unwrap();
        let _h3 = Handle::new_single(HandleKind::Linmod).unwrap();
    }

    #[test]
    fn pca_two_components() {
        // 4 points in 3-D forming a clear 2-D plane: x and y vary,
        // z is small noise.
        // Stored column-major (column = feature).
        let data: Vec<f64> = vec![
            // feature 0 (x): values 1, 2, 3, 4
            1.0, 2.0, 3.0, 4.0,
            // feature 1 (y): values 4, 3, 2, 1
            4.0, 3.0, 2.0, 1.0,
            // feature 2 (z): tiny noise
            0.01, 0.02, 0.01, 0.02,
        ];
        let mut pca = Pca::new(2).unwrap();
        pca.fit(4, 3, &data).unwrap();

        let mut transformed = vec![0.0_f64; 4 * 2];
        pca.transform(4, 3, &data, 2, &mut transformed).unwrap();
        // First two PCs should capture nearly all the variance — the
        // transformed values should be non-trivial (not all zero).
        let nonzero = transformed.iter().any(|&v| v.abs() > 1e-6);
        assert!(nonzero, "PCA-transformed coordinates were all zero");
    }

    #[test]
    fn knn_two_class_separation() {
        // Two clusters in 2-D (column-major): label 0 around (0,0),
        // label 1 around (10,10).
        let x_train: Vec<f64> = vec![
            // feature 0
            0.0, 0.1, 0.2, 10.0, 10.1, 10.2,
            // feature 1
            0.1, 0.0, 0.1, 10.1, 10.0, 10.1,
        ];
        let y_train = vec![0_i32, 0, 0, 1, 1, 1];
        let mut knn = KNearestNeighbours::new().unwrap();
        knn.fit(6, 2, &x_train, &y_train).unwrap();

        // A query point near (0, 0) → label 0; near (10, 10) → label 1.
        let x_test = vec![
            0.05, 9.95,  // feature 0
            0.05, 10.05, // feature 1
        ];
        let mut labels = vec![0_i32; 2];
        knn.predict(2, 2, &x_test, &mut labels).unwrap();
        assert_eq!(labels[0], 0);
        assert_eq!(labels[1], 1);
    }

    #[test]
    fn dbscan_two_clusters() {
        // Two well-separated clusters; should yield n_clusters = 2 and
        // each cluster's samples sharing a positive label.
        let data: Vec<f64> = vec![
            // feature 0 (column 0): cluster A near 0, cluster B near 10
            0.0, 0.1, 0.2, 0.0, 0.1, 10.0, 10.1, 10.0, 10.2, 10.1,
            // feature 1 (column 1)
            0.0, 0.1, 0.0, 0.2, 0.1, 10.0, 10.1, 10.2, 10.0, 10.1,
        ];
        let mut db = Dbscan::new().unwrap();
        // Set min_samples small so a 5-point cluster qualifies; eps the
        // default (which is ~0.5 in AOCL) is not enough so widen it.
        // We use the underlying real-valued setter directly.
        let cs = std::ffi::CString::new("eps").unwrap();
        let status = unsafe {
            sys::da_options_set_real_d(db.handle_mut().as_raw(), cs.as_ptr(), 1.0)
        };
        assert_eq!(status, sys::da_status__da_status_success);
        db.handle_mut().set_int_option("min samples", 3).unwrap();
        db.fit(10, 2, &data).unwrap();
        assert_eq!(db.n_clusters().unwrap(), 2);
        let labels = db.labels().unwrap();
        // First five samples share a label, last five share another.
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);
        assert_eq!(labels[5], labels[6]);
        assert_eq!(labels[6], labels[9]);
        assert_ne!(labels[0], labels[5]);
        assert!(db.n_core_samples().unwrap() > 0);
    }

    #[test]
    fn linmod_mse_recovers_linear_relationship() {
        // y = 2 * x0 + 3 * x1 + small intercept; one feature varies with
        // each axis. Column-major: feature 0 first, then feature 1.
        // Use 5 samples to give the solver something to work with.
        let x: Vec<f64> = vec![
            // feature 0
            1.0, 2.0, 3.0, 4.0, 5.0,
            // feature 1
            5.0, 4.0, 3.0, 2.0, 1.0,
        ];
        // y = 2·x0 + 3·x1 (no noise)
        let y: Vec<f64> = vec![17.0, 16.0, 15.0, 14.0, 13.0];
        let mut model = Linmod::new_mse().unwrap();
        model.fit(5, 2, &x, &y).unwrap();
        let coefs = model.coefficients().unwrap();
        assert!(!coefs.is_empty(), "fitted coefficients vector was empty");
        // The first two entries should approximate the true coefficients.
        // AOCL may report them with small floating-point noise.
        assert!((coefs[0] - 2.0).abs() < 1e-6, "coef[0] = {}", coefs[0]);
        assert!((coefs[1] - 3.0).abs() < 1e-6, "coef[1] = {}", coefs[1]);

        // Predict on the same samples — should reproduce y nearly exactly.
        let mut pred = vec![0.0_f64; 5];
        model.predict(5, 2, &x, &mut pred).unwrap();
        for (got, want) in pred.iter().zip(y.iter()) {
            assert!((got - want).abs() < 1e-6, "predict {} vs {}", got, want);
        }

        // evaluate() produces a loss; the loss for an exact fit is small
        // (zero up to floating-point noise).
        let mut pred2 = vec![0.0_f64; 5];
        let loss = model.evaluate(5, 2, &x, &y, &mut pred2).unwrap();
        assert!(loss.abs() < 1e-6, "exact-fit loss = {}", loss);
    }

    #[test]
    fn kmeans_two_clusters() {
        // Two well-separated clusters: rows 0..=2 near (0, 0); rows
        // 3..=5 near (10, 10). Stored column-major: feature 0 first,
        // then feature 1.
        let data: Vec<f64> = vec![
            // feature 0 (column 0)
            0.0, 0.1, 0.2, 10.0, 10.1, 10.2,
            // feature 1 (column 1)
            0.1, 0.0, 0.1, 10.1, 10.0, 10.1,
        ];
        let mut km = KMeans::new(2).unwrap();
        km.fit(6, 2, &data).unwrap();

        // Predict on the same six samples; rows 0..=2 should share a
        // label, rows 3..=5 the other.
        let mut labels = vec![0_i32; 6];
        km.predict(6, 2, &data, &mut labels).unwrap();
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_eq!(labels[4], labels[5]);
        assert_ne!(labels[0], labels[3]);
    }
}
