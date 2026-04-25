//! Cross-component error type.

use thiserror::Error;

/// Crate result alias.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors that can be returned by safe AOCL wrappers.
///
/// Each AOCL component has its own native error code conventions; this enum
/// is the lowest common denominator. Component-specific modules layer their
/// own typed errors on top where useful.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Error {
    /// An AOCL routine returned a non-success status code.
    #[error("AOCL '{component}' returned error code {code}: {message}")]
    Status {
        /// AOCL component that produced the error (e.g. "blis", "libflame").
        component: &'static str,
        /// Native return code from the AOCL routine.
        code: i64,
        /// Human-readable interpretation when one is available.
        message: String,
    },

    /// Invalid arguments passed by the caller (dimension mismatch, null buffer,
    /// etc.). These are detected client-side before invoking AOCL.
    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    /// A memory allocation requested by AOCL failed.
    #[error("AOCL allocation failed in component '{0}'")]
    AllocationFailed(&'static str),

    /// A C string returned by AOCL was not valid UTF-8.
    #[error("AOCL returned a non-UTF-8 string: {0}")]
    InvalidUtf8(#[from] std::str::Utf8Error),

    /// A C string returned by AOCL was missing its NUL terminator within bounds.
    #[error("AOCL returned a string without NUL terminator")]
    MissingNul,
}
