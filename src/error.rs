//! Error types used across CashVec.

use std::fmt;

/// Errors returned by CashVec operations.
///
/// CashVec never panics on invalid user input; every failure is reported
/// through this type.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum CashVecError {
    /// The vector has zero dimensions.
    EmptyVector,
    /// Two dimensions that must agree do not.
    DimensionMismatch { expected: usize, actual: usize },
    /// The vector contains NaN or an infinity at `index`.
    InvalidValue { index: usize },
    /// The vector has zero length (all components zero); cosine is undefined.
    ZeroNorm,
    /// The squared norm overflowed `f32` (components are too large).
    NormOverflow,
    /// An item with this ID is already indexed.
    DuplicateId(String),
    /// A configuration parameter is out of range.
    InvalidConfig(String),
    /// A similarity threshold is NaN or outside `[-1, 1]`.
    InvalidThreshold,
    /// The index holds the maximum number of vectors it can address.
    CapacityExceeded,
}

impl fmt::Display for CashVecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyVector => write!(f, "vector is empty"),
            Self::DimensionMismatch { expected, actual } => {
                write!(f, "dimension mismatch: expected {expected}, got {actual}")
            }
            Self::InvalidValue { index } => {
                write!(f, "vector contains NaN or infinity at index {index}")
            }
            Self::ZeroNorm => write!(f, "vector has zero norm"),
            Self::NormOverflow => write!(f, "vector norm overflowed f32"),
            Self::DuplicateId(id) => write!(f, "id already indexed: {id}"),
            Self::InvalidConfig(msg) => write!(f, "invalid configuration: {msg}"),
            Self::InvalidThreshold => write!(f, "threshold must be a number in [-1, 1]"),
            Self::CapacityExceeded => write!(f, "index capacity exceeded"),
        }
    }
}

impl std::error::Error for CashVecError {}

/// Convenience result alias.
pub type Result<T> = std::result::Result<T, CashVecError>;
