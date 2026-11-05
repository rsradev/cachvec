//! Vector representation, validation and numeric operations.

pub mod operations;
pub mod validation;

pub use operations::{dot, euclidean_distance, norm, normalize, normalized};
pub use validation::{validate, validate_dimension};

use crate::error::Result;

/// A validated embedding: non-empty, with only finite `f32` components.
#[derive(Debug, Clone, PartialEq)]
pub struct Vector(Vec<f32>);

impl Vector {
    /// Creates a vector, rejecting empty input, NaN and infinities.
    pub fn new(data: Vec<f32>) -> Result<Self> {
        validate(&data)?;
        Ok(Self(data))
    }

    /// Number of dimensions.
    pub fn dim(&self) -> usize {
        self.0.len()
    }

    /// Borrow the components.
    pub fn as_slice(&self) -> &[f32] {
        &self.0
    }

    /// Consume into the underlying `Vec`.
    pub fn into_inner(self) -> Vec<f32> {
        self.0
    }

    /// L2 norm.
    pub fn norm(&self) -> f32 {
        norm(&self.0)
    }

    /// Dot product with another vector of the same dimension.
    pub fn dot(&self, other: &Vector) -> Result<f32> {
        dot(&self.0, &other.0)
    }

    /// Unit-length copy.
    pub fn normalized(&self) -> Result<Vector> {
        Ok(Vector(normalized(&self.0)?))
    }

    /// Euclidean distance to another vector of the same dimension.
    pub fn euclidean_distance(&self, other: &Vector) -> Result<f32> {
        euclidean_distance(&self.0, &other.0)
    }

    /// Cosine similarity with another vector of the same dimension.
    pub fn cosine_similarity(&self, other: &Vector) -> Result<f32> {
        crate::similarity::cosine_similarity(&self.0, &other.0)
    }
}

impl AsRef<[f32]> for Vector {
    fn as_ref(&self) -> &[f32] {
        &self.0
    }
}

impl TryFrom<Vec<f32>> for Vector {
    type Error = crate::error::CachVecError;
    fn try_from(v: Vec<f32>) -> Result<Self> {
        Vector::new(v)
    }
}
