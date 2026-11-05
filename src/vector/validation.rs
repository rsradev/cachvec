//! Input validation for embedding vectors.

use crate::error::{CashVecError, Result};

/// Validates that `v` is non-empty and contains only finite values.
pub fn validate(v: &[f32]) -> Result<()> {
    if v.is_empty() {
        return Err(CashVecError::EmptyVector);
    }
    if let Some(index) = v.iter().position(|x| !x.is_finite()) {
        return Err(CashVecError::InvalidValue { index });
    }
    Ok(())
}

/// Validates `v` (see [`validate`]) and that its length equals `expected`.
pub fn validate_dimension(v: &[f32], expected: usize) -> Result<()> {
    validate(v)?;
    if v.len() != expected {
        return Err(CashVecError::DimensionMismatch {
            expected,
            actual: v.len(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_empty_nan_inf() {
        assert_eq!(validate(&[]), Err(CashVecError::EmptyVector));
        assert_eq!(
            validate(&[1.0, f32::NAN]),
            Err(CashVecError::InvalidValue { index: 1 })
        );
        assert_eq!(
            validate(&[f32::INFINITY]),
            Err(CashVecError::InvalidValue { index: 0 })
        );
        assert_eq!(
            validate(&[f32::NEG_INFINITY]),
            Err(CashVecError::InvalidValue { index: 0 })
        );
        assert!(validate(&[0.0, -0.0, 1.5]).is_ok());
    }

    #[test]
    fn checks_dimension() {
        assert_eq!(
            validate_dimension(&[1.0, 2.0], 3),
            Err(CashVecError::DimensionMismatch {
                expected: 3,
                actual: 2
            })
        );
        assert!(validate_dimension(&[1.0, 2.0], 2).is_ok());
    }
}
