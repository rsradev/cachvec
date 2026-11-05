//! Cosine similarity.

use crate::error::{CashVecError, Result};
use crate::vector::operations::{dot_unchecked, norm};
use crate::vector::validation::validate;

/// Cosine similarity of two vectors, clamped to `[-1, 1]`.
///
/// Fails on empty/non-finite input, mismatched dimensions, or a zero-norm
/// operand (the angle is undefined).
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> Result<f32> {
    validate(a)?;
    validate(b)?;
    if a.len() != b.len() {
        return Err(CashVecError::DimensionMismatch {
            expected: a.len(),
            actual: b.len(),
        });
    }
    let (na, nb) = (norm(a), norm(b));
    if !na.is_finite() || !nb.is_finite() {
        return Err(CashVecError::NormOverflow);
    }
    if na == 0.0 || nb == 0.0 {
        return Err(CashVecError::ZeroNorm);
    }
    Ok((dot_unchecked(a, b) / (na * nb)).clamp(-1.0, 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_values() {
        assert!((cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]).unwrap() - 1.0).abs() < 1e-6);
        assert!(cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]).unwrap().abs() < 1e-6);
        assert!((cosine_similarity(&[1.0, 0.0], &[-2.0, 0.0]).unwrap() + 1.0).abs() < 1e-6);
    }

    #[test]
    fn scale_invariant_and_symmetric() {
        let a = [0.3, -1.2, 4.0, 0.5, 2.2];
        let b = [1.0, 0.4, -0.7, 3.0, 0.1];
        let s1 = cosine_similarity(&a, &b).unwrap();
        let s2 = cosine_similarity(&b, &a).unwrap();
        assert_eq!(s1, s2);
        let a2: Vec<f32> = a.iter().map(|x| x * 10.0).collect();
        assert!((cosine_similarity(&a2, &b).unwrap() - s1).abs() < 1e-5);
    }

    #[test]
    fn errors() {
        assert_eq!(
            cosine_similarity(&[0.0, 0.0], &[1.0, 1.0]),
            Err(CashVecError::ZeroNorm)
        );
        assert!(cosine_similarity(&[1.0], &[1.0, 2.0]).is_err());
        assert!(cosine_similarity(&[], &[]).is_err());
        assert!(cosine_similarity(&[f32::NAN], &[1.0]).is_err());
    }
}
