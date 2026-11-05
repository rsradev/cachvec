//! Core numeric operations on `f32` slices.
//!
//! Accumulation uses four independent lanes with a fixed reduction order, so
//! results are deterministic and the compiler can auto-vectorize the loop.

use super::validation::validate;
use crate::error::{CachVecError, Result};

/// Dot product without validation. Iterates over the shorter slice.
#[inline]
pub(crate) fn dot_unchecked(a: &[f32], b: &[f32]) -> f32 {
    let ca = a.chunks_exact(4);
    let cb = b.chunks_exact(4);
    let (ra, rb) = (ca.remainder(), cb.remainder());
    let mut acc = [0.0f32; 4];
    for (x, y) in ca.zip(cb) {
        acc[0] += x[0] * y[0];
        acc[1] += x[1] * y[1];
        acc[2] += x[2] * y[2];
        acc[3] += x[3] * y[3];
    }
    let mut tail = 0.0f32;
    for (x, y) in ra.iter().zip(rb) {
        tail += x * y;
    }
    ((acc[0] + acc[1]) + (acc[2] + acc[3])) + tail
}

fn check_pair(a: &[f32], b: &[f32]) -> Result<()> {
    validate(a)?;
    validate(b)?;
    if a.len() != b.len() {
        return Err(CachVecError::DimensionMismatch {
            expected: a.len(),
            actual: b.len(),
        });
    }
    Ok(())
}

/// Dot product of two equal-length vectors.
pub fn dot(a: &[f32], b: &[f32]) -> Result<f32> {
    check_pair(a, b)?;
    Ok(dot_unchecked(a, b))
}

/// Euclidean (L2) norm. Returns `+inf` if the squared norm overflows.
pub fn norm(v: &[f32]) -> f32 {
    dot_unchecked(v, v).sqrt()
}

/// Normalizes `v` in place to unit length.
///
/// Fails with [`CachVecError::ZeroNorm`] for the zero vector and
/// [`CachVecError::NormOverflow`] if the norm is not representable.
pub fn normalize(v: &mut [f32]) -> Result<()> {
    validate(v)?;
    let n = norm(v);
    if !n.is_finite() {
        return Err(CachVecError::NormOverflow);
    }
    if n == 0.0 {
        return Err(CachVecError::ZeroNorm);
    }
    let inv = 1.0 / n;
    v.iter_mut().for_each(|x| *x *= inv);
    Ok(())
}

/// Returns a unit-length copy of `v`.
pub fn normalized(v: &[f32]) -> Result<Vec<f32>> {
    let mut out = v.to_vec();
    normalize(&mut out)?;
    Ok(out)
}

/// Euclidean distance between two equal-length vectors.
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> Result<f32> {
    check_pair(a, b)?;
    let mut sum = 0.0f32;
    for (x, y) in a.iter().zip(b) {
        let d = x - y;
        sum += d * d;
    }
    Ok(sum.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_matches_naive_for_odd_lengths() {
        for n in 1..20usize {
            let a: Vec<f32> = (0..n).map(|i| i as f32 + 1.0).collect();
            let b: Vec<f32> = (0..n).map(|i| 2.0 * i as f32 - 3.0).collect();
            let naive: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
            assert!((dot(&a, &b).unwrap() - naive).abs() < 1e-3, "n={n}");
        }
    }

    #[test]
    fn dot_rejects_mismatch() {
        assert!(matches!(
            dot(&[1.0], &[1.0, 2.0]),
            Err(CachVecError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn normalize_gives_unit_length() {
        let v = normalized(&[3.0, 4.0]).unwrap();
        assert!((norm(&v) - 1.0).abs() < 1e-6);
        assert!((v[0] - 0.6).abs() < 1e-6);
    }

    #[test]
    fn normalize_rejects_zero_and_overflow() {
        assert_eq!(normalized(&[0.0, 0.0]), Err(CachVecError::ZeroNorm));
        assert_eq!(normalized(&[1e30, 1e30]), Err(CachVecError::NormOverflow));
    }

    #[test]
    fn euclidean_basic() {
        assert!((euclidean_distance(&[0.0, 0.0], &[3.0, 4.0]).unwrap() - 5.0).abs() < 1e-6);
    }
}
