//! Hamming distance and its relation to angular similarity.
//!
//! For random-hyperplane hashing, one bit of two fingerprints differs with
//! probability `θ/π`, where `θ = arccos(cos_sim)` is the angle between the
//! source vectors. With `b` bits the expected Hamming distance is `b·θ/π`, so
//! the cosine can be estimated as `cos(π·d/b)`. The estimate has standard
//! deviation roughly `sqrt(p(1-p)/b)` in `d/b`; more bits tighten it.

use crate::error::{CashVecError, Result};

/// Number of differing bits between two equal-length word slices.
pub fn hamming_distance(a: &[u64], b: &[u64]) -> Result<u32> {
    if a.len() != b.len() {
        return Err(CashVecError::DimensionMismatch {
            expected: a.len(),
            actual: b.len(),
        });
    }
    Ok(a.iter().zip(b).map(|(x, y)| (x ^ y).count_ones()).sum())
}

/// Estimates cosine similarity from a Hamming distance over `bits` bits.
pub fn estimate_cosine(distance: u32, bits: usize) -> f32 {
    if bits == 0 {
        return 0.0;
    }
    let frac = (distance as f64 / bits as f64).clamp(0.0, 1.0);
    (std::f64::consts::PI * frac).cos() as f32
}

/// Largest Hamming distance (over `bits` bits) whose *expected* angle still
/// corresponds to a cosine of at least `min_cosine`.
pub fn max_hamming_for_cosine(min_cosine: f32, bits: usize) -> u32 {
    let c = (min_cosine as f64).clamp(-1.0, 1.0);
    (bits as f64 * c.acos() / std::f64::consts::PI).floor() as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distance_basics() {
        assert_eq!(hamming_distance(&[0b1011], &[0b0001]).unwrap(), 2);
        assert_eq!(hamming_distance(&[u64::MAX, 0], &[0, 0]).unwrap(), 64);
        assert!(hamming_distance(&[1], &[1, 2]).is_err());
    }

    #[test]
    fn symmetric() {
        let (a, b) = ([0xDEAD_BEEFu64, 7], [0x1234_5678u64, 9]);
        assert_eq!(hamming_distance(&a, &b), hamming_distance(&b, &a));
    }

    #[test]
    fn cosine_estimate_endpoints() {
        assert!((estimate_cosine(0, 128) - 1.0).abs() < 1e-6);
        assert!(estimate_cosine(64, 128).abs() < 1e-6);
        assert!((estimate_cosine(128, 128) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn max_hamming_monotone() {
        assert_eq!(max_hamming_for_cosine(1.0, 128), 0);
        assert!(max_hamming_for_cosine(0.9, 128) < max_hamming_for_cosine(0.5, 128));
        assert_eq!(max_hamming_for_cosine(0.0, 128), 64);
    }
}
