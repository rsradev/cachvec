//! SimHash for dense embeddings.
//!
//! SimHash maps a vector to a `b`-bit signature, one bit per random
//! hyperplane (`bit_i = [v · r_i >= 0]`). The planes are drawn
//! deterministically from a seed, so equal inputs always give equal
//! signatures, and scaling a vector never changes its signature.
//!
//! # Hamming distance vs. angular similarity
//!
//! For vectors at angle `θ`, each bit differs independently with probability
//! `θ/π`. Hence `E[hamming] = b·θ/π` and the cosine can be estimated from an
//! observed distance `d` as `cos(π·d/b)`. The estimate is noisy: its standard
//! deviation in `d/b` is about `sqrt(p(1-p)/b)` with `p = θ/π`; use more bits
//! for a tighter estimate. Signatures are a *filter*, not a verdict: confirm
//! with exact cosine similarity when precision matters.
//!
//! Two hashers are provided:
//! * [`VectorHasher`] needs only the bit count; it generates plane
//!   coordinates on the fly for whatever dimension it is given. Convenient,
//!   but slower per call.
//! * [`SimHasher`] fixes the dimension and materializes the planes once,
//!   which is faster for repeated hashing. Both produce identical signatures
//!   for the same `(bits, seed)`.

use super::lsh::HyperplaneSet;
use super::BinaryFingerprint;
use crate::error::{CashVecError, Result};
use crate::rng::{fill_plane, plane_key, DEFAULT_SEED};
use crate::vector::operations::dot_unchecked;
use crate::vector::validation::validate;

/// Maximum supported signature size in bits.
pub const MAX_BITS: usize = 4096;

fn check_bits(bits: usize) -> Result<()> {
    if (1..=MAX_BITS).contains(&bits) {
        Ok(())
    } else {
        Err(CashVecError::InvalidConfig(format!(
            "bits must be in 1..={MAX_BITS}, got {bits}"
        )))
    }
}

/// Dimension-agnostic SimHash. `VectorHasher::new(128).hash(&embedding)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VectorHasher {
    bits: usize,
    seed: u64,
}

impl VectorHasher {
    /// A hasher producing `bits`-bit signatures with the default seed.
    ///
    /// `bits` is validated lazily: [`hash`](Self::hash) returns
    /// [`CashVecError::InvalidConfig`] unless `1 <= bits <= 4096`.
    pub fn new(bits: usize) -> Self {
        Self {
            bits,
            seed: DEFAULT_SEED,
        }
    }

    /// Like [`new`](Self::new) with an explicit seed.
    pub fn with_seed(bits: usize, seed: u64) -> Self {
        Self { bits, seed }
    }

    /// Signature size in bits.
    pub fn bits(&self) -> usize {
        self.bits
    }

    /// Hashes a vector of any (non-zero) dimension.
    pub fn hash(&self, v: &[f32]) -> Result<BinaryFingerprint> {
        check_bits(self.bits)?;
        validate(v)?;
        let mut scratch = vec![0.0f32; v.len()];
        let mut fp = BinaryFingerprint::zeros(self.bits);
        for p in 0..self.bits {
            fill_plane(plane_key(self.seed, p as u64), &mut scratch);
            if dot_unchecked(v, &scratch) >= 0.0 {
                fp.set_bit(p);
            }
        }
        Ok(fp)
    }
}

/// Fixed-dimension SimHash with precomputed hyperplanes.
#[derive(Debug, Clone)]
pub struct SimHasher {
    planes: HyperplaneSet,
}

impl SimHasher {
    /// Creates a hasher for `dim`-dimensional vectors with `bits`-bit output.
    pub fn new(dim: usize, bits: usize) -> Result<Self> {
        Self::with_seed(dim, bits, DEFAULT_SEED)
    }

    /// Like [`new`](Self::new) with an explicit seed.
    pub fn with_seed(dim: usize, bits: usize, seed: u64) -> Result<Self> {
        check_bits(bits)?;
        Ok(Self {
            planes: HyperplaneSet::new(dim, bits, seed)?,
        })
    }

    /// Input dimension.
    pub fn dim(&self) -> usize {
        self.planes.dim()
    }

    /// Signature size in bits.
    pub fn bits(&self) -> usize {
        self.planes.count()
    }

    /// Hashes `v`; its length must equal [`dim`](Self::dim).
    pub fn hash(&self, v: &[f32]) -> Result<BinaryFingerprint> {
        self.planes.signature(v)
    }

    /// True if the signatures of `a` and `b` are consistent with a cosine
    /// similarity of at least `min_cosine` (see the module docs; this is an
    /// estimate).
    pub fn is_similar(&self, a: &[f32], b: &[f32], min_cosine: f32) -> Result<bool> {
        self.hash(a)?.is_similar_to(&self.hash(b)?, min_cosine)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(dim: usize, salt: u32) -> Vec<f32> {
        (0..dim)
            .map(|i| (((i as u32 + 1).wrapping_mul(2654435761).wrapping_add(salt)) % 1000) as f32 / 500.0 - 1.0)
            .collect()
    }

    #[test]
    fn deterministic_and_scale_invariant() {
        let h = VectorHasher::new(128);
        let v = sample(32, 1);
        let w: Vec<f32> = v.iter().map(|x| x * 5.0).collect();
        assert_eq!(h.hash(&v).unwrap(), h.hash(&v).unwrap());
        assert_eq!(h.hash(&v).unwrap(), h.hash(&w).unwrap());
    }

    #[test]
    fn lazy_and_fixed_hashers_agree() {
        let v = sample(48, 3);
        let a = VectorHasher::with_seed(100, 5).hash(&v).unwrap();
        let b = SimHasher::with_seed(48, 100, 5).unwrap().hash(&v).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn bits_validation() {
        let v = [1.0, 2.0];
        assert!(VectorHasher::new(0).hash(&v).is_err());
        assert!(VectorHasher::new(MAX_BITS + 1).hash(&v).is_err());
        assert!(SimHasher::new(2, 0).is_err());
        assert!(VectorHasher::new(1).hash(&v).is_ok());
    }

    #[test]
    fn opposite_vectors_have_max_distance() {
        let h = SimHasher::new(16, 64).unwrap();
        let v = sample(16, 9);
        let neg: Vec<f32> = v.iter().map(|x| -x).collect();
        // Planes through the origin: sign flips on every bit (no exact zeros).
        assert_eq!(h.hash(&v).unwrap().hamming_distance(&h.hash(&neg).unwrap()).unwrap(), 64);
    }

    #[test]
    fn hamming_tracks_angle() {
        // Orthogonal-ish vs. near-identical pairs should be well separated.
        let h = SimHasher::new(64, 512).unwrap();
        let a = sample(64, 11);
        let mut near = a.clone();
        near[0] += 0.05;
        let far = sample(64, 12345);
        let d_near = h.hash(&a).unwrap().hamming_distance(&h.hash(&near).unwrap()).unwrap();
        let d_far = h.hash(&a).unwrap().hamming_distance(&h.hash(&far).unwrap()).unwrap();
        assert!(d_near < d_far, "{d_near} !< {d_far}");
    }

    #[test]
    fn rejects_invalid_vectors() {
        let h = VectorHasher::new(8);
        assert!(h.hash(&[]).is_err());
        assert!(h.hash(&[f32::NAN]).is_err());
        let s = SimHasher::new(3, 8).unwrap();
        assert!(s.hash(&[1.0, 2.0]).is_err());
    }
}
