//! Exact (bit-level) vector fingerprinting.
//!
//! An [`ExactFingerprint`] identifies vector *data*, not meaning: two vectors
//! that are merely similar get unrelated fingerprints. It is a 128-bit
//! non-cryptographic hash, so collisions are possible in principle (about
//! 2^-128 per pair); consumers that need certainty must compare the vectors.
//!
//! Canonicalization, applied before hashing:
//! * `-0.0` is mapped to `+0.0` (they compare equal, so they hash equal).
//! * NaN and infinities are rejected, so no NaN payload ambiguity exists.
//! * The dimension is hashed first, so `[0.0]` and `[0.0, 0.0]` differ.
//! * Components are hashed by their IEEE-754 bit pattern, little-endian
//!   independent (we hash integers, not bytes), so results are identical on
//!   every platform.

use crate::error::Result;
use crate::vector::validation::validate;
use std::fmt;

/// A 128-bit exact fingerprint of a canonicalized vector.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ExactFingerprint(u128);

impl ExactFingerprint {
    /// The raw 128-bit value.
    pub fn as_u128(&self) -> u128 {
        self.0
    }

    /// Big-endian byte representation.
    pub fn to_bytes(&self) -> [u8; 16] {
        self.0.to_be_bytes()
    }
}

impl fmt::Display for ExactFingerprint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:032x}", self.0)
    }
}

impl fmt::Debug for ExactFingerprint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ExactFingerprint({self})")
    }
}

/// Canonical bit pattern of a finite float (`-0.0` becomes `+0.0`).
#[inline]
pub(crate) fn canonical_bits(x: f32) -> u32 {
    if x == 0.0 {
        0
    } else {
        x.to_bits()
    }
}

/// True if two (validated) vectors are equal after canonicalization.
pub(crate) fn canonical_eq(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| canonical_bits(*x) == canonical_bits(*y))
}

const K1: u64 = 0x87C3_7B91_1142_53D5;
const K2: u64 = 0x4CF5_AD43_2745_937F;

#[inline]
fn fmix64(mut k: u64) -> u64 {
    k ^= k >> 33;
    k = k.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    k ^= k >> 33;
    k = k.wrapping_mul(0xC4CE_B9FE_1A85_EC53);
    k ^ (k >> 33)
}

/// Computes exact fingerprints. Stateless and cheap to copy.
#[derive(Debug, Clone, Copy, Default)]
pub struct ExactFingerprinter;

impl ExactFingerprinter {
    /// Creates a fingerprinter.
    pub fn new() -> Self {
        Self
    }

    /// Fingerprints a vector. Fails on empty, NaN or infinite input.
    pub fn fingerprint(&self, v: &[f32]) -> Result<ExactFingerprint> {
        fingerprint(v)
    }
}

/// Fingerprints a vector. See the [module docs](self) for canonicalization.
pub fn fingerprint(v: &[f32]) -> Result<ExactFingerprint> {
    validate(v)?;
    let mut h1: u64 = 0x243F_6A88_85A3_08D3;
    let mut h2: u64 = 0x1319_8A2E_0370_7344;
    let mut words: u64 = 0;

    let mut absorb = |w: u64| {
        h1 = (h1.rotate_left(27) ^ w.wrapping_mul(K1)).wrapping_mul(5).wrapping_add(0x52DC_E729);
        h2 = (h2.rotate_left(31) ^ w.wrapping_mul(K2)).wrapping_mul(5).wrapping_add(0x3849_5AB5);
        h2 = h2.wrapping_add(h1);
        words += 1;
    };

    absorb(v.len() as u64);
    let mut it = v.chunks_exact(2);
    for pair in &mut it {
        absorb(canonical_bits(pair[0]) as u64 | (canonical_bits(pair[1]) as u64) << 32);
    }
    if let [last] = it.remainder() {
        absorb(canonical_bits(*last) as u64);
    }

    h1 ^= words;
    h2 ^= words;
    h1 = h1.wrapping_add(h2);
    h2 = h2.wrapping_add(h1);
    h1 = fmix64(h1);
    h2 = fmix64(h2);
    h1 = h1.wrapping_add(h2);
    h2 = h2.wrapping_add(h1);
    Ok(ExactFingerprint(((h1 as u128) << 64) | h2 as u128))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::CachVecError;

    #[test]
    fn deterministic_and_discriminating() {
        let a = fingerprint(&[0.1, 0.2, 0.3]).unwrap();
        assert_eq!(a, fingerprint(&[0.1, 0.2, 0.3]).unwrap());
        assert_ne!(a, fingerprint(&[0.1, 0.2, 0.3001]).unwrap());
        assert_ne!(a, fingerprint(&[0.2, 0.1, 0.3]).unwrap());
    }

    #[test]
    fn negative_zero_is_canonicalized() {
        assert_eq!(
            fingerprint(&[0.0, 1.0]).unwrap(),
            fingerprint(&[-0.0, 1.0]).unwrap()
        );
    }

    #[test]
    fn dimension_is_part_of_input() {
        assert_ne!(fingerprint(&[0.0]).unwrap(), fingerprint(&[0.0, 0.0]).unwrap());
        assert_ne!(
            fingerprint(&[0.0, 0.0]).unwrap(),
            fingerprint(&[0.0, 0.0, 0.0]).unwrap()
        );
    }

    #[test]
    fn rejects_invalid() {
        assert_eq!(fingerprint(&[]), Err(CachVecError::EmptyVector));
        assert!(fingerprint(&[f32::NAN]).is_err());
        assert!(fingerprint(&[f32::INFINITY]).is_err());
    }

    #[test]
    fn no_collisions_on_many_vectors() {
        let mut seen = std::collections::HashSet::new();
        for i in 0..20_000u32 {
            let v = [i as f32, (i * 7) as f32 * 0.5, 1.0];
            assert!(seen.insert(fingerprint(&v).unwrap()));
        }
    }

    #[test]
    fn display_is_32_hex_chars() {
        assert_eq!(fingerprint(&[1.0]).unwrap().to_string().len(), 32);
    }
}
