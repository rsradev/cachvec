//! Vector fingerprinting: exact hashes, SimHash and random-hyperplane LSH.

pub mod exact;
pub mod lsh;
pub mod simhash;

pub use exact::{fingerprint as exact_fingerprint, ExactFingerprint, ExactFingerprinter};
pub use lsh::{HyperplaneSet, Lsh, LshConfig};
pub use simhash::{SimHasher, VectorHasher};

use crate::error::{CashVecError, Result};
use crate::similarity::hamming;
use std::fmt;

/// A fixed-length binary signature (one bit per hyperplane).
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct BinaryFingerprint {
    words: Vec<u64>,
    bits: usize,
}

impl BinaryFingerprint {
    /// All-zero signature of `bits` bits.
    pub(crate) fn zeros(bits: usize) -> Self {
        Self {
            words: vec![0; bits.div_ceil(64)],
            bits,
        }
    }

    #[inline]
    pub(crate) fn set_bit(&mut self, i: usize) {
        self.words[i / 64] |= 1 << (i % 64);
    }

    /// Number of bits.
    pub fn bits(&self) -> usize {
        self.bits
    }

    /// Packed words (bit `i` is bit `i % 64` of word `i / 64`; unused high
    /// bits of the last word are zero).
    pub fn words(&self) -> &[u64] {
        &self.words
    }

    /// Value of bit `i`, or `None` if out of range.
    pub fn bit(&self, i: usize) -> Option<bool> {
        (i < self.bits).then(|| self.words[i / 64] >> (i % 64) & 1 == 1)
    }

    /// Hamming distance to `other`; fails if the bit lengths differ.
    pub fn hamming_distance(&self, other: &Self) -> Result<u32> {
        if self.bits != other.bits {
            return Err(CashVecError::DimensionMismatch {
                expected: self.bits,
                actual: other.bits,
            });
        }
        hamming::hamming_distance(&self.words, &other.words)
    }

    /// Cosine similarity estimated from the Hamming distance
    /// (see [`simhash`] for accuracy).
    pub fn estimated_cosine(&self, other: &Self) -> Result<f32> {
        Ok(hamming::estimate_cosine(self.hamming_distance(other)?, self.bits))
    }

    /// True if the estimated cosine is at least `min_cosine`.
    pub fn is_similar_to(&self, other: &Self, min_cosine: f32) -> Result<bool> {
        if min_cosine.is_nan() || !(-1.0..=1.0).contains(&min_cosine) {
            return Err(CashVecError::InvalidThreshold);
        }
        let d = self.hamming_distance(other)?;
        Ok(d <= hamming::max_hamming_for_cosine(min_cosine, self.bits))
    }
}

impl fmt::Display for BinaryFingerprint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for w in self.words.iter().rev() {
            write!(f, "{w:016x}")?;
        }
        Ok(())
    }
}

impl fmt::Debug for BinaryFingerprint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BinaryFingerprint({} bits, 0x{self})", self.bits)
    }
}
