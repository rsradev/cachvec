//! Random-hyperplane locality-sensitive hashing.
//!
//! For each hyperplane `r_i` the hash bit is `h_i(v) = [v · r_i >= 0]`.
//! Two vectors at angle `θ` agree on a bit with probability `1 - θ/π`.
//!
//! [`Lsh`] groups hyperplanes into `num_tables` independent tables of
//! `bits_per_table` bits each. The `bits_per_table` bits of a table form a
//! bucket key. Two vectors share a bucket in one table with probability
//! `(1 - θ/π)^k` and in at least one of `L` tables with probability
//! `1 - (1 - (1 - θ/π)^k)^L`, so high-cosine pairs are far more likely to
//! collide than dissimilar ones.
//!
//! Hyperplanes are generated deterministically from the seed.

use super::BinaryFingerprint;
use crate::error::{CashVecError, Result};
use crate::rng::{fill_plane, plane_key, DEFAULT_SEED};
use crate::vector::operations::dot_unchecked;
use crate::vector::validation::validate_dimension;

/// A fixed-dimension set of random hyperplanes.
#[derive(Debug, Clone)]
pub struct HyperplaneSet {
    dim: usize,
    count: usize,
    planes: Vec<f32>,
}

impl HyperplaneSet {
    /// Generates `count` planes of dimension `dim` from `seed`.
    pub fn new(dim: usize, count: usize, seed: u64) -> Result<Self> {
        if dim == 0 {
            return Err(CashVecError::InvalidConfig("dim must be >= 1".into()));
        }
        if count == 0 {
            return Err(CashVecError::InvalidConfig(
                "hyperplane count must be >= 1".into(),
            ));
        }
        let total = dim
            .checked_mul(count)
            .filter(|t| t.checked_mul(4).is_some() && *t <= (1usize << 32))
            .ok_or_else(|| CashVecError::InvalidConfig("dim * planes too large".into()))?;
        let mut planes = vec![0.0f32; total];
        for (p, chunk) in planes.chunks_exact_mut(dim).enumerate() {
            fill_plane(plane_key(seed, p as u64), chunk);
        }
        Ok(Self { dim, count, planes })
    }

    /// Vector dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Number of planes.
    pub fn count(&self) -> usize {
        self.count
    }

    /// Sign bit of `v` against plane `plane`. `v` must have length `dim`.
    #[inline]
    pub(crate) fn sign_bit(&self, plane: usize, v: &[f32]) -> bool {
        let p = &self.planes[plane * self.dim..(plane + 1) * self.dim];
        dot_unchecked(v, p) >= 0.0
    }

    /// Full binary signature of `v` (one bit per plane).
    pub fn signature(&self, v: &[f32]) -> Result<BinaryFingerprint> {
        validate_dimension(v, self.dim)?;
        let mut fp = BinaryFingerprint::zeros(self.count);
        for p in 0..self.count {
            if self.sign_bit(p, v) {
                fp.set_bit(p);
            }
        }
        Ok(fp)
    }

    /// Approximate heap size in bytes.
    pub fn memory_usage(&self) -> usize {
        self.planes.len() * std::mem::size_of::<f32>()
    }
}

/// Parameters of an [`Lsh`] instance (also used as the index configuration).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LshConfig {
    /// Vector dimension.
    pub dim: usize,
    /// Number of independent hash tables (`L`). More tables raise recall.
    pub num_tables: usize,
    /// Hyperplanes per table (`k`, 1..=64). More bits raise selectivity.
    pub bits_per_table: usize,
    /// Seed for deterministic hyperplane generation.
    pub seed: u64,
}

impl LshConfig {
    /// Defaults: 16 tables × 12 bits, which gives roughly 99% candidate
    /// recall at cosine 0.95 (see the README for the derivation).
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            num_tables: 16,
            bits_per_table: 12,
            seed: DEFAULT_SEED,
        }
    }

    /// Sets the number of tables.
    pub fn with_tables(mut self, num_tables: usize) -> Self {
        self.num_tables = num_tables;
        self
    }

    /// Sets the bits per table.
    pub fn with_bits_per_table(mut self, bits: usize) -> Self {
        self.bits_per_table = bits;
        self
    }

    /// Sets the seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Checks parameter ranges.
    pub fn validate(&self) -> Result<()> {
        if self.dim == 0 {
            return Err(CashVecError::InvalidConfig("dim must be >= 1".into()));
        }
        if !(1..=1024).contains(&self.num_tables) {
            return Err(CashVecError::InvalidConfig(
                "num_tables must be in 1..=1024".into(),
            ));
        }
        if !(1..=64).contains(&self.bits_per_table) {
            return Err(CashVecError::InvalidConfig(
                "bits_per_table must be in 1..=64".into(),
            ));
        }
        Ok(())
    }
}

/// Multi-table random-hyperplane LSH.
#[derive(Debug, Clone)]
pub struct Lsh {
    config: LshConfig,
    planes: HyperplaneSet,
}

impl Lsh {
    /// Builds the hyperplanes for `config`.
    pub fn new(config: LshConfig) -> Result<Self> {
        config.validate()?;
        let planes = HyperplaneSet::new(
            config.dim,
            config.num_tables * config.bits_per_table,
            config.seed,
        )?;
        Ok(Self { config, planes })
    }

    /// The configuration.
    pub fn config(&self) -> &LshConfig {
        &self.config
    }

    /// One bucket key per table. Validates `v`.
    pub fn keys(&self, v: &[f32]) -> Result<Vec<u64>> {
        validate_dimension(v, self.config.dim)?;
        let mut out = Vec::with_capacity(self.config.num_tables);
        self.keys_unchecked(v, &mut out);
        Ok(out)
    }

    /// Appends one key per table to `out`; `v` must already be validated.
    pub(crate) fn keys_unchecked(&self, v: &[f32], out: &mut Vec<u64>) {
        let k = self.config.bits_per_table;
        for t in 0..self.config.num_tables {
            let mut key = 0u64;
            for b in 0..k {
                if self.planes.sign_bit(t * k + b, v) {
                    key |= 1 << b;
                }
            }
            out.push(key);
        }
    }

    /// Approximate heap size in bytes.
    pub fn memory_usage(&self) -> usize {
        self.planes.memory_usage()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> LshConfig {
        LshConfig::new(8).with_tables(4).with_bits_per_table(8)
    }

    #[test]
    fn config_validation() {
        assert!(Lsh::new(LshConfig::new(0)).is_err());
        assert!(Lsh::new(cfg().with_tables(0)).is_err());
        assert!(Lsh::new(cfg().with_bits_per_table(0)).is_err());
        assert!(Lsh::new(cfg().with_bits_per_table(65)).is_err());
        assert!(Lsh::new(cfg()).is_ok());
    }

    #[test]
    fn keys_are_deterministic_and_seed_dependent() {
        let v = [0.3, -0.2, 0.9, 0.1, -0.7, 0.4, 0.0, 1.0];
        let a = Lsh::new(cfg()).unwrap().keys(&v).unwrap();
        let b = Lsh::new(cfg()).unwrap().keys(&v).unwrap();
        let c = Lsh::new(cfg().with_seed(99)).unwrap().keys(&v).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.len(), 4);
        assert_ne!(a, c);
    }

    #[test]
    fn scaling_does_not_change_keys() {
        let lsh = Lsh::new(cfg()).unwrap();
        let v = [0.3, -0.2, 0.9, 0.1, -0.7, 0.4, 0.2, 1.0];
        let w: Vec<f32> = v.iter().map(|x| x * 3.0).collect();
        assert_eq!(lsh.keys(&v).unwrap(), lsh.keys(&w).unwrap());
    }

    #[test]
    fn rejects_bad_vectors() {
        let lsh = Lsh::new(cfg()).unwrap();
        assert!(lsh.keys(&[1.0; 7]).is_err());
        assert!(lsh.keys(&[]).is_err());
        assert!(lsh.keys(&[f32::NAN; 8]).is_err());
    }
}
