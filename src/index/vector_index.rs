//! In-memory vector index with LSH candidate retrieval.

use super::buckets::BucketTable;
use crate::error::{CachVecError, Result};
use crate::fingerprint::exact::{canonical_eq, fingerprint, ExactFingerprint};
use crate::fingerprint::lsh::{Lsh, LshConfig};
use crate::vector::operations::{dot_unchecked, norm};
use crate::vector::validation::validate_dimension;
use std::collections::HashMap;

/// Index configuration: dimension, LSH shape and seed.
pub type IndexConfig = LshConfig;

/// A verified search hit.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    /// ID given at insertion.
    pub id: String,
    /// Exact cosine similarity with the query.
    pub score: f32,
}

/// Validates a similarity threshold (finite, within `[-1, 1]`).
pub fn check_threshold(t: f32) -> Result<()> {
    if t.is_nan() || !(-1.0..=1.0).contains(&t) {
        Err(CachVecError::InvalidThreshold)
    } else {
        Ok(())
    }
}

/// Validated query data computed once per call.
struct Prepared {
    inv_norm: f32,
}

/// In-memory index of fixed-dimension vectors.
///
/// Vectors live contiguously in one arena. Removal swap-removes the last slot
/// into the hole, so slots stay dense and memory is reclaimed immediately.
///
/// Search is two-stage: LSH buckets yield *candidates* (cheap, approximate),
/// then each candidate is verified with exact cosine similarity. Shared
/// buckets or equal exact fingerprints never imply similarity by themselves.
#[derive(Debug, Clone)]
pub struct VectorIndex {
    dim: usize,
    lsh: Lsh,
    data: Vec<f32>,
    inv_norms: Vec<f32>,
    ids: Vec<String>,
    exact: Vec<ExactFingerprint>,
    keys: Vec<u64>,
    id_map: HashMap<String, u32>,
    exact_map: HashMap<ExactFingerprint, Vec<u32>>,
    tables: Vec<BucketTable>,
}

impl VectorIndex {
    /// Creates an empty index.
    pub fn new(config: IndexConfig) -> Result<Self> {
        let lsh = Lsh::new(config)?;
        Ok(Self {
            dim: lsh.config().dim,
            tables: vec![BucketTable::new(); lsh.config().num_tables],
            lsh,
            data: Vec::new(),
            inv_norms: Vec::new(),
            ids: Vec::new(),
            exact: Vec::new(),
            keys: Vec::new(),
            id_map: HashMap::new(),
            exact_map: HashMap::new(),
        })
    }

    /// The configuration.
    pub fn config(&self) -> &IndexConfig {
        self.lsh.config()
    }

    /// Vector dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Number of indexed vectors.
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// True if nothing is indexed.
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// True if `id` is indexed.
    pub fn contains(&self, id: &str) -> bool {
        self.id_map.contains_key(id)
    }

    fn prepare(&self, v: &[f32]) -> Result<Prepared> {
        validate_dimension(v, self.dim)?;
        let n = norm(v);
        if !n.is_finite() {
            return Err(CachVecError::NormOverflow);
        }
        if n == 0.0 {
            return Err(CachVecError::ZeroNorm);
        }
        Ok(Prepared { inv_norm: 1.0 / n })
    }

    /// Inserts a vector under `id`.
    ///
    /// Fails if `id` exists, the vector is invalid (wrong dimension, NaN/inf,
    /// zero norm), or the index is full. The index is unchanged on failure.
    pub fn insert(&mut self, id: impl Into<String>, v: &[f32]) -> Result<()> {
        let id = id.into();
        let prep = self.prepare(v)?;
        if self.id_map.contains_key(&id) {
            return Err(CachVecError::DuplicateId(id));
        }
        let slot = u32::try_from(self.ids.len()).map_err(|_| CachVecError::CapacityExceeded)?;
        if slot == u32::MAX {
            return Err(CachVecError::CapacityExceeded);
        }
        let fp = fingerprint(v)?;
        let start = self.keys.len();
        self.lsh.keys_unchecked(v, &mut self.keys);
        for (t, table) in self.tables.iter_mut().enumerate() {
            table.insert(self.keys[start + t], slot);
        }
        self.exact_map.entry(fp).or_default().push(slot);
        self.exact.push(fp);
        self.data.extend_from_slice(v);
        self.inv_norms.push(prep.inv_norm);
        self.id_map.insert(id.clone(), slot);
        self.ids.push(id);
        Ok(())
    }

    /// Removes `id`. Returns whether it was present.
    pub fn remove(&mut self, id: &str) -> bool {
        let Some(slot) = self.id_map.remove(id) else {
            return false;
        };
        let s = slot as usize;
        let last = self.ids.len() - 1;
        let nt = self.tables.len();

        for t in 0..nt {
            self.tables[t].remove(self.keys[s * nt + t], slot);
        }
        if let Some(v) = self.exact_map.get_mut(&self.exact[s]) {
            v.retain(|x| *x != slot);
            if v.is_empty() {
                self.exact_map.remove(&self.exact[s]);
            }
        }

        if s != last {
            let last_slot = last as u32;
            for t in 0..nt {
                self.tables[t].replace(self.keys[last * nt + t], last_slot, slot);
            }
            if let Some(v) = self.exact_map.get_mut(&self.exact[last]) {
                if let Some(x) = v.iter_mut().find(|x| **x == last_slot) {
                    *x = slot;
                }
            }
            if let Some(x) = self.id_map.get_mut(&self.ids[last]) {
                *x = slot;
            }
            self.data.copy_within(last * self.dim..(last + 1) * self.dim, s * self.dim);
            self.keys.copy_within(last * nt..(last + 1) * nt, s * nt);
        }
        self.data.truncate(last * self.dim);
        self.keys.truncate(last * nt);
        self.ids.swap_remove(s);
        self.exact.swap_remove(s);
        self.inv_norms.swap_remove(s);
        true
    }

    fn vector_at(&self, slot: usize) -> &[f32] {
        &self.data[slot * self.dim..(slot + 1) * self.dim]
    }

    /// The stored vector for `id`.
    pub fn get(&self, id: &str) -> Option<&[f32]> {
        self.id_map.get(id).map(|s| self.vector_at(*s as usize))
    }

    /// The exact fingerprint stored for `id`.
    pub fn fingerprint_of(&self, id: &str) -> Option<ExactFingerprint> {
        self.id_map.get(id).map(|s| self.exact[*s as usize])
    }

    /// IDs whose stored fingerprint equals `fp`, sorted. A fingerprint match
    /// is strong evidence but not proof; see [`find_exact`](Self::find_exact).
    pub fn find_by_fingerprint(&self, fp: &ExactFingerprint) -> Vec<&str> {
        let mut out: Vec<&str> = self
            .exact_map
            .get(fp)
            .map(|v| v.iter().map(|s| self.ids[*s as usize].as_str()).collect())
            .unwrap_or_default();
        out.sort_unstable();
        out
    }

    /// IDs of stored vectors that are component-wise identical to `v`
    /// (after `-0.0` canonicalization), sorted. Verified, so hash collisions
    /// cannot produce false matches.
    pub fn find_exact(&self, v: &[f32]) -> Result<Vec<&str>> {
        validate_dimension(v, self.dim)?;
        let fp = fingerprint(v)?;
        let mut out: Vec<&str> = self
            .exact_map
            .get(&fp)
            .into_iter()
            .flatten()
            .filter(|s| canonical_eq(self.vector_at(**s as usize), v))
            .map(|s| self.ids[*s as usize].as_str())
            .collect();
        out.sort_unstable();
        Ok(out)
    }

    fn candidate_slots(&self, v: &[f32]) -> Vec<u32> {
        let mut keys = Vec::with_capacity(self.tables.len());
        self.lsh.keys_unchecked(v, &mut keys);
        let mut slots: Vec<u32> = Vec::new();
        for (table, key) in self.tables.iter().zip(keys) {
            slots.extend_from_slice(table.get(key));
        }
        slots.sort_unstable();
        slots.dedup();
        slots
    }

    /// IDs sharing at least one LSH bucket with `v` (unverified), sorted.
    pub fn candidates(&self, v: &[f32]) -> Result<Vec<&str>> {
        self.prepare(v)?;
        let mut out: Vec<&str> = self
            .candidate_slots(v)
            .into_iter()
            .map(|s| self.ids[s as usize].as_str())
            .collect();
        out.sort_unstable();
        Ok(out)
    }

    fn score(&self, slot: usize, q: &[f32], q_inv: f32) -> f32 {
        (dot_unchecked(q, self.vector_at(slot)) * q_inv * self.inv_norms[slot]).clamp(-1.0, 1.0)
    }

    fn finish(&self, mut hits: Vec<(f32, usize)>, k: usize) -> Vec<SearchResult> {
        hits.sort_unstable_by(|a, b| {
            b.0.total_cmp(&a.0)
                .then_with(|| self.ids[a.1].cmp(&self.ids[b.1]))
        });
        hits.truncate(k);
        hits.into_iter()
            .map(|(score, s)| SearchResult {
                id: self.ids[s].clone(),
                score,
            })
            .collect()
    }

    /// Approximate top-`k` search: LSH candidates verified by exact cosine
    /// similarity, keeping those with `score >= threshold`.
    ///
    /// Results are sorted by score descending, then ID ascending, so ties are
    /// deterministic. May miss true neighbours that share no bucket with the
    /// query (see the README for recall estimates); it never returns a result
    /// below `threshold`.
    pub fn search(&self, query: &[f32], k: usize, threshold: f32) -> Result<Vec<SearchResult>> {
        check_threshold(threshold)?;
        let prep = self.prepare(query)?;
        if k == 0 {
            return Ok(Vec::new());
        }
        let hits = self
            .candidate_slots(query)
            .into_iter()
            .map(|s| (self.score(s as usize, query, prep.inv_norm), s as usize))
            .filter(|(sc, _)| *sc >= threshold)
            .collect();
        Ok(self.finish(hits, k))
    }

    /// Exhaustive top-`k` search over every vector (ground truth for
    /// benchmarking recall). Same ordering and threshold semantics as
    /// [`search`](Self::search).
    pub fn search_brute_force(
        &self,
        query: &[f32],
        k: usize,
        threshold: f32,
    ) -> Result<Vec<SearchResult>> {
        check_threshold(threshold)?;
        let prep = self.prepare(query)?;
        if k == 0 {
            return Ok(Vec::new());
        }
        let hits = (0..self.len())
            .map(|s| (self.score(s, query, prep.inv_norm), s))
            .filter(|(sc, _)| *sc >= threshold)
            .collect();
        Ok(self.finish(hits, k))
    }

    /// Estimated heap usage in bytes (capacity-based, excludes allocator
    /// overhead and `String` heap headers beyond their length).
    pub fn memory_usage(&self) -> usize {
        self.lsh.memory_usage()
            + self.data.capacity() * 4
            + self.inv_norms.capacity() * 4
            + self.keys.capacity() * 8
            + self.exact.capacity() * 16
            + self.ids.iter().map(|s| 24 + s.capacity()).sum::<usize>()
            + self.id_map.capacity() * (24 + 4 + 1)
            + self.id_map.keys().map(|s| s.capacity()).sum::<usize>()
            + self.exact_map.capacity() * (16 + 24 + 1)
            + self.exact_map.values().map(|v| v.capacity() * 4).sum::<usize>()
            + self.tables.iter().map(BucketTable::memory_usage).sum::<usize>()
    }

    /// Per-table `(non-empty buckets, largest bucket)`, for diagnostics.
    pub fn bucket_stats(&self) -> Vec<(usize, usize)> {
        self.tables
            .iter()
            .map(|t| (t.bucket_count(), t.max_bucket_len()))
            .collect()
    }
}
