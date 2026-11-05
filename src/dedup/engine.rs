//! High-level semantic deduplication.

use crate::error::Result;
use crate::index::{check_threshold, IndexConfig, VectorIndex};

/// Deduplication settings.
#[derive(Debug, Clone, PartialEq)]
pub struct DedupConfig {
    /// Index shape (dimension, tables, bits, seed).
    pub index: IndexConfig,
    /// Default cosine threshold for [`DedupEngine::find_similar_default`].
    pub threshold: f32,
}

impl DedupConfig {
    /// Default threshold 0.95 on top of `IndexConfig::new(dim)`.
    pub fn new(dim: usize) -> Self {
        Self {
            index: IndexConfig::new(dim),
            threshold: 0.95,
        }
    }

    /// Sets the default threshold.
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }

    /// Replaces the index configuration.
    pub fn with_index(mut self, index: IndexConfig) -> Self {
        self.index = index;
        self
    }
}

/// A duplicate candidate. The engine reports; the caller decides what to do.
#[derive(Debug, Clone, PartialEq)]
pub struct DuplicateMatch {
    /// ID of the already-indexed document.
    pub id: String,
    /// Cosine similarity (exactly `1.0` for exact duplicates).
    pub score: f32,
    /// True if the stored vector is component-wise identical to the query.
    pub exact: bool,
}

/// Semantic deduplication engine. Never deletes anything on its own.
#[derive(Debug, Clone)]
pub struct DedupEngine {
    index: VectorIndex,
    threshold: f32,
}

impl DedupEngine {
    /// Creates an engine; validates the configuration and threshold.
    pub fn new(config: DedupConfig) -> Result<Self> {
        check_threshold(config.threshold)?;
        Ok(Self {
            index: VectorIndex::new(config.index)?,
            threshold: config.threshold,
        })
    }

    /// Indexes a document embedding under `id`.
    pub fn insert(&mut self, id: impl Into<String>, embedding: &[f32]) -> Result<()> {
        self.index.insert(id, embedding)
    }

    /// Reports duplicates of `embedding` at the default threshold, then
    /// indexes it under `id`. Fails (and indexes nothing) if `id` exists or
    /// the embedding is invalid.
    pub fn check_and_insert(
        &mut self,
        id: impl Into<String>,
        embedding: &[f32],
    ) -> Result<Vec<DuplicateMatch>> {
        let id = id.into();
        if self.index.contains(&id) {
            return Err(crate::error::CashVecError::DuplicateId(id));
        }
        let matches = self.find_similar(embedding, self.threshold)?;
        self.index.insert(id, embedding)?;
        Ok(matches)
    }

    /// IDs of documents whose embedding is identical to `embedding`, sorted.
    pub fn find_exact(&self, embedding: &[f32]) -> Result<Vec<String>> {
        Ok(self
            .index
            .find_exact(embedding)?
            .into_iter()
            .map(str::to_owned)
            .collect())
    }

    /// True if an identical embedding is already indexed.
    pub fn is_exact_duplicate(&self, embedding: &[f32]) -> Result<bool> {
        Ok(!self.index.find_exact(embedding)?.is_empty())
    }

    /// All indexed documents with cosine similarity `>= threshold`, best
    /// first (ties by ID). Exact duplicates are always included (score 1.0)
    /// even if the approximate stage would have missed them.
    pub fn find_similar(&self, embedding: &[f32], threshold: f32) -> Result<Vec<DuplicateMatch>> {
        check_threshold(threshold)?;
        let exact_ids = self.index.find_exact(embedding)?;
        let mut out: Vec<DuplicateMatch> = exact_ids
            .iter()
            .map(|id| DuplicateMatch {
                id: (*id).to_owned(),
                score: 1.0,
                exact: true,
            })
            .collect();
        for r in self.index.search(embedding, usize::MAX, threshold)? {
            if exact_ids.binary_search(&r.id.as_str()).is_err() {
                out.push(DuplicateMatch {
                    id: r.id,
                    score: r.score,
                    exact: false,
                });
            }
        }
        out.sort_by(|a, b| b.score.total_cmp(&a.score).then_with(|| a.id.cmp(&b.id)));
        Ok(out)
    }

    /// [`find_similar`](Self::find_similar) at the configured threshold.
    pub fn find_similar_default(&self, embedding: &[f32]) -> Result<Vec<DuplicateMatch>> {
        self.find_similar(embedding, self.threshold)
    }

    /// Removes a document from the index. Returns whether it was present.
    pub fn remove(&mut self, id: &str) -> bool {
        self.index.remove(id)
    }

    /// True if `id` is indexed.
    pub fn contains(&self, id: &str) -> bool {
        self.index.contains(id)
    }

    /// The stored embedding for `id`.
    pub fn get(&self, id: &str) -> Option<&[f32]> {
        self.index.get(id)
    }

    /// Number of indexed documents.
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// True if no documents are indexed.
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// The configured default threshold.
    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    /// Read access to the underlying index.
    pub fn index(&self) -> &VectorIndex {
        &self.index
    }
}
