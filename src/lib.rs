//! # CashVec
//!
//! Vector fingerprinting and semantic deduplication for embeddings, written
//! against the Rust standard library only.
//!
//! * [`fingerprint::exact`]: deterministic 128-bit fingerprints of vector data.
//! * [`fingerprint::simhash`]: binary SimHash signatures; Hamming distance
//!   approximates angular distance.
//! * [`fingerprint::lsh`]: multi-table random-hyperplane LSH.
//! * [`index::VectorIndex`]: in-memory index with LSH candidate retrieval and
//!   exact cosine verification.
//! * [`dedup::DedupEngine`]: duplicate *reporting* (never deletion).
//!
//! ```
//! use cashvec::{DedupConfig, DedupEngine, VectorHasher};
//!
//! let hasher = VectorHasher::new(128);
//! let fp = hasher.hash(&[0.12, 0.45, 0.78, 0.21]).unwrap();
//! assert_eq!(fp.bits(), 128);
//!
//! let mut engine = DedupEngine::new(DedupConfig::new(4)).unwrap();
//! engine.insert("doc-1", &[0.12, 0.45, 0.78, 0.21]).unwrap();
//! let dups = engine.find_similar(&[0.12, 0.45, 0.78, 0.22], 0.95).unwrap();
//! assert_eq!(dups[0].id, "doc-1");
//! ```

pub mod dedup;
pub mod error;
pub mod fingerprint;
pub mod index;
pub mod rng;
pub mod similarity;
pub mod vector;

pub use dedup::{DedupConfig, DedupEngine, DuplicateMatch};
pub use error::{CashVecError, Result};
pub use fingerprint::{
    exact_fingerprint, BinaryFingerprint, ExactFingerprint, ExactFingerprinter, HyperplaneSet, Lsh,
    LshConfig, SimHasher, VectorHasher,
};
pub use index::{IndexConfig, SearchResult, VectorIndex};
pub use similarity::{cosine_similarity, hamming_distance};
pub use vector::Vector;
