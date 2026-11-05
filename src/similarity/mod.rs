//! Similarity measures.

pub mod cosine;
pub mod hamming;

pub use cosine::cosine_similarity;
pub use hamming::{estimate_cosine, hamming_distance, max_hamming_for_cosine};
