CachVec

High-Performance Vector Fingerprinting & Semantic Deduplication Library

Project type: Open-source Rust library
Language: Rust
Initial version: 0.1.0
License: MIT or Apache-2.0
Primary domain: AI Infrastructure, RAG, Vector Search, Semantic Deduplication

⸻

1. Project Overview

CachVec is a lightweight, high-performance Rust library designed for vector fingerprinting, similarity hashing, and semantic deduplication.

The primary objective is to provide an efficient mechanism for identifying identical and semantically similar embedding vectors without requiring a dedicated vector database.

CachVec should be usable as an independent library in:

* Retrieval-Augmented Generation (RAG) pipelines.
* Semantic search applications.
* AI document processing systems.
* Embedding generation pipelines.
* Vector deduplication systems.
* Distributed AI infrastructure.

The library must initially be implemented entirely from scratch using Rust Standard Library.

No external hashing, vector processing, ANN, or machine learning libraries are permitted in the initial implementation.

2. Problem Statement

Modern RAG applications frequently generate and store large quantities of embedding vectors.

This creates several challenges:

1. Duplicate embeddings consume unnecessary storage.
2. Similar documents may generate highly similar embeddings.
3. Repeated embedding generation increases computational costs.
4. Vector databases may contain redundant information.
5. Traditional cryptographic hashes cannot identify similar vectors.

CachVec aims to address these problems through deterministic vector fingerprinting and approximate similarity hashing.

3. Core Objectives

The initial implementation must support:

1. Exact vector fingerprinting.
2. Approximate similarity hashing.
3. Semantic duplicate candidate detection.
4. In-memory vector indexing.
5. Similarity verification.
6. Top-K candidate retrieval.
7. Configurable hashing parameters.
8. Deterministic fingerprint generation.
9. Memory-efficient indexing.
10. Benchmarking against brute-force search.

4. Architecture

The library must follow a modular architecture.

cachvec/
│
├── Cargo.toml
├── README.md
├── LICENSE
├── SPEC.md
│
├── src/
│   ├── lib.rs
│   │
│   ├── vector/
│   │   ├── mod.rs
│   │   ├── operations.rs
│   │   └── validation.rs
│   │
│   ├── fingerprint/
│   │   ├── mod.rs
│   │   ├── exact.rs
│   │   ├── simhash.rs
│   │   └── lsh.rs
│   │
│   ├── similarity/
│   │   ├── mod.rs
│   │   ├── cosine.rs
│   │   └── hamming.rs
│   │
│   ├── index/
│   │   ├── mod.rs
│   │   ├── buckets.rs
│   │   └── vector_index.rs
│   │
│   └── dedup/
│       ├── mod.rs
│       └── engine.rs
│
├── tests/
│
├── benches/
│
└── examples/

5. Functional Requirements

5.1 Vector Representation

Implement a native vector representation.

Requirements:

* Support floating-point embeddings using f32 initially.
* Support arbitrary vector dimensions.
* Validate vector dimensions.
* Reject invalid numerical values (NaN and Infinity).
* Provide efficient vector operations.

Required operations:

* Dot product.
* Vector normalization.
* Euclidean distance.
* Cosine similarity.

5.2 Exact Fingerprinting

Implement exact vector fingerprinting.

Requirements:

* Deterministic output.
* Identical canonical vector representations must produce identical fingerprints.
* Different vector representations should produce different fingerprints with high probability.
* Define handling of negative zero and floating-point normalization.
* Include vector dimension in the fingerprint input.

Important distinction:

Exact fingerprinting must not be confused with semantic similarity hashing.

The exact fingerprint identifies vector data, not semantic meaning.

5.3 SimHash

Implement a SimHash algorithm for dense embedding vectors.

Requirements:

* Configurable fingerprint size.
* Deterministic projection generation.
* Binary fingerprint representation.
* Hamming distance calculation.
* Configurable similarity threshold.

The implementation must document the relationship between Hamming distance and angular similarity.

5.4 Random Projection LSH

Implement Random Hyperplane Locality-Sensitive Hashing.

For each projection:

h_i(v) = sign(v · r_i)

Where:

* v is the input embedding.
* r_i is a random projection vector.
* h_i is the resulting binary hash component.

Requirements:

* Configurable number of hyperplanes.
* Deterministic random projection generation from a seed.
* Binary hash generation.
* Support for multiple hash tables.
* Bucket-based indexing.
* Candidate retrieval.
* Configurable search parameters.

The implementation must ensure that vectors with high cosine similarity have a higher probability of sharing hash buckets.

5.5 Vector Index

Implement a native in-memory vector index.

Requirements:

* Insert vectors.
* Remove vectors.
* Retrieve vectors by ID.
* Retrieve vectors by fingerprint.
* Find candidate vectors through LSH buckets.
* Support multiple vectors per bucket.
* Handle hash collisions correctly.

The index must not assume that identical fingerprints always imply semantic equivalence.

5.6 Similarity Search

Implement approximate similarity search.

Requirements:

* Candidate retrieval through LSH.
* Exact cosine similarity verification.
* Top-K results.
* Configurable similarity threshold.
* Deterministic result ordering when scores are equal.

The system must distinguish between candidate retrieval and final similarity verification.

5.7 Deduplication Engine

Provide a high-level API for semantic deduplication.

Required operations:

* Insert document embeddings.
* Check for exact duplicates.
* Find semantically similar embeddings.
* Return matching document IDs.
* Configure similarity thresholds.
* Remove indexed documents.

The deduplication engine must not automatically delete documents.

It should return duplicate candidates and similarity scores.

6. Public API

The public API should be simple and ergonomic.

Illustrative interface:

use cachvec::VectorHasher;
fn main() {
    let hasher = VectorHasher::new(128);
    let embedding = vec![
        0.12, 0.45, 0.78, 0.21
    ];
    let fingerprint = hasher.hash(&embedding);
    println!("{:?}", fingerprint);
}

Similarity search:

let mut index = VectorIndex::new(config);
index.insert("doc-1", &embedding_a)?;
let results = index.search(
    &embedding_b,
    10,
    0.95
)?;

Deduplication:

let mut engine = DedupEngine::new(config);
engine.insert("doc-1", &embedding_a)?;
let duplicates = engine.find_similar(
    &embedding_b,
    0.95
)?;

These interfaces are illustrative. Final API design must prioritize consistency, error handling, usability, and future backward compatibility.

7. Performance Requirements

CachVec must prioritize:

* Low computational overhead.
* Low memory consumption.
* Efficient bucket lookup.
* Minimal unnecessary allocations.
* Efficient handling of large embedding collections.
* Thread-safe architecture in future versions.

All performance claims must be supported by reproducible benchmarks.

Initial benchmarks should include:

* 10,000 vectors.
* 100,000 vectors.
* 1,000,000 vectors.

Test dimensions:

* 128.	
* 384.	
* 768.	
* 1536.	

Measure:

* Fingerprint generation latency.
* Vector insertion throughput.
* Search latency.
* Memory consumption.
* Candidate recall.
* Precision of duplicate detection.

Compare approximate search performance against brute-force cosine similarity.

8. Non-Functional Requirements

Reliability

* No panics for invalid user input.
* Explicit error handling.
* Correct handling of hash collisions.
* Stable deterministic behavior.
* Unit and integration tests.

Maintainability

* Clear module boundaries.
* Documented public API.
* Minimal dependencies.
* Idiomatic Rust.
* Meaningful error types.

Portability

* Linux.
* macOS.
* Windows.

9. Implementation Constraints

Initial development must use Rust Standard Library only.

Prohibited in the initial implementation:

* FAISS.
* Annoy.
* HNSW libraries.
* Qdrant.
* Milvus.
* Weaviate.
* PostgreSQL.
* Redis.
* NumPy.
* External LSH implementations.
* External vector processing libraries.

The project must implement its own:

* Vector operations.
* Fingerprint generation.
* Random projection generation.
* LSH.
* Hamming distance.
* Bucket indexing.
* Similarity search.
* In-memory deduplication.

Cargo itself is permitted as the Rust build and package management system.

10. Testing Strategy

Implement comprehensive testing.

Required test categories:

Unit Tests

* Vector operations.
* Exact fingerprinting.
* SimHash.
* LSH.
* Hamming distance.
* Cosine similarity.
* Bucket management.

Integration Tests

* Insert and retrieve.
* Duplicate detection.
* Similarity search.
* Hash collisions.
* Different vector dimensions.
* Empty vectors.
* Invalid numerical values.

Property Tests

Verify:

* Deterministic hashing.
* Identical vectors produce identical exact fingerprints.
* Cosine similarity is symmetric.
* Normalized vectors have approximately unit length.
* Hamming distance is symmetric.

Accuracy Tests

Generate synthetic vector datasets with known angular similarities.

Measure:

* Candidate recall.
* Precision.
* False positives.
* False negatives.

11. Development Roadmap

Phase 1 — Mathematical Core

Version 0.1.0

* Vector representation.
* Vector validation.
* Dot product.
* Cosine similarity.
* Hamming distance.

Phase 2 — Fingerprinting

Version 0.2.0

* Exact fingerprinting.
* SimHash.
* Deterministic fingerprint generation.

Phase 3 — LSH Engine

Version 0.3.0

* Random projections.
* Hyperplane hashing.
* Multiple hash tables.
* Bucket indexing.

Phase 4 — Search Engine

Version 0.4.0

* Candidate retrieval.
* Similarity verification.
* Top-K search.
* Search configuration.

Phase 5 — Deduplication

Version 0.5.0

* Deduplication engine.
* Exact duplicate detection.
* Semantic duplicate detection.
* Document ID management.

Phase 6 — Performance

Version 0.6.0

* Memory optimization.
* Allocation optimization.
* Parallelization experiments.
* Benchmark suite.

Phase 7 — Production Readiness

Version 1.0.0

* Stable public API.
* Complete documentation.
* Error handling.
* Performance report.
* Release preparation.

12. Future Extensions

The following features are explicitly outside the initial MVP but should be considered in the architecture.

* Python bindings.
* PyO3 integration.
* Persistent storage.
* Binary index serialization.
* Distributed indexing.
* Incremental index updates.
* SIMD optimizations.
* Multithreaded indexing.
* Quantized embeddings.
* Integration with LangChain.
* Integration with LangGraph.
* Integration with popular vector databases.
* Custom similarity hashing algorithms.

13. Definition of Done

The initial MVP is considered complete when:

1. A vector can be fingerprinted deterministically.
2. Similar vectors can be mapped to approximate hash neighborhoods.
3. The index can store and retrieve vectors.
4. Similarity search returns verified candidates.
5. Duplicate detection works on synthetic datasets.
6. All core functionality has unit and integration tests.
7. Performance is benchmarked against brute-force search.
8. The library compiles without external dependencies.

14. Engineering Philosophy

CachVec should remain:

* Lightweight.
* Fast.
* Predictable.
* Deterministic.
* Independent of vector database implementations.
* Easy to integrate into existing AI infrastructure.

The project should prioritize algorithmic correctness before optimization.

No performance claims should be made without reproducible benchmarks.

The long-term objective is to develop CachVec into an independent vector fingerprinting and semantic deduplication engine for modern AI and RAG infrastructure.