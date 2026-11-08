# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

CachVec is an open-source library for vector fingerprinting and semantic deduplication of embeddings (exact fingerprints, SimHash, random-hyperplane LSH, in-memory index, dedup engine). The core is a Rust crate; `cachvec-py/` holds PyO3/maturin Python bindings. `SPEC.md` has the full spec and `README.md` documents the Python API.

## Commands

```sh
cargo build
cargo test                                  # unit + doctests + tests/{integration,properties,accuracy}.rs
cargo test --test accuracy                  # one integration test file
cargo test <name>                           # one test by substring (e.g. cargo test cosine_symmetric)
cargo clippy --all-targets
cargo fmt
cargo run --example dedup                   # examples/{basic,dedup}.rs
cargo bench --bench compare                 # LSH vs brute force, 10k vectors
cargo bench --bench compare -- --full       # 10k/100k/1M x 128/384/768/1536
cargo bench --bench compare -- --n 50000 --dim 384
```

Python bindings (the repo has a `.venv/`; `cachvec-py` is a workspace member and needs PyO3/numpy crates, unlike the core):

```sh
pip install maturin numpy pytest
(cd cachvec-py && maturin develop --release)
pytest cachvec-py/tests                     # single test: pytest cachvec-py/tests/test_cachvec.py -k name
```

## Hard constraints

- The core `cachvec` crate has **zero dependencies** (Rust std only). Do not add crates to the root `[dependencies]`; even RNG, hashing and vector math are implemented in-tree (`src/rng.rs`, `src/fingerprint/exact.rs`, `src/vector/`). Dependencies are allowed only in `cachvec-py`.
- No panics on invalid user input: validate and return `CachVecError` (`src/error.rs`) — NaN/Inf, wrong dimension, zero norm, norm overflow, bad threshold.
- Deterministic output: same seed + same input must give the same fingerprints, buckets and result ordering (ties broken by ID).
- No performance claims without reproducible numbers from `benches/compare.rs`.
- `DedupEngine` only **reports** duplicate candidates and scores; it never deletes.

## Architecture

Search is deliberately two-stage, and this separation is the key invariant: **LSH buckets and equal exact fingerprints only produce candidates; every returned hit is re-verified with exact cosine similarity.** Never treat a shared bucket or fingerprint as proof of similarity.

- `vector/` — validation (`validate_dimension`, finite checks) and raw ops (`dot_unchecked`, `norm`, normalize, Euclidean). Callers validate first, then use the `_unchecked` fast paths.
- `fingerprint/` — three distinct things, don't conflate them: `exact` (128-bit hash of canonicalized vector data, incl. dimension and -0.0 handling; identifies data, not meaning), `simhash` (`SimHasher`/`BinaryFingerprint`; Hamming distance approximates angle), `lsh` (`HyperplaneSet` + `Lsh`: `num_tables` tables × `bits_per_table` bits, planes generated deterministically from `seed`). `VectorHasher` is the simple public facade.
- `index/` — `VectorIndex` stores vectors contiguously in one `f32` arena with parallel `ids`/`inv_norms`/`exact`/`keys` arrays and `u32` slots; removal swap-removes the last slot into the hole, so remapping `id_map`, `exact_map` and every `BucketTable` entry must stay in sync on insert/remove. Inserts must leave the index unchanged on failure. `IndexConfig` is a type alias for `LshConfig`.
- `dedup/` — `DedupEngine` wraps `VectorIndex` and adds exact-duplicate checks plus `find_similar` / `check_and_insert` with a configurable default threshold.
- `similarity/` — cosine and Hamming primitives.
- `cachvec-py/src/lib.rs` wraps the Rust types as the `cachvec._cachvec` extension module; `python/cachvec/__init__.py` re-exports it, and `text.py` (NumPy-only, imported in a try/except) adds chunking/embedder helpers in pure Python.

Tests: `tests/common/mod.rs` provides a seeded xorshift RNG and synthetic data shared by the integration, property and accuracy tests (property tests are hand-rolled with that RNG, not proptest).

## Roadmap context

`Cargo.toml` is at 0.1.0 and the code already covers the SPEC's phases 1–5 plus Python bindings. Remaining spec items: parallelization experiments and the full benchmark/performance report (phase 6), stable API and release prep (phase 7). Future extensions (persistence, SIMD, multithreading, quantization) should be kept possible but are out of scope for the MVP.
