# CashVec

Vector fingerprinting and semantic deduplication for embeddings. Pure Rust, **zero dependencies** (std only), licensed MIT or Apache-2.0.

```rust
use cashvec::{DedupConfig, DedupEngine, VectorHasher};

let fp = VectorHasher::new(128).hash(&[0.12, 0.45, 0.78, 0.21])?;   // SimHash

let mut engine = DedupEngine::new(DedupConfig::new(4))?;
engine.insert("doc-1", &[0.12, 0.45, 0.78, 0.21])?;
let dups = engine.find_similar(&[0.12, 0.45, 0.78, 0.22], 0.95)?;   // [DuplicateMatch { id, score, exact }]
```

## What's in the box

| Component | Purpose |
|---|---|
| `exact_fingerprint` | Deterministic 128-bit hash of vector *data* (`-0.0` canonicalized, dimension included). Not semantic. |
| `VectorHasher` / `SimHasher` | Binary SimHash signatures; Hamming distance ↔ angle. |
| `Lsh` / `LshConfig` | Multi-table random-hyperplane LSH, seeded and deterministic. |
| `VectorIndex` | In-memory index: insert, remove, get by ID / fingerprint, LSH candidates, verified top-K search. |
| `DedupEngine` | Reports duplicate candidates with scores. **Never deletes anything.** |

## How search works

1. **Candidates** — the query is hashed into `L` tables of `k` bits; vectors sharing a bucket in any table are candidates.
2. **Verification** — every candidate is scored with exact cosine similarity and kept only if `score >= threshold`.
3. **Ordering** — score descending, ties broken by ID, so results are deterministic.

Shared buckets and equal fingerprints are never treated as proof of similarity.

### Recall model

Two vectors at angle θ agree on one hyperplane bit with probability `1 − θ/π`. They share a bucket in one table with probability `p^k` and in at least one of `L` tables with probability `1 − (1 − p^k)^L`. Defaults are `L = 16`, `k = 12`:

| cosine | per-bit agree | candidate probability |
|---|---|---|
| 0.99 | 0.955 | 1.00 |
| 0.95 | 0.899 | 0.995 |
| 0.90 | 0.856 | 0.933 |
| 0.80 | 0.795 | 0.65 |
| 0.00 | 0.500 | 0.004 |

Tune `with_tables` / `with_bits_per_table` to trade recall against candidate count. Search can miss neighbours that share no bucket; `search_brute_force` gives ground truth.

## SimHash: Hamming distance vs. angle

Each bit differs with probability `θ/π`, so `E[hamming] = bits · θ/π` and `cos θ ≈ cos(π · d / bits)` (`BinaryFingerprint::estimated_cosine`). The estimate is noisy (std ≈ `sqrt(p(1−p)/bits)` in `d/bits`); use more bits for accuracy and verify with real cosine when precision matters.

## Determinism

Hyperplane coordinates are a pure integer function of `(seed, plane, i)` (no `libm`, no platform RNG), so fingerprints and bucket keys are identical across platforms and runs. Float dot products use a fixed 4-lane summation order; no fused multiply-add is introduced.

## Errors

No panics on user input. Everything returns `Result<_, CashVecError>`: empty vectors, NaN/∞, dimension mismatches, zero-norm vectors, duplicate IDs, bad config and bad thresholds.

## Benchmarks

```
cargo bench --bench compare                          # 10k × 128
cargo bench --bench compare -- --n 20000 --dim 768
cargo bench --bench compare -- --full                # 10k/100k/1M × 128/384/768/1536 (skips combos needing >8 GiB)
```

Synthetic seeded data; queries are noisy copies of indexed vectors (cosine ≈ 0.97). Measured on the development machine (macOS, release build, single thread), default config (16 tables × 12 bits), threshold 0.9, top-10:

| n | dim | LSH search | brute force | speedup | recall@10 | candidates/query |
|---|---|---|---|---|---|---|
| 10,000 | 128 | 14.7 µs | 370.6 µs | 25× | 1.000 | 49.7 (0.5%) |
| 20,000 | 768 | 98.7 µs | 4,689.9 µs | 47× | 1.000 | 82.2 (0.4%) |

Other measurements at the same settings: SimHash (128 bit) takes 28.8 µs/vector at dim 128 and 135 µs at dim 768; insertion runs at ~117k vectors/s (dim 128) and ~22k/s (dim 768). Index memory is an estimate, about 3× the raw vectors at dim 128 (14.7 vs 4.9 MiB at 10k), mostly the 16 bucket tables. The 100k and 1M configurations have **not** been run yet, so no claims are made for them. Treat these numbers as indicative for one machine, and rerun the benchmark on yours.

## Python

Bindings live in `cashvec-py/` (PyO3 + maturin); see `cashvec-py/README.md`. A runnable tour is in `notebooks/cashvec_demo.ipynb`:

```sh
python3 -m venv .venv && source .venv/bin/activate
pip install maturin numpy matplotlib jupyterlab
(cd cashvec-py && maturin develop --release)
jupyter lab notebooks/cashvec_demo.ipynb
```

## Status

Version 0.1.0 implements the MVP scope in `SPEC.md`. Not yet done: SIMD, multithreading, persistence, Python bindings.

## License

MIT or Apache-2.0, at your option (`LICENSE-MIT`, `LICENSE-APACHE`).
