# CashVec 0.1.0 — Implementation Spec

Records the decisions made where the project brief left room. The brief is `Claude.md`.

## Scope
All MVP items are implemented in 0.1.0 (roadmap phases 1–5 plus a benchmark harness). Not done: SIMD, parallelism, persistence, bindings; the 100k/1M benchmark runs; and a memory figure measured as RSS (the index reports a capacity-based estimate).

## Decisions
- **Dependencies:** none. Tests, properties and benches use hand-written seeded RNGs and `harness = false`.
- **Vectors:** `f32`, any dimension ≥ 1. NaN/∞ rejected. Zero-norm vectors are rejected by the index and cosine (angle undefined).
- **Exact fingerprint:** 128-bit non-cryptographic hash over `[dim, canonical f32 bit patterns]`; `-0.0 → +0.0`. Equal fingerprints are verified against the stored vectors in `find_exact`.
- **Projections:** each coordinate is a sum of four 16-bit uniform draws (Irwin–Hall), a pure integer function of `(seed, plane, i)`. Chosen over Box–Muller to avoid `ln`/`cos` platform differences. It is only approximately Gaussian, so the projection directions are not exactly uniform on the sphere.
- **Hash bit:** `[v · r >= 0]`.
- **LSH shape:** `L` tables × `k` bits (k ≤ 64, key is a `u64`). Defaults 16 × 12.
- **Index storage:** one contiguous `f32` arena; swap-remove on delete keeps slots dense. Original vectors are stored (not normalized) with a precomputed inverse norm.
- **Ordering:** score descending (`total_cmp`), ties by ID ascending.
- **Thresholds:** must be in `[-1, 1]`, else `InvalidThreshold`.
- **Dedup:** reports only. Exact duplicates are always included with score 1.0, independent of the LSH stage.
- **Capacity:** at most `u32::MAX - 1` vectors per index.
- **Thread safety:** types are plain data (`Send + Sync` when unshared-mutated); no internal locking yet.

## Known limitations
- Search is approximate: neighbours sharing no bucket with the query are missed. See README for the recall model.
- Insertion cost is `L·k` dot products of length `dim`; very high `L·k·dim` is expensive in time and memory (`L·k·dim·4` bytes of planes).
- Scores use `f32` accumulation; differences around 1e-6 versus an `f64` reference are expected.
