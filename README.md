# CachVec

Fast vector fingerprinting and semantic deduplication for embeddings, with no vector database required.

Find exact and near-duplicate embeddings in your RAG pipeline, skip repeated work, and keep your index clean. CachVec runs in-process, has no required dependencies (NumPy is optional), and gives deterministic results.

```python
import numpy as np
import cachvec

engine = cachvec.DedupEngine(dim=384, threshold=0.95)

for doc_id, embedding in documents:              # embedding: numpy array or list of floats
    duplicates = engine.check_and_insert(doc_id, embedding)
    for m in duplicates:
        print(f"{doc_id} ~ {m.id}  cosine={m.score:.3f}  exact={m.exact}")
```

CachVec **reports** duplicate candidates and their scores. It never deletes anything; you decide what to do with them.

## Install

```sh
pip install cachvec
```

> Not published to PyPI yet. Until then, see [Install from source](#install-from-source).

Requires Python 3.9+. NumPy is optional; plain lists work everywhere.

## Quickstart

### Find similar embeddings

```python
index = cachvec.VectorIndex(dim=384)
index.insert_many(ids, matrix)                       # matrix: (n, 384) float32/float64 array
index.insert("doc-new", embedding)

index.search(query, k=10, threshold=0.95)            # [("doc-12", 0.987), ("doc-7", 0.961), ...]
index.search_many(queries, k=10, threshold=0.95)     # one result list per row
```

Results are sorted best-first (ties broken by ID) and every hit has a verified cosine similarity at or above `threshold`.

### Deduplicate

```python
engine = cachvec.DedupEngine(dim=384, threshold=0.95)
engine.insert("doc-1", emb_a)

engine.is_exact_duplicate(emb_a)                     # True
engine.find_similar(emb_b)                           # [DuplicateMatch(id="doc-1", score=0.98, exact=False)]
engine.find_similar(emb_b, threshold=0.9)            # override the default threshold
engine.check_and_insert("doc-2", emb_b)              # report duplicates, then index it
engine.remove("doc-1")
```

### Fingerprints

```python
# Exact: identifies the vector's data (not its meaning). Same data -> same hex string.
cachvec.exact_fingerprint(embedding)                 # "0c309483e547e0d097e4f3bfb8926624"

# SimHash: similar vectors get similar bit signatures.
hasher = cachvec.SimHasher(dim=384, bits=256)
a, b = hasher.hash(emb_a), hasher.hash(emb_b)
a.hamming_distance(b)                                # e.g. 20 (out of 256 bits)
a.estimated_cosine(b)                                # approximate cosine similarity
a.is_similar_to(b, 0.9)
a.to_bytes()                                         # compact binary form, e.g. for storage

cachvec.VectorHasher(bits=128).hash(embedding)       # same, but works with any dimension
```

## How it works

1. **Candidates.** Each vector is hashed (random-hyperplane LSH) into several hash tables. Vectors that land in the same bucket as the query become candidates. This avoids comparing the query with every stored vector.
2. **Verification.** Every candidate is scored with exact cosine similarity and kept only if it meets your threshold.
3. **Ordering.** Results are sorted by score, then ID, so output is deterministic.

Sharing a bucket or a fingerprint is never treated as proof of similarity.

Search is **approximate**: a true neighbour that shares no bucket with the query can be missed. See [Tuning](#tuning).

## Tuning

`VectorIndex` and `DedupEngine` take two knobs:

| Parameter | Default | Effect |
|---|---|---|
| `num_tables` | 16 | More tables: higher recall, more candidates, more memory |
| `bits_per_table` | 12 | More bits: more selective buckets, fewer candidates, lower recall |

Theoretical chance that a stored vector is retrieved as a candidate, by its cosine similarity to the query (defaults; the notebook compares this curve with measurements):

| cosine | 0.99 | 0.95 | 0.90 | 0.80 | 0.50 | 0.00 |
|---|---|---|---|---|---|---|
| candidate | ~100% | ~99.5% | ~93% | ~65% | ~12% | ~0.4% |

Dense, near-duplicate detection (cosine ≥ 0.9) is the sweet spot. For looser similarity, increase `num_tables` or decrease `bits_per_table`. `index.search_brute_force(query)` gives exact results for comparison, and `index.candidates(query)` shows what the hash stage retrieves.

**SimHash accuracy.** Each signature bit differs with probability θ/π, so `cos θ ≈ cos(π · d / bits)`. The estimate is noisy; more bits make it tighter. Use it as a fast filter, and verify with a real cosine when precision matters.

## Performance

Measured in Python (`notebooks/cachvec_demo.ipynb`) on one development laptop: 20,000 random 384-dimensional vectors, 200 queries, cosine ≈ 0.96 to a planted neighbour, `threshold=0.9`, `k=10`, default settings.

| | CachVec | NumPy brute force |
|---|---|---|
| Search latency | 46 µs/query | 1,117 µs/query |
| Recall@10 | 1.000 (200/200) | exact |
| Candidates examined | ~88 of 20,000 | all |

Bulk insertion ran at about 41,000 vectors/s, and the index used about 61 MiB (the raw vectors are 29 MiB; the rest is hash tables). These are single runs on synthetic data, so treat them as indicative and rerun the notebook on your hardware and data. Results on real embeddings will differ, and I have not benchmarked 100k+ vectors from Python.

## Text: chunking and embedding

`TextChunker` splits text into overlapping chunks by length (characters, never cutting words in half), an embedder turns chunks into vectors, and `TextVectorizer` ties them to an index. (Requires NumPy.)

```python
from cachvec import TextChunker, HashingEmbedder, TextVectorizer

vec = TextVectorizer(
    TextChunker(chunk_size=1000, overlap=200),   # max 1000 characters per chunk, 200 shared
    HashingEmbedder(dim=384),
)

chunks, matrix = vec.vectorize(long_text)       # chunks[i].text / .start / .end  <->  matrix[i]

index = cachvec.VectorIndex(vec.dim)
vec.add(index, "report-2024", long_text)        # inserts ids "report-2024#0", "report-2024#1", ...

engine = cachvec.DedupEngine(vec.dim, threshold=0.95)
for chunk, matches in vec.dedup(engine, "doc-2", other_text):   # reports, never deletes
    if matches:
        print(chunk.index, "looks like", matches[0].id, matches[0].score)
```

**Embedders.** The built-in `HashingEmbedder` needs no model or download. It hashes words and word pairs, so it detects **shared wording** (copied or lightly edited text), not paraphrases. For semantic similarity, plug in a real model:

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
vec = TextVectorizer(embedder=cachvec.CallableEmbedder(model.encode, dim=384))
```

Any function that maps a list of strings to an `(n, dim)` array works. Chunks with no embeddable content (e.g. only punctuation) are dropped. Chunking is length-based by default; `unit="words"` and `unit="sentences"` are also available, and `snap_to_words=False` gives exact character cuts. Sentence splitting is a simple punctuation rule and will split abbreviations like "e.g.".

**Searching with a short query.** A query is usually only loosely similar to a passage (cosine around 0.5 to 0.7), which the default index settings are not tuned for; they find near-duplicates. For retrieval-style queries, build the index with `bits_per_table=6` (and keep `num_tables=16` or more), or use `index.search_brute_force(q, k, 0.0)` on small corpora. Embed the query with `vec.embed_query(text)`.

## Behavior and guarantees

- **Deterministic.** The same input and `seed` always give the same fingerprints and bucket assignments, across runs and platforms.
- **Scale invariant.** Hashing and search depend on direction, not length (cosine similarity).
- **Validated input.** NaN, infinity, empty vectors, zero vectors, and wrong dimensions raise `cachvec.CachVecError` (a subclass of `ValueError`). Nothing crashes the interpreter.
- **Input types.** 1-D `float32`/`float64` arrays or lists for single vectors; 2-D arrays (one vector per row) for batches.
- **Threading.** Search and hashing release the GIL. Indexes are not internally synchronized, so guard shared instances with a lock if you write from multiple threads.
- **In memory only.** There is no persistence yet.

## API summary

| Name | Purpose |
|---|---|
| `VectorIndex(dim, num_tables=16, bits_per_table=12, seed=None)` | `insert`, `insert_many`, `remove`, `get`, `search`, `search_many`, `search_brute_force`, `candidates`, `find_exact`, `fingerprint_of`, `memory_usage`, `len()`, `in` |
| `DedupEngine(dim, threshold=0.95, num_tables=16, bits_per_table=12, seed=None)` | `insert`, `check_and_insert`, `find_similar`, `find_exact`, `is_exact_duplicate`, `remove`, `get`, `len()`, `in` |
| `DuplicateMatch` | `.id`, `.score`, `.exact` |
| `SimHasher(dim, bits, seed=None)` | `hash`, `hash_many` |
| `VectorHasher(bits, seed=None)` | `hash` (any dimension) |
| `Fingerprint` | `hamming_distance`, `estimated_cosine`, `is_similar_to`, `to_bytes`, `bits`; hashable and comparable |
| `exact_fingerprint(v)`, `cosine_similarity(a, b)` | helpers |
| `TextChunker`, `Chunk`, `HashingEmbedder`, `CallableEmbedder`, `TextVectorizer` | text chunking and embedding (see above) |
| `CachVecError` | raised on invalid input |

`insert_many` stops at the first error; rows before it stay inserted.

## Try it

`notebooks/cachvec_demo.ipynb` is a runnable walkthrough with plots: SimHash accuracy, recall against theory, tuning, and a dedup simulation.

## Install from source

The core is compiled, so building from source needs a [Rust toolchain](https://rustup.rs).

```sh
git clone <repo-url> && cd cachvec
python3 -m venv .venv && source .venv/bin/activate
pip install maturin numpy matplotlib jupyterlab pytest
(cd cachvec-py && maturin develop --release)
pytest cachvec-py/tests
jupyter lab notebooks/cachvec_demo.ipynb
```

`maturin build --release` produces a wheel. One `abi3` wheel per platform covers Python 3.9 and later.

## License

MIT or Apache-2.0, at your option.
