# cashvec (Python)

Vector fingerprinting (SimHash, random-hyperplane LSH) and semantic deduplication for embeddings. Python bindings for the Rust [CashVec](https://github.com/) library.

```python
import numpy as np
import cashvec

# SimHash: Hamming distance approximates angle
h = cashvec.SimHasher(dim=384, bits=256)
a, b = h.hash(emb_a), h.hash(emb_b)
a.hamming_distance(b), a.estimated_cosine(b)

# Similarity search
index = cashvec.VectorIndex(dim=384)
index.insert_many(ids, matrix)                     # (n, 384) float32/float64 array
index.search(query, k=10, threshold=0.95)          # [(id, cosine), ...]
index.search_many(queries, k=10, threshold=0.95)   # batch, GIL released

# Deduplication (reports; never deletes)
engine = cashvec.DedupEngine(dim=384, threshold=0.95)
for m in engine.check_and_insert("doc-1", emb):
    print(m.id, m.score, m.exact)
```

Inputs may be numpy `float32`/`float64` arrays (1-D for single vectors, 2-D row-per-vector for batches) or plain lists. Errors raise `cashvec.CashVecError`, a subclass of `ValueError`. Search is approximate (LSH candidates verified by exact cosine). `VectorIndex` and `DedupEngine` are not internally synchronized; guard shared instances with a lock when writing from multiple threads.

## Development

```sh
python3 -m venv .venv && source .venv/bin/activate
pip install maturin pytest numpy
cd cashvec-py && maturin develop --release
pytest
```

Build wheels with `maturin build --release` (one `abi3` wheel per platform covers Python ≥ 3.9).
