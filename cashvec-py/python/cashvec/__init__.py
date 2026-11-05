"""CashVec: vector fingerprinting and semantic deduplication."""

from ._cashvec import (
    CashVecError,
    DedupEngine,
    DuplicateMatch,
    Fingerprint,
    SimHasher,
    VectorHasher,
    VectorIndex,
    cosine_similarity,
    exact_fingerprint,
)

__all__ = [
    "CashVecError",
    "DedupEngine",
    "DuplicateMatch",
    "Fingerprint",
    "SimHasher",
    "VectorHasher",
    "VectorIndex",
    "cosine_similarity",
    "exact_fingerprint",
]
