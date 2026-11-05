"""CachVec: vector fingerprinting and semantic deduplication."""

from ._cachvec import (
    CachVecError,
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
    "CachVecError",
    "DedupEngine",
    "DuplicateMatch",
    "Fingerprint",
    "SimHasher",
    "VectorHasher",
    "VectorIndex",
    "cosine_similarity",
    "exact_fingerprint",
]

try:  # text helpers need NumPy; the core API does not
    from .text import (  # noqa: F401
        Chunk,
        CallableEmbedder,
        HashingEmbedder,
        TextChunker,
        TextVectorizer,
    )

    __all__ += ["Chunk", "CallableEmbedder", "HashingEmbedder", "TextChunker", "TextVectorizer"]
except ImportError:  # numpy not installed
    pass
