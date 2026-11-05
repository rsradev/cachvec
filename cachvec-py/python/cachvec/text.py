"""Text -> chunks -> vectors, feeding CachVec indexes.

Requires NumPy. The built-in :class:`HashingEmbedder` is lexical (it sees shared
words and word pairs, not meaning). For semantic similarity plug in a real
embedding model with :class:`CallableEmbedder`.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from ._cachvec import CachVecError

__all__ = ["Chunk", "TextChunker", "HashingEmbedder", "CallableEmbedder", "TextVectorizer"]

_WORD = re.compile(r"\S+")
_SENTENCE = re.compile(r"\S[^.!?\n]*(?:[.!?]+|(?=\n)|$)")
_TOKEN = re.compile(r"\w+")


@dataclass(frozen=True)
class Chunk:
    """A piece of the source text. ``text == source[start:end]``."""

    text: str
    index: int
    start: int
    end: int


class TextChunker:
    """Split text into overlapping chunks by length.

    Args:
        chunk_size: maximum chunk length, in ``unit``s (default 1000 characters).
        overlap: units shared between consecutive chunks (``0 <= overlap < chunk_size``).
        unit: ``"chars"`` (default), ``"words"`` or ``"sentences"``.
        snap_to_words: for ``unit="chars"``, avoid cutting words in half by moving
            chunk ends back (and starts forward) to whitespace. A word longer than
            the window is still cut. Ignored for other units.

    Chunks never exceed ``chunk_size`` units, and the last may be shorter. In
    ``"chars"`` mode chunk text is stripped of surrounding whitespace, and
    ``text[chunk.start:chunk.end] == chunk.text`` always holds. Whitespace-only
    text yields no chunks. Sentence splitting is a simple punctuation/newline
    rule and will split abbreviations such as "e.g."
    """

    def __init__(self, chunk_size: int = 1000, overlap: int = 200, unit: str = "chars",
                 snap_to_words: bool = True):
        if unit not in ("words", "sentences", "chars"):
            raise ValueError("unit must be 'chars', 'words' or 'sentences'")
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        if not 0 <= overlap < chunk_size:
            raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")
        self.chunk_size, self.overlap, self.unit = chunk_size, overlap, unit
        self.snap_to_words = snap_to_words

    def chunk(self, text: str) -> List[Chunk]:
        if not isinstance(text, str):
            raise TypeError("text must be a str")
        if self.unit == "chars":
            return self._chunk_chars(text)
        pattern = _WORD if self.unit == "words" else _SENTENCE
        spans = [m.span() for m in pattern.finditer(text)]
        n, step = len(spans), self.chunk_size - self.overlap
        out: List[Chunk] = []
        i = 0
        while i < n:
            j = min(i + self.chunk_size, n)
            start, end = spans[i][0], spans[j - 1][1]
            piece = text[start:end]
            if piece.strip():
                out.append(Chunk(piece, len(out), start, end))
            if j >= n:
                break
            i += step
        return out

    def _chunk_chars(self, text: str) -> List[Chunk]:
        n, size = len(text), self.chunk_size
        out: List[Chunk] = []
        i = 0
        while i < n:
            j = min(i + size, n)
            if self.snap_to_words and j < n and not text[j - 1].isspace() and not text[j].isspace():
                k = j - 1
                while k > i and not text[k].isspace():
                    k -= 1
                if k > i:
                    j = k
            piece = text[i:j]
            body = piece.strip()
            if body:
                start = i + (len(piece) - len(piece.lstrip()))
                out.append(Chunk(body, len(out), start, start + len(body)))
            if j >= n:
                break
            nxt = j - self.overlap
            if nxt <= i:
                nxt = j
            elif self.snap_to_words and not text[nxt - 1].isspace() and not text[nxt].isspace():
                m = nxt
                while m < j and not text[m].isspace():
                    m += 1
                if m < j:
                    nxt = m + 1
            i = nxt
        return out


@lru_cache(maxsize=1 << 18)
def _hash_feature(seed: int, token: str) -> int:
    d = hashlib.blake2b(f"{seed}:{token}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(d, "little")


class HashingEmbedder:
    """Dependency-free, deterministic bag-of-words embedder (feature hashing).

    Words and n-grams are hashed into ``dim`` buckets with a random sign, weighted
    by ``1 + log(count)``, then L2-normalized. Texts sharing vocabulary get high
    cosine similarity; paraphrases with different words do not. Text with no word
    characters embeds to the zero vector.
    """

    def __init__(self, dim: int = 384, ngram_range: Tuple[int, int] = (1, 2), seed: int = 0,
                 lowercase: bool = True):
        lo, hi = ngram_range
        if dim < 1 or lo < 1 or hi < lo:
            raise ValueError("need dim >= 1 and 1 <= ngram_range[0] <= ngram_range[1]")
        self.dim, self.ngram_range, self.seed, self.lowercase = dim, (lo, hi), seed, lowercase

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        lo, hi = self.ngram_range
        for r, text in enumerate(texts):
            tokens = _TOKEN.findall(text.lower() if self.lowercase else text)
            feats = Counter(
                " ".join(tokens[i:i + n])
                for n in range(lo, hi + 1)
                for i in range(len(tokens) - n + 1)
            )
            row = out[r]
            for feat, count in feats.items():
                h = _hash_feature(self.seed, feat)
                row[(h & 0xFFFFFFFF) % self.dim] += (1.0 if (h >> 63) & 1 else -1.0) * (1.0 + math.log(count))
            norm = float(np.linalg.norm(row))
            if norm > 0:
                row /= norm
        return out


class CallableEmbedder:
    """Wrap any ``fn(list[str]) -> array-like (n, dim)``, e.g. a model's ``encode``.

    >>> CallableEmbedder(SentenceTransformer("all-MiniLM-L6-v2").encode, dim=384)
    """

    def __init__(self, fn: Callable[[List[str]], object], dim: int):
        self.fn, self.dim = fn, dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if len(texts) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        arr = np.asarray(self.fn(list(texts)), dtype=np.float32)
        if arr.shape != (len(texts), self.dim):
            raise ValueError(f"embedder returned shape {arr.shape}, expected {(len(texts), self.dim)}")
        return arr


class TextVectorizer:
    """Chunk text and embed the chunks.

    Args:
        chunker: defaults to ``TextChunker()`` (1000 characters, 200 overlap).
        embedder: any object with ``dim`` and ``embed(list[str]) -> (n, dim)``;
            defaults to ``HashingEmbedder()``.

    Chunks that embed to the zero vector (e.g. punctuation only) are dropped
    because cosine similarity is undefined for them; the remaining chunks keep
    their original ``index``.
    """

    def __init__(self, chunker: Optional[TextChunker] = None, embedder: Optional[object] = None):
        self.chunker = chunker or TextChunker()
        self.embedder = embedder or HashingEmbedder()

    @property
    def dim(self) -> int:
        return self.embedder.dim

    def vectorize(self, text: str) -> Tuple[List[Chunk], np.ndarray]:
        """Return ``(chunks, matrix)`` with one row per chunk."""
        chunks = self.chunker.chunk(text)
        if not chunks:
            return [], np.zeros((0, self.dim), dtype=np.float32)
        matrix = np.asarray(self.embedder.embed([c.text for c in chunks]), dtype=np.float32)
        keep = np.linalg.norm(matrix, axis=1) > 0
        if not keep.all():
            chunks = [c for c, k in zip(chunks, keep) if k]
            matrix = matrix[keep]
        return chunks, matrix

    def embed_query(self, text: str) -> np.ndarray:
        """Embed ``text`` as a single vector (no chunking), for ``search``."""
        v = np.asarray(self.embedder.embed([text]), dtype=np.float32)[0]
        if not np.linalg.norm(v) > 0:
            raise CachVecError("text produced a zero vector (no embeddable content)")
        return v

    @staticmethod
    def chunk_id(doc_id: str, chunk: Chunk) -> str:
        return f"{doc_id}#{chunk.index}"

    def add(self, target, doc_id: str, text: str) -> List[Chunk]:
        """Insert every chunk into a ``VectorIndex`` or ``DedupEngine`` as ``"<doc_id>#<n>"``."""
        chunks, matrix = self.vectorize(text)
        if hasattr(target, "insert_many"):
            target.insert_many([self.chunk_id(doc_id, c) for c in chunks], matrix)
        else:
            for c, row in zip(chunks, matrix):
                target.insert(self.chunk_id(doc_id, c), row)
        return chunks

    def dedup(self, engine, doc_id: str, text: str) -> List[Tuple[Chunk, list]]:
        """``engine.check_and_insert`` for each chunk; returns ``(chunk, matches)`` pairs.

        An empty ``matches`` list means the chunk looked new. Nothing is deleted.
        """
        chunks, matrix = self.vectorize(text)
        return [(c, engine.check_and_insert(self.chunk_id(doc_id, c), row)) for c, row in zip(chunks, matrix)]
