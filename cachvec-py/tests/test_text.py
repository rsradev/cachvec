import numpy as np
import pytest

import cachvec
from cachvec import CallableEmbedder, HashingEmbedder, TextChunker, TextVectorizer

DOC_A = ("Rust is a systems programming language focused on safety and speed. "
         "It guarantees memory safety without a garbage collector. "
         "Many teams use Rust for performance critical services.")
DOC_B = ("Bananas are a popular fruit rich in potassium. They ripen quickly at room "
         "temperature. Smoothies often include bananas and yogurt.")


def test_word_chunks_overlap_and_offsets():
    text = " ".join(f"w{i}" for i in range(25))
    chunks = TextChunker(chunk_size=10, overlap=3, unit="words").chunk(text)
    assert [len(c.text.split()) for c in chunks] == [10, 10, 10, 4]
    assert chunks[0].text.split()[-3:] == chunks[1].text.split()[:3]
    for i, c in enumerate(chunks):
        assert c.index == i and text[c.start:c.end] == c.text
    covered = {w for c in chunks for w in c.text.split()}
    assert covered == set(text.split())


def test_no_trailing_overlap_only_chunk():
    chunks = TextChunker(chunk_size=5, overlap=2, unit="words").chunk(" ".join("x" * 1 for _ in range(8)))
    assert [len(c.text.split()) for c in chunks] == [5, 5]


def test_sentence_and_char_units():
    s = TextChunker(chunk_size=2, overlap=1, unit="sentences").chunk("One. Two! Three? Four.")
    assert [c.text for c in s] == ["One. Two!", "Two! Three?", "Three? Four."]
    c = TextChunker(chunk_size=4, overlap=1, unit="chars").chunk("abcdefghij")
    assert [x.text for x in c] == ["abcd", "defg", "ghij"]


def test_default_chunking_is_by_length():
    c = TextChunker()
    assert (c.unit, c.chunk_size, c.overlap) == ("chars", 1000, 200)
    text = ("lorem ipsum dolor sit amet " * 200).strip()
    chunks = c.chunk(text)
    assert len(chunks) > 5
    for ch in chunks:
        assert len(ch.text) <= 1000 and text[ch.start:ch.end] == ch.text
        assert not ch.text[0].isspace() and not ch.text[-1].isspace()
        assert ch.text.split()[0] in text.split() and ch.text.split()[-1] in text.split()  # no cut words
    assert chunks[0].end > chunks[1].start                 # consecutive chunks overlap
    assert chunks[-1].end == len(text)                     # whole text covered


def test_length_chunks_without_snapping_cut_exactly():
    text = "word " * 100
    for ch in TextChunker(chunk_size=50, overlap=10, snap_to_words=False).chunk(text):
        assert len(ch.text) <= 50
    snapped = TextChunker(chunk_size=50, overlap=10).chunk(text)
    assert all(set(ch.text.split()) == {"word"} for ch in snapped)


def test_length_chunking_covers_text_and_terminates_on_long_words():
    text = "x" * 2500 + " tail"
    chunks = TextChunker(chunk_size=1000, overlap=100).chunk(text)
    assert all(len(c.text) <= 1000 for c in chunks)
    assert "".join(c.text for c in chunks).count("x") >= 2500
    assert chunks[-1].text.endswith("tail")
    # overlap close to size must still make progress
    assert len(TextChunker(chunk_size=10, overlap=9).chunk("ab cd ef gh ij kl mn")) > 1


def test_chunker_edge_cases_and_validation():
    assert TextChunker().chunk("") == [] and TextChunker().chunk("  \n ") == []
    assert len(TextChunker(chunk_size=1000).chunk("short text")) == 1
    for bad in (dict(chunk_size=0), dict(chunk_size=5, overlap=5), dict(chunk_size=5, overlap=-1),
                dict(unit="bytes")):
        with pytest.raises(ValueError):
            TextChunker(**bad)
    with pytest.raises(TypeError):
        TextChunker().chunk(123)


def test_hashing_embedder_properties():
    e = HashingEmbedder(dim=256)
    m = e.embed([DOC_A, DOC_A, DOC_B, "!!!", ""])
    assert m.shape == (5, 256) and m.dtype == np.float32
    np.testing.assert_array_equal(m[0], m[1])                       # deterministic
    np.testing.assert_array_equal(m, HashingEmbedder(dim=256).embed([DOC_A, DOC_A, DOC_B, "!!!", ""]))
    assert abs(np.linalg.norm(m[0]) - 1) < 1e-5
    assert not m[3].any() and not m[4].any()                         # no tokens -> zero
    assert m[0] @ m[2] < 0.3                                         # unrelated
    near = e.embed([DOC_A.replace("speed", "performance")])[0]
    assert m[0] @ near > 0.8                                         # small edit -> similar
    assert not np.array_equal(e.embed(["hello world"]), HashingEmbedder(dim=256, seed=1).embed(["hello world"]))
    assert np.array_equal(e.embed(["Hello, WORLD"]), e.embed(["hello world"]))


def test_vectorizer_drops_empty_chunks_and_keeps_indices():
    v = TextVectorizer(TextChunker(chunk_size=3, overlap=0, unit="words"), HashingEmbedder(64))
    chunks, mat = v.vectorize("alpha beta gamma !!! ??? ... delta epsilon zeta")
    assert [c.index for c in chunks] == [0, 2] and mat.shape == (2, 64)
    empty_chunks, empty_mat = v.vectorize("")
    assert empty_chunks == [] and empty_mat.shape == (0, 64)
    with pytest.raises(cachvec.CachVecError):
        v.embed_query("!!!")


def test_end_to_end_search_over_chunked_documents():
    v = TextVectorizer(TextChunker(chunk_size=12, overlap=3, unit="words"), HashingEmbedder(384))
    # Query-to-passage cosines are low (~0.6), so use short buckets (6 bits);
    # the default 12-bit tables are tuned for near-duplicates.
    index = cachvec.VectorIndex(v.dim, num_tables=16, bits_per_table=6)
    chunks_a = v.add(index, "rust", DOC_A)
    v.add(index, "fruit", DOC_B)
    assert len(index) >= len(chunks_a) + 2
    q = v.embed_query("rust guarantees memory safety without a garbage collector")
    hits = index.search(q, k=3, threshold=0.2)
    assert hits and hits[0][0].startswith("rust#")
    assert all(not h[0].startswith("fruit#") for h in hits)
    assert hits[0][0] == index.search_brute_force(q, 1, 0.0)[0][0]


def test_dedup_flags_repeated_document_chunks():
    v = TextVectorizer(TextChunker(chunk_size=10, overlap=0, unit="words"), HashingEmbedder(256))
    engine = cachvec.DedupEngine(v.dim, threshold=0.95)
    first = v.dedup(engine, "orig", DOC_A)
    assert all(m == [] for _, m in first)
    repeat = v.dedup(engine, "copy", DOC_A)
    assert all(m and m[0].id.startswith("orig#") and m[0].exact for _, m in repeat)
    other = v.dedup(engine, "other", DOC_B)
    assert all(m == [] for _, m in other)
    assert len(engine) == 2 * len(first) + len(other)  # reported, never deleted


def test_callable_embedder_plugs_in_and_validates_shape():
    def fake_model(texts):
        return np.stack([np.eye(8)[len(t) % 8] + 0.1 for t in texts])

    v = TextVectorizer(TextChunker(chunk_size=2, overlap=0, unit="words"), CallableEmbedder(fake_model, dim=8))
    index = cachvec.VectorIndex(8)
    v.add(index, "d", "one two three four five six")
    assert len(index) == 3
    with pytest.raises(ValueError):
        CallableEmbedder(lambda t: np.zeros((len(t), 3)), dim=8).embed(["x"])
