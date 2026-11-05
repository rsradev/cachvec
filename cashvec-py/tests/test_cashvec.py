import numpy as np
import pytest

import cashvec


def rand(rng, n, dim):
    return rng.standard_normal((n, dim)).astype(np.float32)


def at_cosine(rng, base, c):
    b = base / np.linalg.norm(base)
    w = rng.standard_normal(len(b)).astype(np.float32)
    w -= (w @ b) * b
    w /= np.linalg.norm(w)
    return (c * b + np.sqrt(1 - c * c) * w).astype(np.float32)


def test_hasher_spec_example():
    fp = cashvec.VectorHasher(128).hash([0.12, 0.45, 0.78, 0.21])
    assert fp.bits == 128
    assert len(fp.to_bytes()) == 16
    assert len(str(fp)) == 32


def test_inputs_list_f32_f64_agree():
    v = [0.5, -1.0, 2.0, 0.25]
    h = cashvec.VectorHasher(64, seed=3)
    a = h.hash(v)
    assert a == h.hash(np.array(v, dtype=np.float32))
    assert a == h.hash(np.array(v, dtype=np.float64))
    assert hash(a) == hash(h.hash(v))


def test_simhash_matches_lazy_hasher_and_hash_many():
    rng = np.random.default_rng(0)
    m = rand(rng, 5, 32)
    s = cashvec.SimHasher(32, 100, seed=9)
    lazy = cashvec.VectorHasher(100, seed=9)
    many = s.hash_many(m)
    assert many == [lazy.hash(r) for r in m]
    assert many == [s.hash(r) for r in m]


def test_hamming_and_cosine_estimate():
    rng = np.random.default_rng(1)
    base = rng.standard_normal(128).astype(np.float32)
    near = at_cosine(rng, base, 0.95)
    h = cashvec.SimHasher(128, 1024)
    a, b = h.hash(base), h.hash(near)
    assert a.hamming_distance(a) == 0
    assert a.hamming_distance(b) == b.hamming_distance(a)
    assert abs(a.estimated_cosine(b) - 0.95) < 0.1
    assert a.is_similar_to(b, 0.8)


def test_exact_fingerprint():
    assert cashvec.exact_fingerprint([0.0, 1.0]) == cashvec.exact_fingerprint([-0.0, 1.0])
    assert cashvec.exact_fingerprint([1.0]) != cashvec.exact_fingerprint([1.0, 0.0])
    # non-contiguous view works
    arr = np.arange(10, dtype=np.float32)
    assert cashvec.exact_fingerprint(arr[::2]) == cashvec.exact_fingerprint(arr[::2].copy())


def test_cosine_similarity():
    assert cashvec.cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cashvec.cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0, abs=1e-6)


def test_errors_are_value_errors():
    assert issubclass(cashvec.CashVecError, ValueError)
    idx = cashvec.VectorIndex(3)
    with pytest.raises(cashvec.CashVecError):
        idx.insert("a", [1.0, float("nan"), 0.0])
    with pytest.raises(ValueError):
        idx.insert("a", [1.0, 2.0])
    with pytest.raises(cashvec.CashVecError):
        idx.insert("z", [0.0, 0.0, 0.0])
    with pytest.raises(cashvec.CashVecError):
        idx.search([1, 0, 0], 5, 2.0)
    with pytest.raises(cashvec.CashVecError):
        cashvec.VectorHasher(0).hash([1.0])
    with pytest.raises(cashvec.CashVecError):
        cashvec.VectorIndex(0)
    idx.insert("a", [1, 2, 3])
    with pytest.raises(cashvec.CashVecError):
        idx.insert("a", [1, 2, 3])
    assert len(idx) == 1


def test_index_insert_search_remove():
    rng = np.random.default_rng(2)
    dim = 64
    idx = cashvec.VectorIndex(dim)
    base = rng.standard_normal(dim).astype(np.float32)
    idx.insert("exact", base)
    idx.insert("close", at_cosine(rng, base, 0.99))
    idx.insert("far", rng.standard_normal(dim))
    assert len(idx) == 3 and "exact" in idx and "nope" not in idx
    hits = idx.search(base, k=10, threshold=0.9)
    assert [h[0] for h in hits] == ["exact", "close"]
    assert hits[0][1] >= hits[1][1]
    np.testing.assert_allclose(idx.get("exact"), base)
    assert idx.get("nope") is None
    assert idx.find_exact(base) == ["exact"]
    assert idx.fingerprint_of("exact") == cashvec.exact_fingerprint(base)
    assert idx.remove("exact") and not idx.remove("exact")
    assert [h[0] for h in idx.search(base, 10, 0.9)] == ["close"]


def test_batch_apis_and_recall():
    rng = np.random.default_rng(3)
    n, dim = 2000, 128
    data = rand(rng, n, dim)
    idx = cashvec.VectorIndex(dim)
    idx.insert_many([f"d{i}" for i in range(n)], data)
    assert len(idx) == n

    queries = np.stack([at_cosine(rng, data[i * 7], 0.96) for i in range(50)])
    approx = idx.search_many(queries, k=10, threshold=0.9)
    assert len(approx) == 50
    found = total = 0
    for q, a in zip(queries, approx):
        truth = {h[0] for h in idx.search_brute_force(q, 10, 0.9)}
        total += len(truth)
        found += len(truth & {h[0] for h in a})
        assert a == idx.search(q, 10, 0.9)
    assert total >= 50 and found / total >= 0.95


def test_fortran_ordered_and_f64_matrices():
    rng = np.random.default_rng(4)
    data = rand(rng, 20, 16)
    a = cashvec.VectorIndex(16)
    a.insert_many([str(i) for i in range(20)], data)
    b = cashvec.VectorIndex(16)
    b.insert_many([str(i) for i in range(20)], np.asfortranarray(data))
    c = cashvec.VectorIndex(16)
    c.insert_many([str(i) for i in range(20)], data.astype(np.float64))
    for i in range(20):
        np.testing.assert_array_equal(a.get(str(i)), b.get(str(i)))
        np.testing.assert_allclose(a.get(str(i)), c.get(str(i)))


def test_insert_many_validation():
    idx = cashvec.VectorIndex(4)
    with pytest.raises(ValueError):
        idx.insert_many(["a"], np.zeros((2, 4), dtype=np.float32))
    with pytest.raises(cashvec.CashVecError):
        idx.insert_many(["a", "b"], np.ones((2, 5), dtype=np.float32))


def test_dedup_engine():
    rng = np.random.default_rng(5)
    dim = 48
    eng = cashvec.DedupEngine(dim, threshold=0.95)
    a = rng.standard_normal(dim).astype(np.float32)
    eng.insert("doc-1", a)
    eng.insert("doc-2", rng.standard_normal(dim))
    assert eng.is_exact_duplicate(a) and eng.find_exact(a) == ["doc-1"]

    near = at_cosine(rng, a, 0.98)
    (m,) = eng.find_similar(near)
    assert m.id == "doc-1" and not m.exact and 0.97 < m.score < 0.99
    (m,) = eng.find_similar(a, threshold=0.99)
    assert (m.id, m.score, m.exact) == ("doc-1", 1.0, True)
    assert "doc-1" in repr(m)

    report = eng.check_and_insert("doc-3", near)
    assert [r.id for r in report] == ["doc-1"]
    assert len(eng) == 3 and "doc-3" in eng
    with pytest.raises(cashvec.CashVecError):
        eng.check_and_insert("doc-3", near)
    assert eng.remove("doc-1") and len(eng) == 2
    with pytest.raises(cashvec.CashVecError):
        cashvec.DedupEngine(4, threshold=2.0)
