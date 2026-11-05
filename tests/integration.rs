mod common;
use cashvec::*;
use common::Rng;

fn index(dim: usize) -> VectorIndex {
    VectorIndex::new(IndexConfig::new(dim)).unwrap()
}

#[test]
fn insert_and_retrieve() {
    let mut idx = index(4);
    idx.insert("a", &[1.0, 2.0, 3.0, 4.0]).unwrap();
    idx.insert("b", &[4.0, 3.0, 2.0, 1.0]).unwrap();
    assert_eq!(idx.len(), 2);
    assert_eq!(idx.get("a"), Some(&[1.0, 2.0, 3.0, 4.0][..]));
    assert_eq!(idx.get("zzz"), None);
    let fp = idx.fingerprint_of("b").unwrap();
    assert_eq!(idx.find_by_fingerprint(&fp), vec!["b"]);
    assert_eq!(idx.find_exact(&[1.0, 2.0, 3.0, 4.0]).unwrap(), vec!["a"]);
}

#[test]
fn duplicate_id_is_rejected_and_index_unchanged() {
    let mut idx = index(2);
    idx.insert("a", &[1.0, 0.0]).unwrap();
    assert_eq!(
        idx.insert("a", &[0.0, 1.0]),
        Err(CashVecError::DuplicateId("a".into()))
    );
    assert_eq!(idx.len(), 1);
    assert_eq!(idx.get("a"), Some(&[1.0, 0.0][..]));
}

#[test]
fn invalid_inputs_never_panic() {
    let mut idx = index(3);
    assert_eq!(idx.insert("e", &[]), Err(CashVecError::EmptyVector));
    assert_eq!(idx.insert("n", &[1.0, f32::NAN, 0.0]), Err(CashVecError::InvalidValue { index: 1 }));
    assert_eq!(idx.insert("i", &[f32::INFINITY, 0.0, 0.0]), Err(CashVecError::InvalidValue { index: 0 }));
    assert_eq!(idx.insert("z", &[0.0; 3]), Err(CashVecError::ZeroNorm));
    assert_eq!(idx.insert("o", &[1e30; 3]), Err(CashVecError::NormOverflow));
    assert!(matches!(idx.insert("d", &[1.0; 2]), Err(CashVecError::DimensionMismatch { .. })));
    assert!(idx.is_empty());
    assert!(idx.search(&[], 5, 0.5).is_err());
    assert!(idx.search(&[1.0, 0.0, 0.0], 5, f32::NAN).is_err());
    assert!(idx.search(&[1.0, 0.0, 0.0], 5, 1.5).is_err());
    assert_eq!(idx.search(&[1.0, 0.0, 0.0], 0, 0.5).unwrap(), vec![]);
}

#[test]
fn different_dimensions() {
    for dim in [1, 2, 3, 17, 128, 384, 768, 1536] {
        let mut rng = Rng::new(dim as u64);
        let mut idx = index(dim);
        let v = rng.vector(dim);
        idx.insert("x", &v).unwrap();
        let hits = idx.search(&v, 3, 0.99).unwrap();
        assert_eq!(hits.len(), 1, "dim {dim}");
        assert_eq!(hits[0].id, "x");
        assert!(hits[0].score > 0.9999);
    }
}

#[test]
fn search_finds_near_neighbours_and_orders_results() {
    let dim = 64;
    let mut rng = Rng::new(1);
    let mut idx = index(dim);
    let base = rng.vector(dim);
    idx.insert("exact", &base).unwrap();
    idx.insert("close", &common::at_cosine(&mut rng, &base, 0.99)).unwrap();
    idx.insert("far", &rng.vector(dim)).unwrap();
    let hits = idx.search(&base, 10, 0.9).unwrap();
    let ids: Vec<_> = hits.iter().map(|h| h.id.as_str()).collect();
    assert_eq!(ids, vec!["exact", "close"]);
    assert!(hits[0].score >= hits[1].score);
    assert!(hits.iter().all(|h| h.score >= 0.9));
}

#[test]
fn equal_scores_are_ordered_by_id() {
    let mut idx = index(3);
    for id in ["c", "a", "b"] {
        idx.insert(id, &[1.0, 2.0, 3.0]).unwrap();
    }
    let hits = idx.search(&[1.0, 2.0, 3.0], 10, 0.5).unwrap();
    let ids: Vec<_> = hits.iter().map(|h| h.id.as_str()).collect();
    assert_eq!(ids, vec!["a", "b", "c"]);
    assert_eq!(idx.search(&[1.0, 2.0, 3.0], 2, 0.5).unwrap().len(), 2);
}

#[test]
fn hash_collisions_are_verified_not_trusted() {
    // 1 table x 1 bit: nearly everything collides, so candidates are noisy,
    // but search results must still be filtered by true cosine similarity.
    let cfg = IndexConfig::new(32).with_tables(1).with_bits_per_table(1);
    let mut idx = VectorIndex::new(cfg).unwrap();
    let mut rng = Rng::new(5);
    let q = rng.vector(32);
    for i in 0..200 {
        idx.insert(format!("d{i}"), &rng.vector(32)).unwrap();
    }
    let cands = idx.candidates(&q).unwrap();
    assert!(cands.len() > 50, "expected heavy bucket sharing, got {}", cands.len());
    let hits = idx.search(&q, 1000, 0.8).unwrap();
    for h in &hits {
        let real = cosine_similarity(&q, idx.get(&h.id).unwrap()).unwrap();
        assert!(real >= 0.8 - 1e-4);
    }
    // every candidate is distinct from "similar": most are rejected
    assert!(hits.len() < cands.len());
}

#[test]
fn remove_keeps_index_consistent() {
    let mut rng = Rng::new(9);
    let dim = 24;
    let mut idx = index(dim);
    let vecs: Vec<Vec<f32>> = (0..50).map(|_| rng.vector(dim)).collect();
    for (i, v) in vecs.iter().enumerate() {
        idx.insert(format!("d{i}"), v).unwrap();
    }
    for i in (0..50).step_by(3) {
        assert!(idx.remove(&format!("d{i}")));
        assert!(!idx.remove(&format!("d{i}")));
    }
    for (i, v) in vecs.iter().enumerate() {
        let id = format!("d{i}");
        if i % 3 == 0 {
            assert!(!idx.contains(&id));
            assert!(idx.find_exact(v).unwrap().is_empty());
        } else {
            assert_eq!(idx.get(&id), Some(&v[..]), "{id}");
            assert_eq!(idx.find_exact(v).unwrap(), vec![id.as_str()]);
            assert_eq!(idx.search(v, 1, 0.999).unwrap()[0].id, id);
        }
    }
    // removed IDs can be reused
    idx.insert("d0", &vecs[0]).unwrap();
    assert_eq!(idx.search(&vecs[0], 1, 0.999).unwrap()[0].id, "d0");
    // drain everything
    let ids: Vec<String> = (0..50).map(|i| format!("d{i}")).collect();
    for id in &ids {
        idx.remove(id);
    }
    assert!(idx.is_empty());
    assert!(idx.bucket_stats().iter().all(|(b, _)| *b == 0));
}

#[test]
fn exact_duplicates_and_negative_zero() {
    let mut idx = index(3);
    idx.insert("a", &[0.0, 1.0, 2.0]).unwrap();
    idx.insert("b", &[0.0, 1.0, 2.0]).unwrap();
    assert_eq!(idx.find_exact(&[-0.0, 1.0, 2.0]).unwrap(), vec!["a", "b"]);
}

#[test]
fn hasher_example_from_spec() {
    let fp = VectorHasher::new(128).hash(&[0.12, 0.45, 0.78, 0.21]).unwrap();
    assert_eq!(fp.bits(), 128);
}

#[test]
fn dedup_engine_workflow() {
    let dim = 48;
    let mut rng = Rng::new(21);
    let mut engine = DedupEngine::new(DedupConfig::new(dim)).unwrap();
    let a = rng.vector(dim);
    engine.insert("doc-1", &a).unwrap();
    engine.insert("doc-2", &rng.vector(dim)).unwrap();

    assert!(engine.is_exact_duplicate(&a).unwrap());
    assert_eq!(engine.find_exact(&a).unwrap(), vec!["doc-1"]);

    let near = common::at_cosine(&mut rng, &a, 0.98);
    let m = engine.find_similar(&near, 0.95).unwrap();
    assert_eq!(m.len(), 1);
    assert_eq!(m[0].id, "doc-1");
    assert!(!m[0].exact);
    assert!(m[0].score > 0.97 && m[0].score < 0.99);

    let m = engine.find_similar(&a, 0.95).unwrap();
    assert_eq!((m[0].id.as_str(), m[0].score, m[0].exact), ("doc-1", 1.0, true));

    // check_and_insert reports first, then indexes
    let report = engine.check_and_insert("doc-3", &near).unwrap();
    assert_eq!(report.len(), 1);
    assert!(engine.contains("doc-3"));
    assert_eq!(engine.len(), 3);
    assert!(engine.check_and_insert("doc-3", &near).is_err());

    // engine reports only; nothing is deleted until asked
    assert_eq!(engine.len(), 3);
    assert!(engine.remove("doc-1"));
    assert!(engine.find_exact(&a).unwrap().is_empty());
    assert!(DedupEngine::new(DedupConfig::new(4).with_threshold(2.0)).is_err());
}
