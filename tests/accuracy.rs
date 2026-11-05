//! Accuracy of hashing and search on synthetic data with known angles.
mod common;
use cashvec::*;
use common::{at_cosine, Rng};

/// Mean absolute error of the cosine estimate from SimHash Hamming distance.
#[test]
fn simhash_estimates_cosine() {
    let dim = 128;
    let h = SimHasher::new(dim, 1024).unwrap();
    let mut rng = Rng::new(3);
    for &target in &[0.95f32, 0.8, 0.5, 0.0, -0.5] {
        let mut err = 0.0f32;
        let trials = 40;
        for _ in 0..trials {
            let a = rng.vector(dim);
            let b = at_cosine(&mut rng, &a, target);
            let est = h.hash(&a).unwrap().estimated_cosine(&h.hash(&b).unwrap()).unwrap();
            err += (est - target).abs();
        }
        let mae = err / trials as f32;
        assert!(mae < 0.06, "target {target}: mean abs error {mae}");
    }
}

/// Pairs near the threshold collide in far more buckets than unrelated ones.
#[test]
fn similar_vectors_collide_more_often() {
    let dim = 96;
    let lsh = Lsh::new(LshConfig::new(dim)).unwrap();
    let mut rng = Rng::new(4);
    let trials = 300;
    let shared = |cos: Option<f32>, rng: &mut Rng| -> f64 {
        let mut total = 0usize;
        for _ in 0..trials {
            let a = rng.vector(dim);
            let b = match cos {
                Some(c) => at_cosine(rng, &a, c),
                None => rng.vector(dim),
            };
            let (ka, kb) = (lsh.keys(&a).unwrap(), lsh.keys(&b).unwrap());
            total += ka.iter().zip(&kb).filter(|(x, y)| x == y).count();
        }
        total as f64 / trials as f64
    };
    let high = shared(Some(0.97), &mut rng);
    let mid = shared(Some(0.8), &mut rng);
    let low = shared(None, &mut rng);
    assert!(high > mid && mid > low, "{high} {mid} {low}");
    assert!(high > 5.0, "expected several shared buckets at 0.97, got {high}");
    assert!(low < 0.1, "unrelated vectors should rarely share, got {low}");
}

/// Candidate recall, search recall, precision, false positives/negatives
/// against brute force on a planted-duplicate dataset.
#[test]
fn search_recall_and_precision_on_planted_duplicates() {
    let (dim, n, n_dups) = (128, 2000, 100);
    let threshold = 0.9;
    let mut rng = Rng::new(77);
    let mut idx = VectorIndex::new(IndexConfig::new(dim)).unwrap();
    let base: Vec<Vec<f32>> = (0..n).map(|_| rng.vector(dim)).collect();
    for (i, v) in base.iter().enumerate() {
        idx.insert(format!("d{i}"), v).unwrap();
    }

    let (mut true_pos, mut false_neg, mut cand_hit) = (0, 0, 0);
    for i in 0..n_dups {
        let q = at_cosine(&mut rng, &base[i * 7], 0.96);
        let truth: Vec<String> = idx.search_brute_force(&q, usize::MAX, threshold).unwrap().into_iter().map(|r| r.id).collect();
        assert!(truth.contains(&format!("d{}", i * 7)), "planted duplicate must be ground truth");
        let found = idx.search(&q, usize::MAX, threshold).unwrap();
        // precision: nothing below the threshold is ever returned
        for r in &found {
            assert!(r.score >= threshold);
            assert!(truth.contains(&r.id), "false positive {}", r.id);
        }
        let cands = idx.candidates(&q).unwrap();
        for t in &truth {
            if cands.contains(&t.as_str()) {
                cand_hit += 1;
            }
            if found.iter().any(|r| &r.id == t) {
                true_pos += 1;
            } else {
                false_neg += 1;
            }
        }
    }
    let total = (true_pos + false_neg) as f64;
    let recall = true_pos as f64 / total;
    let cand_recall = cand_hit as f64 / total;
    eprintln!("recall={recall:.3} candidate_recall={cand_recall:.3} fn={false_neg}");
    assert!(recall >= 0.95, "recall {recall}");
    assert!(cand_recall >= recall - 1e-9);
}

/// Unrelated vectors must not be reported as duplicates.
#[test]
fn no_false_positives_among_random_vectors() {
    let dim = 128;
    let mut rng = Rng::new(8);
    let mut engine = DedupEngine::new(DedupConfig::new(dim)).unwrap();
    for i in 0..500 {
        engine.insert(format!("d{i}"), &rng.vector(dim)).unwrap();
    }
    for _ in 0..50 {
        assert!(engine.find_similar(&rng.vector(dim), 0.9).unwrap().is_empty());
    }
}
