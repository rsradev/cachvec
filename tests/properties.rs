//! Property-style tests over many seeded random inputs (no external crates).
mod common;
use cashvec::similarity::hamming_distance;
use cashvec::vector::{norm, normalized};
use cashvec::*;
use common::Rng;

const CASES: u64 = 200;

fn random_dim(rng: &mut Rng) -> usize {
    1 + (rng.next_u64() % 200) as usize
}

#[test]
fn hashing_is_deterministic() {
    for seed in 0..CASES {
        let mut rng = Rng::new(seed);
        let d = random_dim(&mut rng);
        let v = rng.vector(d);
        let h = VectorHasher::with_seed(64, seed);
        assert_eq!(h.hash(&v).unwrap(), h.hash(&v).unwrap());
        assert_eq!(exact_fingerprint(&v).unwrap(), exact_fingerprint(&v.clone()).unwrap());
    }
}

#[test]
fn identical_vectors_have_identical_exact_fingerprints_and_distinct_ones_differ() {
    for seed in 0..CASES {
        let mut rng = Rng::new(seed);
        let d = random_dim(&mut rng);
        let v = rng.vector(d);
        let mut w = v.clone();
        let i = (rng.next_u64() as usize) % w.len();
        w[i] = f32::from_bits(w[i].to_bits() ^ 1); // flip the lowest mantissa bit
        assert_ne!(exact_fingerprint(&v).unwrap(), exact_fingerprint(&w).unwrap());
    }
}

#[test]
fn cosine_is_symmetric_and_bounded() {
    for seed in 0..CASES {
        let mut rng = Rng::new(seed);
        let d = random_dim(&mut rng);
        let (a, b) = (rng.vector(d), rng.vector(d));
        let (ab, ba) = (cosine_similarity(&a, &b).unwrap(), cosine_similarity(&b, &a).unwrap());
        assert_eq!(ab, ba);
        assert!((-1.0..=1.0).contains(&ab));
    }
}

#[test]
fn normalized_vectors_have_unit_length() {
    for seed in 0..CASES {
        let mut rng = Rng::new(seed);
        let d = random_dim(&mut rng);
        let scale = 10f32.powi((rng.next_u64() % 7) as i32 - 3);
        let v: Vec<f32> = rng.vector(d).iter().map(|x| x * scale).collect();
        let n = normalized(&v).unwrap();
        assert!((norm(&n) - 1.0).abs() < 1e-4, "seed {seed}");
    }
}

#[test]
fn hamming_is_symmetric_and_bounded_by_bits() {
    for seed in 0..CASES {
        let mut rng = Rng::new(seed);
        let d = random_dim(&mut rng);
        let h = SimHasher::with_seed(d, 100, seed).unwrap();
        let (a, b) = (h.hash(&rng.vector(d)).unwrap(), h.hash(&rng.vector(d)).unwrap());
        let ab = a.hamming_distance(&b).unwrap();
        assert_eq!(ab, b.hamming_distance(&a).unwrap());
        assert_eq!(ab, hamming_distance(a.words(), b.words()).unwrap());
        assert!(ab <= 100);
        assert_eq!(a.hamming_distance(&a).unwrap(), 0);
    }
}

#[test]
fn lsh_keys_are_scale_invariant() {
    for seed in 0..CASES {
        let mut rng = Rng::new(seed);
        let d = random_dim(&mut rng);
        let lsh = Lsh::new(LshConfig::new(d).with_seed(seed)).unwrap();
        let v = rng.vector(d);
        let s: Vec<f32> = v.iter().map(|x| x * 4.0).collect(); // power of two: exact
        assert_eq!(lsh.keys(&v).unwrap(), lsh.keys(&s).unwrap());
    }
}
