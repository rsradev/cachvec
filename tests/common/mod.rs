//! Shared helpers for tests: a tiny deterministic RNG and synthetic data.
#![allow(dead_code)]

pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    pub fn next_u64(&mut self) -> u64 {
        // xorshift64*
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    pub fn uniform(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    /// Approximately standard normal (Irwin-Hall, n = 12).
    pub fn gauss(&mut self) -> f32 {
        (0..12).map(|_| self.uniform()).sum::<f32>() - 6.0
    }
    pub fn vector(&mut self, dim: usize) -> Vec<f32> {
        (0..dim).map(|_| self.gauss()).collect()
    }
}

pub fn dot(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum()
}

pub fn unit(v: &[f32]) -> Vec<f32> {
    let n = dot(v, v).sqrt() as f32;
    v.iter().map(|x| x / n).collect()
}

/// A unit vector whose cosine with `base` is (very nearly) `cos`.
pub fn at_cosine(rng: &mut Rng, base: &[f32], cos: f32) -> Vec<f32> {
    let b = unit(base);
    let mut w = rng.vector(b.len());
    let proj = dot(&w, &b) as f32;
    for (wi, bi) in w.iter_mut().zip(&b) {
        *wi -= proj * bi;
    }
    let w = unit(&w);
    let sin = (1.0 - cos * cos).max(0.0).sqrt();
    b.iter().zip(&w).map(|(x, y)| cos * x + sin * y).collect()
}
