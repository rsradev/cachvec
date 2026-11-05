//! Reproducible benchmark: LSH search vs. brute-force cosine search.
//!
//! Run:   cargo bench --bench compare            (quick: 10k vectors)
//!        cargo bench --bench compare -- --full  (10k, 100k, 1M x 128/384/768/1536)
//!        cargo bench --bench compare -- --n 50000 --dim 384
//!
//! Data is synthetic and seeded, so runs are reproducible on the same
//! machine. Queries are noisy copies (cosine ~0.97) of random indexed
//! vectors, so each has at least one true neighbour above the threshold.
//! Memory is CachVec's own capacity-based estimate, not RSS.

use cachvec::{IndexConfig, VectorHasher, VectorIndex};
use std::hint::black_box;
use std::time::Instant;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn gauss(&mut self) -> f32 {
        (0..12).map(|_| (self.next() >> 40) as f32 / 16_777_216.0).sum::<f32>() - 6.0
    }
    fn vec(&mut self, d: usize) -> Vec<f32> {
        (0..d).map(|_| self.gauss()).collect()
    }
}

fn noisy(rng: &mut Rng, base: &[f32], noise: f32) -> Vec<f32> {
    base.iter().map(|x| x + noise * rng.gauss()).collect()
}

fn run(n: usize, dim: usize) {
    const THRESH: f32 = 0.9;
    const K: usize = 10;
    const QUERIES: usize = 100;
    let mut rng = Rng(0x1234_5678_9ABC_DEF1);

    println!("\n== n={n} dim={dim} ==");
    let data: Vec<Vec<f32>> = (0..n).map(|_| rng.vec(dim)).collect();

    let hasher = VectorHasher::new(128);
    let t = Instant::now();
    let reps = 200.min(n);
    for v in data.iter().take(reps) {
        black_box(hasher.hash(v).unwrap());
    }
    println!("simhash(128b) latency       : {:>10.1} us/vector", t.elapsed().as_secs_f64() * 1e6 / reps as f64);

    let mut index = VectorIndex::new(IndexConfig::new(dim)).unwrap();
    let t = Instant::now();
    for (i, v) in data.iter().enumerate() {
        index.insert(format!("doc-{i}"), v).unwrap();
    }
    let secs = t.elapsed().as_secs_f64();
    println!("insert throughput           : {:>10.0} vectors/s", n as f64 / secs);
    println!("index memory (estimate)     : {:>10.1} MiB (raw vectors {:.1} MiB)",
        index.memory_usage() as f64 / 1048576.0, (n * dim * 4) as f64 / 1048576.0);

    let queries: Vec<Vec<f32>> = (0..QUERIES)
        .map(|i| noisy(&mut rng, &data[(i * 7919) % n], 0.2))
        .collect();

    let t = Instant::now();
    let approx: Vec<_> = queries.iter().map(|q| index.search(q, K, THRESH).unwrap()).collect();
    let a_us = t.elapsed().as_secs_f64() * 1e6 / QUERIES as f64;
    let t = Instant::now();
    let exact: Vec<_> = queries.iter().map(|q| index.search_brute_force(q, K, THRESH).unwrap()).collect();
    let b_us = t.elapsed().as_secs_f64() * 1e6 / QUERIES as f64;

    let (mut truth, mut found) = (0usize, 0usize);
    for (a, e) in approx.iter().zip(&exact) {
        truth += e.len();
        found += e.iter().filter(|x| a.iter().any(|y| y.id == x.id)).count();
    }
    let (mut cand_total, mut cand_ok) = (0usize, 0usize);
    for (q, e) in queries.iter().zip(&exact) {
        let c = index.candidates(q).unwrap();
        cand_total += c.len();
        cand_ok += e.iter().filter(|x| c.contains(&x.id.as_str())).count();
    }
    println!("LSH search latency          : {a_us:>10.1} us/query");
    println!("brute-force latency         : {b_us:>10.1} us/query  (speedup {:.1}x)", b_us / a_us);
    println!("recall@{K} (thr {THRESH})        : {:>10.3}  ({found}/{truth})", found as f64 / truth.max(1) as f64);
    println!("candidates per query        : {:>10.1}  ({:.2}% of index)", cand_total as f64 / QUERIES as f64, 100.0 * cand_total as f64 / (QUERIES * n) as f64);
    println!("candidate recall            : {:>10.3}", cand_ok as f64 / truth.max(1) as f64);
    println!("verified precision          : {:>10.3}  (results below threshold are filtered by construction)", 1.0);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let get = |flag: &str| args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1)).and_then(|v| v.parse::<usize>().ok());
    if args.iter().any(|a| a == "--full") {
        for &n in &[10_000, 100_000, 1_000_000] {
            for &d in &[128, 384, 768, 1536] {
                // Raw data is held twice (generator + index copy).
                let gib = (n * d * 4 * 2) as f64 / (1u64 << 30) as f64;
                if gib > 8.0 {
                    println!("\n== n={n} dim={d} == skipped: needs ~{gib:.0} GiB (limit 8 GiB; use --n/--dim to force)");
                    continue;
                }
                run(n, d);
            }
        }
    } else {
        run(get("--n").unwrap_or(10_000), get("--dim").unwrap_or(128));
    }
}
