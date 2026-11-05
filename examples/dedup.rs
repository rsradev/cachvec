use cashvec::{DedupConfig, DedupEngine, IndexConfig, VectorIndex};

fn main() -> cashvec::Result<()> {
    let a = [0.12, 0.45, 0.78, 0.21];
    let b = [0.12, 0.45, 0.78, 0.22];

    // Similarity search
    let mut index = VectorIndex::new(IndexConfig::new(4))?;
    index.insert("doc-1", &a)?;
    for hit in index.search(&b, 10, 0.95)? {
        println!("search: {} (cosine {:.4})", hit.id, hit.score);
    }

    // Deduplication: reports candidates, never deletes.
    let mut engine = DedupEngine::new(DedupConfig::new(4))?;
    engine.insert("doc-1", &a)?;
    for m in engine.find_similar(&b, 0.95)? {
        println!("duplicate candidate: {} (cosine {:.4}, exact: {})", m.id, m.score, m.exact);
    }
    Ok(())
}
