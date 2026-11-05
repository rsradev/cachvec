use cashvec::{exact_fingerprint, VectorHasher};

fn main() {
    let embedding = vec![0.12, 0.45, 0.78, 0.21];
    let near = vec![0.12, 0.45, 0.78, 0.22];

    // Exact fingerprint: identifies the vector *data*.
    println!("exact:  {}", exact_fingerprint(&embedding).unwrap());

    // SimHash: nearby vectors get nearby signatures.
    let hasher = VectorHasher::new(128);
    let a = hasher.hash(&embedding).unwrap();
    let b = hasher.hash(&near).unwrap();
    println!("simhash: {a}");
    println!(
        "hamming = {}, estimated cosine = {:.3}",
        a.hamming_distance(&b).unwrap(),
        a.estimated_cosine(&b).unwrap()
    );
}
