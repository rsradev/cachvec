//! Deterministic, counter-based pseudo-random generation.
//!
//! Random projection coordinates are a pure function of `(seed, plane, i)`.
//! This makes them reproducible across platforms (only integer arithmetic and
//! exact integer-to-float conversions are used) and lets a hasher generate any
//! plane without storing it.
//!
//! Each coordinate is the sum of four independent uniform 16-bit draws
//! (an Irwin–Hall variate), which is bell-shaped and close enough to Gaussian
//! for random-hyperplane hashing while staying cheap and bit-exact.

/// Default seed used when none is supplied.
pub const DEFAULT_SEED: u64 = 0x0CA5_4C0D_E5ED_0001;

const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;

/// SplitMix64 finalizer-based mixer.
#[inline]
pub(crate) fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(GOLDEN);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Per-plane stream key derived from the seed and plane index.
#[inline]
pub(crate) fn plane_key(seed: u64, plane: u64) -> u64 {
    splitmix64(seed ^ splitmix64(plane.wrapping_mul(0xD1B5_4A32_D192_ED03)))
}

/// Coordinate `i` of the plane with key `key`. Mean 0, symmetric.
#[inline]
pub(crate) fn plane_coord(key: u64, i: u64) -> f32 {
    let r = splitmix64(key.wrapping_add(i.wrapping_mul(GOLDEN)));
    let s = (r & 0xFFFF) + ((r >> 16) & 0xFFFF) + ((r >> 32) & 0xFFFF) + (r >> 48);
    // Mean of the sum is 4 * 32767.5 = 131070; scale only matters for range.
    (s as f32 - 131_070.0) * (1.0 / 65_536.0)
}

/// Fill `out` with the coordinates of the plane with key `key`.
pub(crate) fn fill_plane(key: u64, out: &mut [f32]) {
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = plane_coord(key, i as u64);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coords_are_deterministic_and_centered() {
        let key = plane_key(7, 3);
        assert_eq!(plane_coord(key, 10), plane_coord(key, 10));
        let n = 100_000u64;
        let mean: f64 = (0..n).map(|i| plane_coord(key, i) as f64).sum::<f64>() / n as f64;
        assert!(mean.abs() < 0.01, "mean {mean}");
    }

    #[test]
    fn different_planes_differ() {
        assert_ne!(plane_key(1, 0), plane_key(1, 1));
        assert_ne!(plane_key(1, 0), plane_key(2, 0));
    }
}
