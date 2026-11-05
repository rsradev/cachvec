//! Bucket storage for one LSH table.
//!
//! A bucket maps a hash key to every slot that produced it. Different vectors
//! can share a key (that is the point of LSH, and also how true hash
//! collisions look), so buckets only ever yield *candidates*.

use std::collections::HashMap;

/// One hash table: key -> slots.
#[derive(Debug, Default, Clone)]
pub struct BucketTable {
    buckets: HashMap<u64, Vec<u32>>,
}

impl BucketTable {
    /// Empty table.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds `slot` to the bucket for `key`.
    pub fn insert(&mut self, key: u64, slot: u32) {
        self.buckets.entry(key).or_default().push(slot);
    }

    /// Removes `slot` from the bucket for `key`; drops the bucket if empty.
    /// Returns whether the slot was present.
    pub fn remove(&mut self, key: u64, slot: u32) -> bool {
        let Some(b) = self.buckets.get_mut(&key) else {
            return false;
        };
        let Some(pos) = b.iter().position(|s| *s == slot) else {
            return false;
        };
        b.swap_remove(pos);
        if b.is_empty() {
            self.buckets.remove(&key);
        }
        true
    }

    /// Renames `old` to `new` inside the bucket for `key`.
    pub fn replace(&mut self, key: u64, old: u32, new: u32) {
        if let Some(b) = self.buckets.get_mut(&key) {
            if let Some(s) = b.iter_mut().find(|s| **s == old) {
                *s = new;
            }
        }
    }

    /// Slots in the bucket for `key` (empty if none).
    pub fn get(&self, key: u64) -> &[u32] {
        self.buckets.get(&key).map_or(&[], Vec::as_slice)
    }

    /// Number of non-empty buckets.
    pub fn bucket_count(&self) -> usize {
        self.buckets.len()
    }

    /// Size of the largest bucket.
    pub fn max_bucket_len(&self) -> usize {
        self.buckets.values().map(Vec::len).max().unwrap_or(0)
    }

    /// Approximate heap size in bytes.
    pub fn memory_usage(&self) -> usize {
        // Per entry: key + Vec header + hashmap control byte; plus slot data.
        self.buckets.capacity() * (8 + 24 + 1)
            + self.buckets.values().map(|v| v.capacity() * 4).sum::<usize>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_get_remove() {
        let mut t = BucketTable::new();
        t.insert(5, 1);
        t.insert(5, 2);
        t.insert(6, 3);
        assert_eq!(t.bucket_count(), 2);
        let mut b = t.get(5).to_vec();
        b.sort();
        assert_eq!(b, vec![1, 2]);
        assert!(t.remove(5, 1));
        assert!(!t.remove(5, 1));
        assert!(t.remove(6, 3));
        assert_eq!(t.bucket_count(), 1);
        assert!(t.get(6).is_empty());
    }

    #[test]
    fn replace_renames_slot() {
        let mut t = BucketTable::new();
        t.insert(1, 10);
        t.replace(1, 10, 4);
        assert_eq!(t.get(1), &[4]);
    }
}
