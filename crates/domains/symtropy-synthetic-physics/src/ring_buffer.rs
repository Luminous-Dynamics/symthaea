// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Fixed-capacity ring buffer for time-series metric storage.
//!
//! Used to store the last N frames of [`GraphMetrics`] without heap reallocation.
//! Also used by `symthaea-projection` for the 64-frame waterfall history.

use serde::{Deserialize, Serialize};

/// A fixed-capacity circular buffer. Overwrites oldest entries when full.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RingBuffer<T> {
    data: Vec<Option<T>>,
    capacity: usize,
    head: usize, // next write position
    len: usize,  // current number of valid entries
}

impl<T: Clone> RingBuffer<T> {
    /// Create a new ring buffer with the given capacity.
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "capacity must be non-zero");
        Self {
            data: vec![None; capacity],
            capacity,
            head: 0,
            len: 0,
        }
    }

    /// Push a new entry, overwriting the oldest if full.
    pub fn push(&mut self, value: T) {
        self.data[self.head] = Some(value);
        self.head = (self.head + 1) % self.capacity;
        if self.len < self.capacity {
            self.len += 1;
        }
    }

    /// Number of valid entries currently stored.
    pub fn len(&self) -> usize {
        self.len
    }

    /// True if no entries have been written yet.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// True if the buffer is at capacity (oldest entries are being overwritten).
    pub fn is_full(&self) -> bool {
        self.len == self.capacity
    }

    /// Return the most recently pushed entry.
    pub fn latest(&self) -> Option<&T> {
        if self.is_empty() {
            return None;
        }
        let idx = (self.head + self.capacity - 1) % self.capacity;
        self.data[idx].as_ref()
    }

    /// Return all entries in chronological order (oldest first).
    pub fn as_slice(&self) -> Vec<&T> {
        if self.is_empty() {
            return vec![];
        }
        let start = if self.is_full() { self.head } else { 0 };
        (0..self.len)
            .map(|i| {
                let idx = (start + i) % self.capacity;
                self.data[idx].as_ref().expect("valid entry in len range")
            })
            .collect()
    }

    /// Return entries as owned values in chronological order.
    pub fn to_vec(&self) -> Vec<T> {
        self.as_slice().into_iter().cloned().collect()
    }

    /// Access by age: 0 = most recent, 1 = one tick ago, etc.
    pub fn get_by_age(&self, age: usize) -> Option<&T> {
        if age >= self.len {
            return None;
        }
        let idx = (self.head + self.capacity - 1 - age) % self.capacity;
        self.data[idx].as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_and_retrieve() {
        let mut buf: RingBuffer<i32> = RingBuffer::new(4);
        buf.push(1);
        buf.push(2);
        buf.push(3);
        assert_eq!(buf.len(), 3);
        assert_eq!(buf.latest(), Some(&3));
    }

    #[test]
    fn overwrites_when_full() {
        let mut buf: RingBuffer<i32> = RingBuffer::new(4);
        for i in 0..6 {
            buf.push(i);
        }
        assert_eq!(buf.len(), 4);
        // Should contain [2, 3, 4, 5] in order
        let slice = buf.to_vec();
        assert_eq!(slice, vec![2, 3, 4, 5]);
    }

    #[test]
    fn age_access() {
        let mut buf: RingBuffer<i32> = RingBuffer::new(8);
        for i in 0..5 {
            buf.push(i);
        }
        assert_eq!(buf.get_by_age(0), Some(&4)); // most recent
        assert_eq!(buf.get_by_age(1), Some(&3));
        assert_eq!(buf.get_by_age(4), Some(&0)); // oldest
        assert_eq!(buf.get_by_age(5), None); // out of range
    }

    #[test]
    fn as_slice_chronological() {
        let mut buf: RingBuffer<i32> = RingBuffer::new(4);
        buf.push(10);
        buf.push(20);
        buf.push(30);
        buf.push(40);
        buf.push(50); // overwrites 10
        let v = buf.to_vec();
        assert_eq!(v, vec![20, 30, 40, 50]);
    }
}
