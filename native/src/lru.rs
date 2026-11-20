//! Insertion-ordered hash map with O(1) move-to-end and removal (Python OrderedDict semantics).
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hash, Hasher};

/// Fast non-cryptographic hasher (FxHash); keys are internal integers, never attacker input.
#[derive(Default, Clone, Copy)]
pub struct Fx(u64);
impl Hasher for Fx {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for chunk in bytes.chunks(8) {
            let mut b = [0u8; 8];
            b[..chunk.len()].copy_from_slice(chunk);
            self.write_u64(u64::from_le_bytes(b));
        }
    }
    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.0 = (self.0.rotate_left(5) ^ i).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }
    #[inline]
    fn write_usize(&mut self, i: usize) { self.write_u64(i as u64); }
    #[inline]
    fn finish(&self) -> u64 { self.0 }
}
pub type FxBuild = BuildHasherDefault<Fx>;
pub type FxMap<K, V> = HashMap<K, V, FxBuild>;

const NIL: usize = usize::MAX;

#[derive(Clone)]
struct Slot<K, V> {
    key: K,
    value: V,
    prev: usize,
    next: usize,
}

#[derive(Clone)]
pub struct Ordered<K: Hash + Eq + Copy, V> {
    map: FxMap<K, usize>,
    slots: Vec<Option<Slot<K, V>>>,
    free: Vec<usize>,
    head: usize,
    tail: usize,
}

impl<K: Hash + Eq + Copy, V> Default for Ordered<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Hash + Eq + Copy, V> Ordered<K, V> {
    pub fn new() -> Self {
        Ordered { map: FxMap::default(), slots: Vec::new(), free: Vec::new(), head: NIL, tail: NIL }
    }
    pub fn len(&self) -> usize {
        self.map.len()
    }
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
    pub fn contains(&self, k: &K) -> bool {
        self.map.contains_key(k)
    }
    pub fn get(&self, k: &K) -> Option<&V> {
        self.map.get(k).map(|&i| &self.slots[i].as_ref().unwrap().value)
    }
    pub fn get_mut(&mut self, k: &K) -> Option<&mut V> {
        match self.map.get(k) {
            Some(&i) => Some(&mut self.slots[i].as_mut().unwrap().value),
            None => None,
        }
    }
    fn unlink(&mut self, i: usize) {
        let (p, n) = {
            let s = self.slots[i].as_ref().unwrap();
            (s.prev, s.next)
        };
        if p != NIL { self.slots[p].as_mut().unwrap().next = n; } else { self.head = n; }
        if n != NIL { self.slots[n].as_mut().unwrap().prev = p; } else { self.tail = p; }
    }
    fn link_back(&mut self, i: usize) {
        {
            let s = self.slots[i].as_mut().unwrap();
            s.prev = self.tail;
            s.next = NIL;
        }
        if self.tail != NIL { self.slots[self.tail].as_mut().unwrap().next = i; } else { self.head = i; }
        self.tail = i;
    }
    /// Insert at the end if absent; if present, replace value keeping position (dict semantics).
    pub fn insert(&mut self, k: K, v: V) {
        if let Some(&i) = self.map.get(&k) {
            self.slots[i].as_mut().unwrap().value = v;
            return;
        }
        let slot = Slot { key: k, value: v, prev: NIL, next: NIL };
        let i = if let Some(i) = self.free.pop() { self.slots[i] = Some(slot); i } else { self.slots.push(Some(slot)); self.slots.len() - 1 };
        self.map.insert(k, i);
        self.link_back(i);
    }
    pub fn move_to_end(&mut self, k: &K) {
        if let Some(&i) = self.map.get(k) {
            if self.tail != i {
                self.unlink(i);
                self.link_back(i);
            }
        }
    }
    pub fn remove(&mut self, k: &K) -> Option<V> {
        let i = self.map.remove(k)?;
        self.unlink(i);
        let s = self.slots[i].take().unwrap();
        self.free.push(i);
        Some(s.value)
    }
    pub fn first_key(&self) -> Option<K> {
        if self.head == NIL { None } else { Some(self.slots[self.head].as_ref().unwrap().key) }
    }
    pub fn pop_first(&mut self) -> Option<(K, V)> {
        let k = self.first_key()?;
        let v = self.remove(&k)?;
        Some((k, v))
    }
    pub fn clear(&mut self) {
        *self = Self::new();
    }
    pub fn iter(&self) -> Iter<'_, K, V> {
        Iter { o: self, cur: self.head }
    }
    /// Keys from the end backwards (newest first).
    pub fn iter_rev(&self) -> RevIter<'_, K, V> {
        RevIter { o: self, cur: self.tail }
    }
}

pub struct Iter<'a, K: Hash + Eq + Copy, V> {
    o: &'a Ordered<K, V>,
    cur: usize,
}
impl<'a, K: Hash + Eq + Copy, V> Iterator for Iter<'a, K, V> {
    type Item = (K, &'a V);
    fn next(&mut self) -> Option<Self::Item> {
        if self.cur == NIL {
            return None;
        }
        let s = self.o.slots[self.cur].as_ref().unwrap();
        self.cur = s.next;
        Some((s.key, &s.value))
    }
}
pub struct RevIter<'a, K: Hash + Eq + Copy, V> {
    o: &'a Ordered<K, V>,
    cur: usize,
}
impl<'a, K: Hash + Eq + Copy, V> Iterator for RevIter<'a, K, V> {
    type Item = (K, &'a V);
    fn next(&mut self) -> Option<Self::Item> {
        if self.cur == NIL {
            return None;
        }
        let s = self.o.slots[self.cur].as_ref().unwrap();
        self.cur = s.prev;
        Some((s.key, &s.value))
    }
}
