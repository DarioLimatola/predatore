//! Context codec and binary associative memory (port of context.py and hdc.py).
use crate::bits;
use crate::key::Key;
use crate::lru::{FxMap, Ordered};
use monade_core::word::Word;

pub const HDC_CANDIDATES: usize = 16;
pub const HDC_NEIGHBORS: usize = 4;
pub const HDC_MIN_CONSENSUS: i64 = 2;
pub const HDC_DISTANCE_DENOMINATOR: usize = 4;
pub const HDC_VOTE_MARGIN: i64 = 1;
const ENCODE_CACHE: usize = 16384;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Codec {
    pub sensors: u32,
    pub history: u32,
    pub action_bits: u32,
    pub actions: usize,
    pub value_bits: u32,
    pub bits: u32,
}

impl Codec {
    pub fn new(sensors: u32, history: u32, action_bits: u32) -> Codec {
        let value_bits = sensors + history * (sensors + action_bits);
        Codec { sensors, history, action_bits, actions: 1 << action_bits, value_bits, bits: value_bits + history }
    }

    /// `history` oldest first, as Python lists it.
    pub fn encode(&self, current: u64, history: &[(u64, usize)]) -> Key {
        let past: &[(u64, usize)] = if self.history == 0 {
            &[]
        } else {
            let h = self.history as usize;
            if history.len() > h { &history[history.len() - h..] } else { history }
        };
        let mut k = Key::from_u64(current);
        let mut shift = self.sensors;
        for &(obs, action) in past.iter().rev() {
            k.or_shifted(obs, shift);
            shift += self.sensors;
            k.or_shifted(action as u64, shift);
            shift += self.action_bits;
        }
        let n = past.len() as u32;
        if n > 0 {
            k.or_shifted((1u64 << n) - 1, self.value_bits);
        }
        k
    }

    pub fn validate(&self, s: &str) -> Option<Key> {
        if s.len() > self.bits as usize || !s.is_ascii() || s.is_empty() || !s.bytes().all(|c| c.is_ascii_digit()) {
            return None;
        }
        let k = Key::parse(s)?;
        if k.to_decimal() != s || k.bit_length() > self.bits {
            return None;
        }
        let valid = k.bits(self.value_bits, self.history.min(64));
        if valid & valid.wrapping_add(1) != 0 {
            return None;
        }
        let used = self.sensors + valid.count_ones() * (self.sensors + self.action_bits);
        // value bits above `used` and below value_bits must be zero
        let mut pos = used;
        while pos < self.value_bits {
            let len = (self.value_bits - pos).min(64);
            if k.bits(pos, len) != 0 {
                return None;
            }
            pos += len;
        }
        Some(k)
    }

    /// (current, history oldest first)
    pub fn decode(&self, k: &Key) -> (u64, Vec<(u64, usize)>) {
        let current = k.bits(0, self.sensors);
        let count = k.bits(self.value_bits, self.history.min(64)).count_ones();
        let mut shift = self.sensors;
        let mut h = Vec::with_capacity(count as usize);
        for _ in 0..count {
            let obs = k.bits(shift, self.sensors);
            shift += self.sensors;
            let a = k.bits(shift, self.action_bits) as usize;
            shift += self.action_bits;
            h.push((obs, a));
        }
        h.reverse();
        (current, h)
    }
}

/// Symbol vectors computed in Python (shake_256) and handed to the kernel.
#[derive(Clone)]
pub struct Symbols {
    pub feature_role: Vec<Vec<u64>>, // per sensor bit
    pub value: [Vec<u64>; 2],       // symbol:'0', symbol:'1'
    pub action_role: Vec<u64>,
    pub action: Vec<Vec<u64>>,       // symbol:'a' per action
    pub hist_role: Vec<u64>,
    pub hist: Vec<Vec<u64>>,         // symbol:'h' for h in 0..=history
    pub tie: Vec<u64>,
}

#[derive(Clone)]
pub struct AssocEntry {
    pub decimal: String,
    pub vector: Vec<u64>,
    pub anchor: u64,
    pub positive: u64,
    pub negative: u64,
}

#[derive(Clone, Default)]
pub struct Lookup {
    pub neighbors: usize,
    pub action: Option<usize>,
    pub distances: Vec<u32>,
}

#[derive(Clone)]
pub struct AssociativeCore {
    pub codec: Codec,
    pub capacity: usize,
    pub width: usize,
    pub sensor_mask: u64,
    pub symbols: Symbols,
    pub entries: Ordered<Key, AssocEntry>,
    pub buckets: FxMap<u64, Ordered<Key, ()>>,
    pub last_lookup: Option<Lookup>,
    cache: FxMap<Key, Vec<u64>>,
    rot: Vec<Vec<Vec<u64>>>,
}

impl AssociativeCore {
    pub fn new(codec: Codec, capacity: usize, width: usize, sensor_mask: u64, symbols: Symbols) -> Self {
        AssociativeCore { codec, capacity, width, sensor_mask, symbols, entries: Ordered::new(), buckets: FxMap::default(), last_lookup: None, cache: FxMap::default(), rot: Vec::new() }
    }

    pub fn encode(&mut self, key: &Key) -> Vec<u64> {
        if let Some(v) = self.cache.get(key) {
            return v.clone();
        }
        let v = self.encode_uncached(key);
        if self.cache.len() >= ENCODE_CACHE {
            self.cache.clear();
        }
        self.cache.insert(*key, v.clone());
        v
    }

    fn rotated(&mut self) -> &Vec<Vec<Vec<u64>>> {
        // [lag][slot] where slot = 2*bit+value for features, then 2*sensors + action.
        if self.rot.is_empty() {
            let w = self.width;
            let n = bits::words(w);
            let mut t = vec![0u64; n];
            let mut buf = vec![0u64; n];
            let sensors = self.codec.sensors as usize;
            for lag in 0..=self.codec.history as usize {
                let mut row = Vec::with_capacity(2 * sensors + self.codec.actions);
                for b in 0..sensors {
                    for v in 0..2 {
                        for i in 0..n { buf[i] = self.symbols.feature_role[b][i] ^ self.symbols.value[v][i]; }
                        let mut out = vec![0u64; n];
                        bits::rotate(&buf, lag * 31, w, &mut out, &mut t);
                        row.push(out);
                    }
                }
                for a in 0..self.codec.actions {
                    for i in 0..n { buf[i] = self.symbols.action_role[i] ^ self.symbols.action[a][i]; }
                    let mut out = vec![0u64; n];
                    bits::rotate(&buf, lag * 31, w, &mut out, &mut t);
                    row.push(out);
                }
                self.rot.push(row);
            }
        }
        &self.rot
    }

    fn encode_uncached(&mut self, key: &Key) -> Vec<u64> {
        let w = self.width;
        let n = bits::words(w);
        let (current, history) = self.codec.decode(key);
        let sensors = self.codec.sensors as usize;
        let mask = self.sensor_mask;
        let mut items: Vec<(u64, Option<usize>)> = vec![(current, None)];
        for &(o, a) in history.iter().rev() {
            items.push((o, Some(a)));
        }
        let hl = history.len();
        let mut hist = vec![0u64; n];
        for i in 0..n { hist[i] = self.symbols.hist_role[i] ^ self.symbols.hist[hl][i]; }
        let tie = self.symbols.tie.clone();
        let rot = self.rotated();
        // bit-sliced counters (planes[i] holds bit i of every per-position count)
        let mut planes: Vec<Vec<u64>> = Vec::new();
        let mut parts = 0usize;
        let add = |v: &[u64], planes: &mut Vec<Vec<u64>>| {
            let mut carry: Vec<u64> = v.to_vec();
            for p in planes.iter_mut() {
                let mut any = 0u64;
                for i in 0..n {
                    let old = p[i];
                    p[i] = old ^ carry[i];
                    carry[i] &= old;
                    any |= carry[i];
                }
                if any == 0 { return; }
            }
            if carry.iter().any(|&x| x != 0) { planes.push(carry); }
        };
        for (lag, (obs, action)) in items.iter().enumerate() {
            for b in 0..sensors {
                if mask >> b & 1 == 1 {
                    let v = ((obs >> b) & 1) as usize;
                    add(&rot[lag][2 * b + v], &mut planes);
                    parts += 1;
                }
            }
            if let Some(a) = action {
                add(&rot[lag][2 * sensors + a], &mut planes);
                parts += 1;
            }
        }
        add(&hist, &mut planes);
        parts += 1;
        // Python hdc.majority on the planes
        let threshold = parts / 2;
        let levels = planes.len().max(usize::BITS as usize - threshold.leading_zeros() as usize);
        let mut equal = vec![u64::MAX; n];
        let mut greater = vec![0u64; n];
        for lvl in (0..levels).rev() {
            for i in 0..n {
                let plane = if lvl < planes.len() { planes[lvl][i] } else { 0 };
                if (threshold >> lvl) & 1 == 1 {
                    equal[i] &= plane;
                } else {
                    greater[i] |= equal[i] & plane;
                    equal[i] &= !plane;
                }
            }
        }
        let mut out = vec![0u64; n];
        for i in 0..n {
            out[i] = greater[i] | if parts % 2 == 0 { equal[i] & tie[i] } else { 0 };
        }
        bits::mask_in_place(&mut out, w);
        out
    }

    fn remove(&mut self, key: &Key) {
        if let Some(old) = self.entries.remove(key) {
            if let Some(b) = self.buckets.get_mut(&old.anchor) {
                b.remove(key);
                if b.is_empty() {
                    self.buckets.remove(&old.anchor);
                }
            }
        }
    }

    pub fn observe(&mut self, key: Key, positive: u64, negative: u64) {
        if positive == 0 && negative == 0 {
            self.remove(&key);
            return;
        }
        let vector = self.encode(&key);
        let anchor = self.codec.decode(&key).0 & self.sensor_mask;
        self.remove(&key);
        self.entries.insert(key, AssocEntry { decimal: key.to_decimal(), vector, anchor, positive, negative });
        self.buckets.entry(anchor).or_default().insert(key, ());
        if self.entries.len() > self.capacity {
            if let Some(first) = self.entries.first_key() {
                self.remove(&first);
            }
        }
    }

    /// Last HDC_CANDIDATES keys of the bucket, oldest first.
    pub fn bucket_tail(&self, anchor: u64) -> Vec<Key> {
        match self.buckets.get(&anchor) {
            None => Vec::new(),
            Some(b) => {
                let mut v: Vec<Key> = b.iter_rev().take(HDC_CANDIDATES).map(|(k, _)| k).collect();
                v.reverse();
                v
            }
        }
    }

    pub fn choose(&mut self, key: &Key, proposed: usize, rejected: u64, allowed: &dyn Fn(&Key) -> bool) -> Option<usize> {
        let (current, _) = self.codec.decode(key);
        let keys = self.bucket_tail(current & self.sensor_mask);
        if keys.len() < HDC_MIN_CONSENSUS as usize {
            return None;
        }
        let query = self.encode(key);
        let width = self.width;
        let mut nearest: Vec<(u32, &str, Key)> = Vec::new();
        for k in keys.iter() {
            if k != key && allowed(k) {
                let e = self.entries.get(k).unwrap();
                // Hamming distance on Monade words (shared foundation, monade-core).
                let d: u32 = query.iter().zip(e.vector.iter())
                    .map(|(&a, &b)| Word::new(a).hamming(Word::new(b)) as u32).sum();
                nearest.push((d, e.decimal.as_str(), *k));
            }
        }
        nearest.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
        let nearest: Vec<(u32, Key)> = nearest
            .into_iter()
            .map(|p| (p.0, p.2))
            .filter(|p| p.0 as usize * HDC_DISTANCE_DENOMINATOR <= width)
            .take(HDC_NEIGHBORS)
            .collect();
        let actions = self.codec.actions;
        let mut pos = vec![0i64; actions];
        let mut neg = vec![0i64; actions];
        for (_, k) in nearest.iter() {
            let e = self.entries.get(k).unwrap();
            for a in 0..actions {
                pos[a] += (e.positive >> a & 1) as i64;
                neg[a] += (e.negative >> a & 1) as i64;
            }
        }
        let mut cands: Vec<usize> = (0..actions)
            .filter(|&a| rejected >> a & 1 == 0 && pos[a] >= HDC_MIN_CONSENSUS && pos[a] > neg[a])
            .collect();
        cands.sort_by_key(|&a| (-(pos[a] - neg[a]), (a + actions - proposed % actions) % actions));
        let mut result = None;
        if let Some(&best) = cands.first() {
            let rival = (0..actions).filter(|&a| a != best).map(|a| pos[a] - neg[a]).max().unwrap_or(0);
            if pos[best] - neg[best] >= rival + HDC_VOTE_MARGIN {
                result = Some(best);
            }
        }
        self.last_lookup = Some(Lookup { neighbors: nearest.len(), action: result, distances: nearest.iter().map(|p| p.0).collect() });
        result
    }
}
