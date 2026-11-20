//! Finite Boolean evidence RAM (port of boolean_engine/memory.py, same semantics).
use crate::key::Key;
use crate::lru::Ordered;

pub const PLANES: usize = 16; // CoreLimits.CONTEXT_COUNTER_BITS
pub const MIN_SUPPORT: u64 = 2; // CONTEXT_MIN_SUPPORT
pub const DECAY_INTERVAL: u64 = 1024; // MEMORY_DECAY_INTERVAL
pub const SOURCE_LIMIT: usize = 8; // MEMORY_SOURCE_LIMIT
pub const EVICTION_WINDOW: usize = 16; // MEMORY_EVICTION_WINDOW
pub const TERMINAL_CREDIT: u32 = 3; // CONTEXT_TERMINAL_CREDIT

pub type Planes = [u64; PLANES];
pub type Row = [Planes; 2];

pub const EMPTY: Row = [[0; PLANES]; 2];

pub fn increment(planes: &mut Planes, lanes: u64, mask: u64) {
    let mut full = mask;
    for p in planes.iter() {
        full &= *p;
    }
    let mut carry = lanes & !full & mask;
    for p in planes.iter_mut() {
        if carry == 0 {
            break;
        }
        let old = *p;
        *p = old ^ carry;
        carry &= old;
    }
}

pub fn greater(a: &Planes, b: &Planes, mask: u64) -> u64 {
    let mut equal = mask;
    let mut result = 0u64;
    for i in (0..PLANES).rev() {
        let (x, y) = (a[i], b[i]);
        result |= equal & x & !y;
        equal &= !(x ^ y);
    }
    result & mask
}

pub fn difference(a: &Planes, b: &Planes, mask: u64) -> Planes {
    let mut borrow = 0u64;
    let mut out = [0u64; PLANES];
    for i in 0..PLANES {
        let (x, y) = (a[i], b[i]);
        out[i] = (x ^ y ^ borrow) & mask;
        borrow = ((!x & (y | borrow)) | (y & borrow)) & mask;
    }
    out
}

pub fn add(a: &Planes, b: &Planes, mask: u64) -> Planes {
    let mut carry = 0u64;
    let mut out = [0u64; PLANES];
    for i in 0..PLANES {
        let (x, y) = (a[i], b[i]);
        out[i] = (x ^ y ^ carry) & mask;
        carry = ((x & y) | (x & carry) | (y & carry)) & mask;
    }
    for v in out.iter_mut() {
        *v |= carry;
    }
    out
}

pub fn minimum(mask: u64) -> Planes {
    let mut m = [0u64; PLANES];
    for (i, v) in m.iter_mut().enumerate() {
        if (MIN_SUPPORT >> i) & 1 == 1 {
            *v = mask;
        }
    }
    m
}

pub fn supported(row: &Row, mask: u64) -> u64 {
    let min = minimum(mask);
    !greater(&min, &row[0], mask) & greater(&row[0], &row[1], mask)
}

#[derive(Clone, Debug, PartialEq)]
pub struct Meta {
    pub last: u64,
    pub decayed: u64,
    pub updates: u64,
    pub sources: Vec<String>,
    pub signature: String,
}

#[derive(Clone, Copy)]
pub struct Evidence {
    pub good: u64,
    pub bad: u64,
    pub margin: Option<Planes>, // None == Python's empty tuple
}

#[derive(Clone)]
pub struct Entry {
    pub row: Row,
    pub terminal: Option<Row>,
    pub teacher: Option<Row>,
    pub meta: Meta,
    pub cache: Option<Evidence>,
    pub packed: Option<(String, Option<String>, Option<String>)>,
}

#[derive(Clone)]
pub struct ContextMemoryCore {
    pub capacity: usize,
    pub actions: usize,
    pub mask: u64,
    pub clock: u64,
    pub rows: Ordered<Key, Entry>,
}

#[derive(Debug)]
pub enum MemError {
    InvalidUpdate,
}

impl ContextMemoryCore {
    pub fn new(capacity: usize, actions: usize) -> Self {
        let mask = if actions >= 64 { u64::MAX } else { (1u64 << actions) - 1 };
        ContextMemoryCore { capacity, actions, mask, clock: 0, rows: Ordered::new() }
    }

    pub fn clear(&mut self) {
        self.rows.clear();
        self.clock = 0;
    }

    pub fn ensure(&mut self, key: Key) {
        if !self.rows.contains(&key) {
            let clock = self.clock;
            self.rows.insert(key, Entry {
                row: EMPTY,
                terminal: None,
                teacher: None,
                meta: Meta { last: clock, decayed: clock, updates: 0, sources: Vec::new(), signature: String::new() },
                cache: None,
                packed: None,
            });
        }
    }

    pub fn expire(&mut self, key: &Key) {
        let clock = self.clock;
        let Some(e) = self.rows.get_mut(key) else { return };
        let n = std::cmp::min(PLANES as u64, (clock - e.meta.decayed) / DECAY_INTERVAL) as usize;
        if n > 0 {
            let shift = |r: &mut Row| {
                for g in r.iter_mut() {
                    let mut out = [0u64; PLANES];
                    out[..PLANES - n].copy_from_slice(&g[n..]);
                    *g = out;
                }
            };
            shift(&mut e.row);
            if let Some(t) = e.terminal.as_mut() { shift(t); }
            if let Some(t) = e.teacher.as_mut() { shift(t); }
            e.meta.decayed = clock;
            e.cache = None;
            e.packed = None;
        }
    }

    fn touch(&mut self, key: &Key) {
        let clock = self.clock;
        if let Some(e) = self.rows.get_mut(key) {
            e.meta.last = clock;
            e.cache = None;
            e.packed = None;
        }
        self.rows.move_to_end(key);
        if self.rows.len() > self.capacity {
            let mut best: Option<(Key, (bool, bool, u64))> = None;
            for (k, e) in self.rows.iter().take(EVICTION_WINDOW) {
                let term = e.terminal.as_ref().map(|t| t[0].iter().any(|&p| p != 0)).unwrap_or(false);
                let rank = (term, !e.meta.sources.is_empty(), e.meta.last);
                if best.as_ref().map(|b| rank < b.1).unwrap_or(true) {
                    best = Some((k, rank));
                }
            }
            if let Some((k, _)) = best {
                self.rows.remove(&k);
            }
        }
    }

    pub fn update(&mut self, key: Key, action: usize, positive: bool, weight: u32, terminal: bool) -> Result<(), MemError> {
        if action >= self.actions || !(1..=TERMINAL_CREDIT).contains(&weight) {
            return Err(MemError::InvalidUpdate);
        }
        self.clock += 1;
        self.ensure(key);
        self.expire(&key);
        let mask = self.mask;
        let e = self.rows.get_mut(&key).unwrap();
        let row = if terminal { e.terminal.get_or_insert(EMPTY) } else { &mut e.row };
        let g = if positive { 0 } else { 1 };
        for _ in 0..weight {
            increment(&mut row[g], 1u64 << action, mask);
        }
        e.meta.updates += 1;
        self.touch(&key);
        Ok(())
    }

    pub fn set_row(&mut self, key: Key, row: Row) {
        self.ensure(key);
        self.rows.get_mut(&key).unwrap().row = row;
        self.touch(&key);
    }

    pub fn set_teacher(&mut self, key: Key, row: Row, sources: Vec<String>, signature: String) -> bool {
        self.ensure(key);
        if self.rows.get(&key).unwrap().meta.signature == signature {
            return false;
        }
        self.expire(&key);
        let e = self.rows.get_mut(&key).unwrap();
        e.teacher = Some(row);
        let mut s = sources;
        s.sort();
        s.dedup();
        s.truncate(SOURCE_LIMIT);
        e.meta.sources = s;
        e.meta.signature = signature;
        self.touch(&key);
        true
    }

    pub fn evidence(&mut self, key: &Key) -> Option<Evidence> {
        if !self.rows.contains(key) {
            return None;
        }
        self.expire(key);
        let mask = self.mask;
        let e = self.rows.get_mut(key).unwrap();
        if e.cache.is_none() {
            let mut row = e.row;
            for bank in [&e.terminal, &e.teacher] {
                if let Some(b) = bank {
                    row = [add(&row[0], &b[0], mask), add(&row[1], &b[1], mask)];
                }
            }
            let min = minimum(mask);
            let (pos, neg) = (&row[0], &row[1]);
            let good = !greater(&min, pos, mask) & greater(pos, neg, mask);
            let bad = !greater(&min, neg, mask) & greater(neg, pos, mask);
            let margin = if good != 0 { Some(difference(pos, neg, mask)) } else { None };
            e.cache = Some(Evidence { good, bad, margin });
        }
        e.cache
    }

    pub fn choose(&mut self, key: &Key, proposed: usize) -> Option<usize> {
        let found = self.evidence(key)?;
        let mut winners = found.good;
        let rejected = found.bad;
        if winners == 0 {
            if rejected & (1u64 << proposed) == 0 {
                return None;
            }
            winners = !rejected & self.mask;
        }
        if let Some(margin) = found.margin {
            for plane in margin.iter().rev() {
                let w = winners & plane;
                if w != 0 {
                    winners = w;
                }
            }
        }
        for offset in 0..self.actions {
            let action = (proposed + offset) % self.actions;
            if (winners >> action) & 1 == 1 {
                return Some(action);
            }
        }
        None
    }

    /// Hex of all planes, one little-endian group of `(actions+7)/8` bytes per value.
    pub fn pack(&self, row: &Row) -> String {
        let size = (self.actions + 7) / 8;
        let mut s = String::with_capacity(PLANES * 2 * size * 2);
        const HEX: &[u8; 16] = b"0123456789abcdef";
        for g in row.iter() {
            for &v in g.iter() {
                for b in 0..size {
                    let byte = if b < 8 { (v >> (8 * b)) as u8 } else { 0 };
                    s.push(HEX[(byte >> 4) as usize] as char);
                    s.push(HEX[(byte & 15) as usize] as char);
                }
            }
        }
        s
    }

    pub fn unpack(&self, hex: &str) -> Option<Row> {
        let size = (self.actions + 7) / 8;
        let raw = hex_decode(hex)?;
        if raw.len() != 2 * PLANES * size {
            return None;
        }
        let mut row = EMPTY;
        for i in 0..2 * PLANES {
            let mut v: u64 = 0;
            for b in 0..size {
                let byte = raw[i * size + b] as u64;
                if b >= 8 {
                    if byte != 0 { return None; }
                    continue;
                }
                v |= byte << (8 * b);
            }
            if v > self.mask {
                return None;
            }
            row[i / PLANES][i % PLANES] = v;
        }
        Some(row)
    }

    pub fn packed(&mut self, key: &Key) -> (String, Option<String>, Option<String>) {
        if let Some(p) = self.rows.get(key).and_then(|e| e.packed.clone()) {
            return p;
        }
        let e = self.rows.get(key).unwrap();
        let p = (self.pack(&e.row), e.terminal.as_ref().map(|t| self.pack(t)), e.teacher.as_ref().map(|t| self.pack(t)));
        self.rows.get_mut(key).unwrap().packed = Some(p.clone());
        p
    }
}

/// Python's bytes.fromhex accepts whitespace between bytes and both letter cases.
pub fn hex_decode(s: &str) -> Option<Vec<u8>> {
    let digits: Vec<u8> = s.bytes().filter(|c| !c.is_ascii_whitespace()).collect();
    if digits.len() % 2 != 0 {
        return None;
    }
    let val = |c: u8| -> Option<u8> {
        match c {
            b'0'..=b'9' => Some(c - b'0'),
            b'a'..=b'f' => Some(c - b'a' + 10),
            b'A'..=b'F' => Some(c - b'A' + 10),
            _ => None,
        }
    };
    digits.chunks(2).map(|p| Some(val(p[0])? << 4 | val(p[1])?)).collect()
}
