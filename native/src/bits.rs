//! Fixed-width bit vectors stored as little-endian u64 words (bit i = word i/64, bit i%64).
//! All results are masked to `width` unless stated otherwise.

#[inline]
pub fn words(width: usize) -> usize {
    (width + 63) / 64
}

#[inline]
pub fn top_mask(width: usize) -> u64 {
    let r = width % 64;
    if r == 0 { u64::MAX } else { (1u64 << r) - 1 }
}

#[inline]
pub fn mask_in_place(v: &mut [u64], width: usize) {
    let n = words(width);
    for x in v.iter_mut().skip(n) {
        *x = 0;
    }
    if n > 0 {
        v[n - 1] &= top_mask(width);
    }
}

/// out = src << s (unbounded inside out.len() words, then caller masks).
pub fn shl(src: &[u64], s: usize, out: &mut [u64]) {
    let ws = s / 64;
    let bs = s % 64;
    for i in (0..out.len()).rev() {
        let mut v = 0u64;
        if i >= ws {
            let j = i - ws;
            if j < src.len() {
                v = src[j] << bs;
            }
            if bs != 0 && j >= 1 && j - 1 < src.len() {
                v |= src[j - 1] >> (64 - bs);
            }
        }
        out[i] = v;
    }
}

/// out = src >> s.
pub fn shr(src: &[u64], s: usize, out: &mut [u64]) {
    let ws = s / 64;
    let bs = s % 64;
    for i in 0..out.len() {
        let j = i + ws;
        let mut v = if j < src.len() { src[j] >> bs } else { 0 };
        if bs != 0 && j + 1 < src.len() {
            v |= src[j + 1] << (64 - bs);
        }
        out[i] = v;
    }
}

/// Python: ((v << s) | (v >> (width - s))) & mask with s = shift % width; `v` may hold
/// bits above `width` (sensor mixing uses the whole observation), exactly as in Python.
pub fn rotate(v: &[u64], shift: usize, width: usize, out: &mut [u64], tmp: &mut [u64]) {
    let s = shift % width;
    let n = out.len();
    if s == 0 {
        for i in 0..n {
            out[i] = if i < v.len() { v[i] } else { 0 };
        }
        mask_in_place(out, width);
        return;
    }
    shl(v, s, out);
    shr(v, width - s, tmp);
    for i in 0..n {
        out[i] |= tmp[i];
    }
    mask_in_place(out, width);
}

#[inline]
pub fn popcount(v: &[u64]) -> u32 {
    v.iter().map(|x| x.count_ones()).sum()
}

#[inline]
pub fn is_zero(v: &[u64]) -> bool {
    v.iter().all(|&x| x == 0)
}

/// Python ``hex(int)``: '0x' + minimal lowercase digits.
pub fn to_hex(v: &[u64]) -> String {
    let mut i = v.len();
    while i > 0 && v[i - 1] == 0 {
        i -= 1;
    }
    if i == 0 {
        return "0x0".into();
    }
    let mut s = format!("0x{:x}", v[i - 1]);
    for j in (0..i - 1).rev() {
        s.push_str(&format!("{:016x}", v[j]));
    }
    s
}

/// Python ``int(s, 16)`` for non-negative values (optional 0x prefix, underscores allowed).
pub fn from_hex(s: &str, n: usize) -> Option<Vec<u64>> {
    let t = s.trim();
    let t = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")).unwrap_or(t);
    let digits: Vec<u8> = t.bytes().filter(|&c| c != b'_').collect();
    if digits.is_empty() {
        return None;
    }
    let mut out = vec![0u64; n];
    for (pos, &c) in digits.iter().rev().enumerate() {
        let d = (c as char).to_digit(16)? as u64;
        let w = pos / 16;
        if w >= n {
            if d != 0 { return None; }
            continue;
        }
        out[w] |= d << (4 * (pos % 16));
    }
    Some(out)
}

/// Bit at position i.
#[inline]
pub fn bit(v: &[u64], i: usize) -> u64 {
    (v[i / 64] >> (i % 64)) & 1
}
