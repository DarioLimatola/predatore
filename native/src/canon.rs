//! Canonical JSON identical to Python's ``json.dumps(obj, sort_keys=True)`` (ensure_ascii,
//! separators ', ' and ': '), for content digests computed without leaving Rust.
use std::fmt::Write;

pub fn string(out: &mut String, s: &str) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 || (c as u32) > 0x7e => {
                let mut buf = [0u16; 2];
                for unit in c.encode_utf16(&mut buf) {
                    let _ = write!(out, "\\u{:04x}", unit);
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Decimal of an unsigned little-endian multiword integer.
pub fn decimal(v: &[u64]) -> String {
    let mut words: Vec<u64> = v.to_vec();
    while words.last() == Some(&0) {
        words.pop();
    }
    if words.is_empty() {
        return "0".into();
    }
    let mut chunks: Vec<u64> = Vec::new(); // base 1e19
    const BASE: u128 = 10_000_000_000_000_000_000;
    while !words.is_empty() {
        let mut rem: u128 = 0;
        for i in (0..words.len()).rev() {
            let cur = (rem << 64) | words[i] as u128;
            words[i] = (cur / BASE) as u64;
            rem = cur % BASE;
        }
        chunks.push(rem as u64);
        while words.last() == Some(&0) {
            words.pop();
        }
    }
    let mut s = chunks.last().unwrap().to_string();
    for c in chunks.iter().rev().skip(1) {
        let _ = write!(s, "{:019}", c);
    }
    s
}

pub fn sha256_hex(data: &str) -> String {
    use sha2::{Digest, Sha256};
    let d = Sha256::digest(data.as_bytes());
    let mut s = String::with_capacity(64);
    for b in d {
        let _ = write!(s, "{:02x}", b);
    }
    s
}
