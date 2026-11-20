//! Context keys: unsigned integers up to 256 bits, written as decimal strings in Python.

/// Little-endian 256-bit unsigned integer.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug, Default)]
pub struct Key(pub [u64; 4]);

impl Key {
    pub const ZERO: Key = Key([0; 4]);

    pub fn from_u64(v: u64) -> Key {
        Key([v, 0, 0, 0])
    }

    pub fn bit_length(&self) -> u32 {
        for i in (0..4).rev() {
            if self.0[i] != 0 {
                return 64 * i as u32 + (64 - self.0[i].leading_zeros());
            }
        }
        0
    }

    /// `self | (v << shift)` for a value that fits in 64 bits.
    pub fn or_shifted(&mut self, v: u64, shift: u32) {
        if v == 0 {
            return;
        }
        let w = (shift / 64) as usize;
        let b = shift % 64;
        if w < 4 {
            self.0[w] |= v << b;
        }
        if b != 0 && w + 1 < 4 {
            self.0[w + 1] |= v >> (64 - b);
        }
    }

    /// Bits `[shift, shift+len)` as u64 (len <= 64).
    pub fn bits(&self, shift: u32, len: u32) -> u64 {
        if len == 0 {
            return 0;
        }
        let w = (shift / 64) as usize;
        let b = shift % 64;
        let mut v = if w < 4 { self.0[w] >> b } else { 0 };
        if b != 0 && w + 1 < 4 {
            v |= self.0[w + 1] << (64 - b);
        }
        if len < 64 {
            v &= (1u64 << len) - 1;
        }
        v
    }

    pub fn parse(s: &str) -> Option<Key> {
        if s.is_empty() || !s.bytes().all(|c| c.is_ascii_digit()) {
            return None;
        }
        let mut k = [0u64; 4];
        for c in s.bytes() {
            let mut carry = (c - b'0') as u128;
            for word in k.iter_mut() {
                let v = (*word as u128) * 10 + carry;
                *word = v as u64;
                carry = v >> 64;
            }
            if carry != 0 {
                return None;
            }
        }
        Some(Key(k))
    }

    pub fn to_decimal(&self) -> String {
        if *self == Key::ZERO {
            return "0".to_string();
        }
        let mut k = self.0;
        let mut digits = Vec::new();
        while k != [0; 4] {
            let mut rem: u128 = 0;
            for i in (0..4).rev() {
                let cur = (rem << 64) | k[i] as u128;
                k[i] = (cur / 10) as u64;
                rem = cur % 10;
            }
            digits.push(b'0' + rem as u8);
        }
        digits.reverse();
        String::from_utf8(digits).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn decimal_round_trip() {
        for s in ["0", "7", "18446744073709551616", "340282366920938463463374607431768211457"] {
            assert_eq!(Key::parse(s).unwrap().to_decimal(), s);
        }
    }
}
