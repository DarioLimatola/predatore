//! Offline P2 trial. One bounded request contains a whole episode.
use monade_core::{size::Size, word::Word};
use std::io::{self, Read, Write};

const END: u32 = u32::MAX;
const MAX_BYTES: usize = 4 * 1024 * 1024;
type Result<T> = std::result::Result<T, &'static str>;

struct Input<'a>(&'a [u8]);
impl Input<'_> {
    fn take<const N: usize>(&mut self) -> Result<[u8; N]> {
        let (a, b) = self.0.split_at_checked(N).ok_or("truncated request")?;
        self.0 = b;
        Ok(a.try_into().unwrap())
    }
    fn int(&mut self) -> Result<usize> {
        Ok(u32::from_le_bytes(self.take()?) as usize)
    }
    fn bounded(&mut self, min: usize, max: usize) -> Result<usize> {
        let n = self.int()?;
        if n < min || n > max {
            Err("out-of-range field")
        } else {
            Ok(n)
        }
    }
    fn word(&mut self) -> Result<Word> {
        Ok(Word::from_bytes(self.take()?))
    }
}
struct Rule {
    dest: usize,
    a: usize,
    b: usize,
    hold: usize,
    sa: u32,
    sb: u32,
    write: Word,
    lut: [Word; 4],
}

fn execute(bytes: &[u8]) -> Result<Vec<u8>> {
    let mut src = Input(bytes);
    if src.take::<8>()? != *b"PDLUT1\0\0" {
        return Err("unrecognized protocol");
    }
    if src.int()? as u64 != Size::WORD.bits() {
        return Err("needs one Monade Word");
    }
    let sensors = src.bounded(1, 32)?;
    let count = src.bounded(1, 4096)?;
    let active = src.bounded(0, count)?;
    let ticks = src.bounded(1, 8)?;
    let outputs = src.bounded(1, 8)?;
    let flags = src.bounded(0, 7)?; // trace vectors, broadcast sensors, memory enabled
    let table = src.bounded(0, 1)? != 0;
    let rows = src.bounded(1, 65536)?;
    let horizon = src.bounded(1, 65536)?;
    let mut cursor = src.bounded(0, rows - 1)?;
    if !table && (cursor != 0 || rows != horizon) {
        return Err("invalid tape");
    }
    if table && outputs != 3 {
        return Err("world contract requires eight actions");
    }
    let stride = 1 + if flags & 1 != 0 { 8 * count } else { 0 };
    if stride * horizon > MAX_BYTES {
        return Err("trace budget exceeded");
    }
    let mut rules = Vec::with_capacity(active);
    let mut seen = vec![false; count];
    for _ in 0..active {
        let dest = src.bounded(0, count - 1)?;
        if seen[dest] {
            return Err("duplicate destination");
        }
        seen[dest] = true;
        rules.push(Rule {
            dest,
            a: src.bounded(0, sensors + count - 1)?,
            b: src.bounded(0, sensors + count - 1)?,
            hold: src.bounded(0, sensors + count)?,
            sa: src.bounded(0, 63)? as u32,
            sb: src.bounded(0, 63)? as u32,
            write: src.word()?,
            lut: [src.word()?, src.word()?, src.word()?, src.word()?],
        });
    }
    let mut readout = Vec::with_capacity(outputs);
    for _ in 0..outputs {
        readout.push((src.bounded(0, count - 1)?, src.bounded(0, 63)?));
    }
    let mut observations = Vec::with_capacity(rows);
    let mut edges = Vec::new();
    for _ in 0..rows {
        let obs = src.word()?;
        if obs.bits() >> sensors != 0 {
            return Err("sensor padding");
        }
        observations.push(obs);
        if table {
            let mut row = [0u32; 8];
            for next in &mut row {
                *next = src.int()? as u32;
                if *next != END && *next as usize >= rows {
                    return Err("invalid world edge");
                }
            }
            edges.push(row);
        }
    }
    if !src.0.is_empty() {
        return Err("trailing bytes");
    }
    let mut inputs = vec![Word::ZERO; sensors + count];
    let mut state = vec![Word::ZERO; count];
    let mut out = Vec::with_capacity(stride * horizon);
    for _ in 0..horizon {
        let obs = observations[cursor].bits();
        for (s, input) in inputs[..sensors].iter_mut().enumerate() {
            let mask = 0u64.wrapping_sub((obs >> s) & 1);
            let mixed = if flags & 2 != 0 {
                0
            } else {
                obs.rotate_left(s as u32)
            };
            *input = Word::new(mixed ^ mask);
        }
        for _ in 0..ticks {
            if flags & 4 != 0 {
                inputs[sensors..].copy_from_slice(&state);
            }
            for r in &rules {
                let a = inputs[r.a].bits().rotate_left(r.sa);
                let b = inputs[r.b].bits().rotate_left(r.sb);
                let mask = if r.hold != 0 && flags & 4 != 0 {
                    r.write.bits() & !inputs[r.hold - 1].bits()
                } else {
                    r.write.bits()
                };
                let value = (!a & !b & r.lut[0].bits())
                    | (!a & b & r.lut[1].bits())
                    | (a & !b & r.lut[2].bits())
                    | (a & b & r.lut[3].bits());
                state[r.dest] =
                    Word::new((inputs[sensors + r.dest].bits() & !mask) | (value & mask));
            }
        }
        let mut action = 0u8;
        for (bit, &(node, lane)) in readout.iter().enumerate() {
            action |= (((state[node].bits() >> lane) & 1) as u8) << bit;
        }
        out.push(action);
        if flags & 1 != 0 {
            for v in &state {
                out.extend_from_slice(&v.to_bytes());
            }
        }
        if table {
            let next = edges[cursor][action as usize];
            if next == END {
                break;
            }
            cursor = next as usize;
        } else {
            cursor += 1;
        }
    }
    Ok(out)
}

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let mut data = Vec::new();
    io::stdin()
        .lock()
        .take(MAX_BYTES as u64 + 1)
        .read_to_end(&mut data)?;
    if data.len() > MAX_BYTES {
        return Err("request budget exceeded".into());
    }
    let out = execute(&data)?;
    io::stdout().lock().write_all(&out)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_malformed() {
        for blob in [b"".as_slice(), b"PDLUT1\0\0", b"garbage!"] {
            assert!(execute(blob).is_err());
        }
    }
}
