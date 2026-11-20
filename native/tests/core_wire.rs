//! Shared golden lanes also consumed by the Python tensor codec test.
use monade_core::word::Word;

#[test]
fn predatore_value_and_known_mask_have_identical_word_order() {
    let rows: Vec<_> = include_str!("../../validation/exchange/tensor_words.txt")
        .lines().filter(|line| !line.starts_with('#')).map(|line| {
            let bytes: Vec<_> = (0..line.len()).step_by(2)
                .map(|i| u8::from_str_radix(&line[i..i + 2], 16).unwrap()).collect();
            bytes.chunks_exact(8)
                .map(|chunk| Word::from_bytes(chunk.try_into().unwrap())).collect::<Vec<_>>()
        }).collect();
    assert_eq!(rows[0], [Word::new(1 | (1 << 63)), Word::new(1 << 7)]);
    assert_eq!(rows[1], [Word::new(u64::MAX), Word::new(0x7f)]);
    assert_eq!(rows[0][0].and(rows[1][0]).popcount(), 2);
    assert_eq!(rows[0][1].and(rows[1][1]), Word::ZERO);
}
