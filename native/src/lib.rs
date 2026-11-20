//! Predatore V7 native episode kernel.
//!
//! The Python package stays the reference implementation; this crate reproduces it
//! bit for bit (same V7 generator, same memories, same decisions) and runs whole
//! episodes per call. Monade words and the Tesseract generator are shared foundations.
pub mod bits;
pub mod canon;
pub mod circuit;
pub mod hdc;
pub mod key;
pub mod lru;
pub mod memory;
pub mod world;
#[cfg(feature = "python")]
pub mod py;
