//! Thrift compact protocol, write side only, just what Parquet metadata needs.

pub const BOOL_TRUE: u8 = 1;
pub const BOOL_FALSE: u8 = 2;
pub const I32: u8 = 5;
pub const I64: u8 = 6;
pub const BINARY: u8 = 8;
pub const LIST: u8 = 9;
pub const STRUCT: u8 = 12;

#[inline]
pub fn varint(out: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        out.push(v as u8 | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

#[inline]
pub fn zigzag(v: i64) -> u64 {
    ((v << 1) ^ (v >> 63)) as u64
}

/// A compact-protocol writer. Fields must be written in increasing id order within a struct.
pub struct Thrift<'a> {
    pub out: &'a mut Vec<u8>,
    stack: Vec<i16>,
    last: i16,
}

impl<'a> Thrift<'a> {
    pub fn new(out: &'a mut Vec<u8>) -> Self {
        Thrift { out, stack: Vec::new(), last: 0 }
    }

    fn header(&mut self, id: i16, ty: u8) {
        let d = id - self.last;
        if d > 0 && d <= 15 {
            self.out.push(((d as u8) << 4) | ty);
        } else {
            self.out.push(ty);
            varint(self.out, zigzag(id as i64));
        }
        self.last = id;
    }

    pub fn i32(&mut self, id: i16, v: i32) {
        self.header(id, I32);
        varint(self.out, zigzag(v as i64));
    }

    pub fn i64(&mut self, id: i16, v: i64) {
        self.header(id, I64);
        varint(self.out, zigzag(v));
    }

    pub fn bool(&mut self, id: i16, v: bool) {
        self.header(id, if v { BOOL_TRUE } else { BOOL_FALSE });
    }

    pub fn binary(&mut self, id: i16, b: &[u8]) {
        self.header(id, BINARY);
        varint(self.out, b.len() as u64);
        self.out.extend_from_slice(b);
    }

    /// Open a struct-valued field.
    pub fn begin(&mut self, id: i16) {
        self.header(id, STRUCT);
        self.stack.push(self.last);
        self.last = 0;
    }

    /// Close a struct (field or list element).
    pub fn end(&mut self) {
        self.out.push(0);
        self.last = self.stack.pop().expect("unbalanced thrift struct");
    }

    /// Start a list-valued field of `n` elements of compact type `elem`.
    pub fn list(&mut self, id: i16, elem: u8, n: usize) {
        self.header(id, LIST);
        if n < 15 {
            self.out.push(((n as u8) << 4) | elem);
        } else {
            self.out.push(0xF0 | elem);
            varint(self.out, n as u64);
        }
    }

    /// Open a struct that is a list element.
    pub fn elem_begin(&mut self) {
        self.stack.push(self.last);
        self.last = 0;
    }

    pub fn elem_i32(&mut self, v: i32) {
        varint(self.out, zigzag(v as i64));
    }

    pub fn elem_binary(&mut self, b: &[u8]) {
        varint(self.out, b.len() as u64);
        self.out.extend_from_slice(b);
    }

    /// End of the top-level struct.
    pub fn stop(&mut self) {
        assert!(self.stack.is_empty(), "unclosed thrift struct");
        self.out.push(0);
    }
}
