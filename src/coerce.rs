use std::str::Utf8Error;

#[repr(C)]
pub struct Coerce<T: ?Sized> {
    pub len: usize,
    pub buf: T,
}

#[derive(Debug)]
pub enum CoerceError {
    BufStartIsNotProperlyAligned,
    IndicesOrder,
    NonUtf8 {
        offset: usize,
        inner: Utf8Error,
    }
}
