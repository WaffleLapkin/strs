use core::fmt;
use std::{error::Error, str::Utf8Error};

#[derive(Debug)]
pub enum FromSliceError {
    IndicesOrder,
    NonUtf8 { offset: usize, inner: Utf8Error },
}

impl fmt::Display for FromSliceError {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IndicesOrder => write!(f, "Indices are out of order"),
            FromSliceError::NonUtf8 { offset, inner } => {
                write!(f, "Data is not valid utf-8 at iffset {}: {}", offset, inner)
            }
        }
    }
}

impl Error for FromSliceError {
    #[inline]
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::IndicesOrder => None,
            FromSliceError::NonUtf8 { offset: _, inner } => Some(inner),
        }
    }
}

#[test]
fn iter() {
    use crate::Strs;

    assert_eq!(<Strs>::EMPTY.iter().collect::<Vec<_>>(), Vec::<&str>::new());
    assert_eq!(<Strs>::EMPTY.iter().len(), 0);

    assert_eq!(
        <Strs>::boxed(&["a", "b'", "ccc"])
            .iter()
            .collect::<Vec<_>>(),
        vec!["a", "b'", "ccc"]
    );
    assert_eq!(
        <Strs>::boxed(&["x", "y", "z"])
            .iter()
            .rev()
            .collect::<Vec<_>>(),
        vec!["z", "y", "x"]
    );

    let strs = <Strs>::boxed(&["x", "y", "z", "a", "b"]);
    let mut iter = strs.iter();
    assert_eq!(iter.len(), 5);
    assert_eq!(iter.next(), Some("x"));
    assert_eq!(iter.next(), Some("y"));
    assert_eq!(iter.next_back(), Some("b"));
    assert_eq!(iter.len(), 2);
    assert_eq!(iter.next(), Some("z"));
    assert_eq!(iter.next_back(), Some("a"));
    assert_eq!(iter.len(), 0);
    assert_eq!(iter.next_back(), None);
    assert_eq!(iter.next(), None);
}
