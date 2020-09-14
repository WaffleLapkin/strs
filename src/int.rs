use std::marker::PhantomData;

pub use crate::{trusted_idx::TrustedIdx, Strs};

/// Source needed to initialize [`Strs`](crate::Strs).
///
/// This structure is used to 'cache' required size computations
pub struct Source<'a, S, Idx: TrustedIdx> {
    slice: &'a [S],
    req: usize,
    size: usize,
    __: PhantomData<Idx>,
}

impl<'a, S: AsRef<str>, Idx: TrustedIdx> Source<'a, S, Idx> {
    /// Creates a new source that can be later used to initialize [`Strs`](crate::Strs)
    pub fn new(slice: &'a [S]) -> Self {
        let (req, size) = Strs::<Idx>::required_idxes_for_and_size(slice);
        Self {
            slice,
            req,
            size,
            __: PhantomData,
        }
    }

    /// Get underling slice that was previously passed to [`new`](Self::new)
    pub fn slice(&self) -> &'a [S] {
        self.slice
    }

    /// Return number of `Idx`es needed to store/init [`Strs`](crate::Strs)
    pub fn required_idxes(&self) -> usize {
        self.req
    }

    pub(crate) fn size(&self) -> usize {
        self.size
    }
}
