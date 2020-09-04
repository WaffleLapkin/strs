use crate::{int::TrustedIdx, Strs};

/// Iterator over strings in [`Strs`]
///
/// See [`Strs::iter`]
#[derive(Debug)]
pub struct Iter<'a, Idx: TrustedIdx> {
    strs: &'a Strs<Idx>,
    // TODO: probably it would be better to implement this with pointers to idx buffer
    forth_idx: Idx,
    back_idx: Idx,
}

impl<'a, Idx: TrustedIdx> Iter<'a, Idx> {
    pub(crate) fn new(strs: &'a Strs<Idx>) -> Self {
        Self {
            strs,
            forth_idx: Idx::ZERO,
            back_idx: strs.len(),
        }
    }
}

impl<'a, Idx: TrustedIdx> Iterator for Iter<'a, Idx> {
    type Item = &'a str;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.forth_idx == self.back_idx {
            return None;
        }

        unsafe {
            // ## Safety
            //
            // `forth_idx` is initialized with 0 and only increases,
            // `bach_idx` is initialized with strs.len() and only decreases,
            // If `forth_idx == back_idx` then they never change again.
            //
            // This means that 0 <= forth_idx < strs.len() and thus it's safe
            let res = Some(self.strs.get_unchecked(self.forth_idx));
            self.forth_idx = self.forth_idx + Idx::ONE;
            res
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }
}

impl<'a, Idx: TrustedIdx> DoubleEndedIterator for Iter<'a, Idx> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.forth_idx == self.back_idx {
            return None;
        }

        unsafe {
            // ## Safety
            //
            // `forth_idx` is initialized with 0 and only increases,
            // `bach_idx` is initialized with strs.len() and only decreases,
            // If `forth_idx == back_idx` then they never change again.
            //
            // This means that 0 <= (back_idx - 1) < strs.len() and thus it's safe
            self.back_idx.dec();
            Some(self.strs.get_unchecked(self.back_idx))
        }
    }
}

impl<Idx: TrustedIdx> ExactSizeIterator for Iter<'_, Idx> {
    #[inline]
    fn len(&self) -> usize {
        (self.back_idx - self.forth_idx).as_usize()
    }

    #[inline]
    #[cfg(feature = "nightly")]
    fn is_empty(&self) -> bool {
        self.back_idx == self.forth_idx
    }
}
