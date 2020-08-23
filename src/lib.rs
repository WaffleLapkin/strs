

#![allow(dead_code)]
#![allow(unused_variables)]
use core::borrow::Borrow;
use std::alloc::*; // TODO
use core::ptr;
use core::mem;
use std::mem::transmute;
use std::str::Utf8Error;

mod coerce;

pub use coerce::{Coerce, CoerceError};

#[repr(C)]
pub struct Strs {
    /// Number of **usize**s in `buf` those are str indices
    len: usize,
    /// ## Invariants
    ///
    /// 1. `buf[..len * size_of::<usize>()]` is valid `[usize]` i.e
    ///   ```ignore
    ///   buf[..len * core::mem::size_of::<usize>()].align_to::<usize>()
    ///   ```
    ///   Returns `([], xs, [])` where `xs` is a `usize` slice
    ///   - `xs` must be sorted such that `xs[n] <= xs[n + 1]`
    /// 2. `buf[len * size_of::<usize>()..]` is valid utf-8 string
    /// 3. `buf[x..y]` such that `x, y ∈ indices` and `x <= y` is also valid utf-8 string
    buf: [u8],
}

impl<T: Borrow<str>> From<Vec<T>> for Box<Strs> {
    fn from(vec: Vec<T>) -> Self {
        // TODO: borrow can be malicious
        let size: usize = vec.iter().map(|s| s.borrow().len()).sum();
        let indices = vec.len() + 1;

        let layout = {
            let base = Layout::new::<usize>();
            let (with_usizes, offset0) = base.extend(Layout::array::<usize>(indices).unwrap()).unwrap();
            let (full, offset1) = with_usizes.extend(Layout::array::<u8>(size).unwrap()).unwrap();

            full.pad_to_align()
        };

        // ## Safety
        //
        // Layout size can't be 0 because we've extended `Layout` of `usize`
        let ptr = unsafe { alloc(layout) };

        if ptr.is_null() {
            #[allow(unreachable_code)]
            return handle_alloc_error(layout); // returns `!`
        }

        // write `len`
        //
        // ## Safety
        //
        // - `len: usize` is the first field of repr(C) struct we've
        //   just alloced
        // - ptr must be valid for write because we've just `alloc`ed it
        //
        unsafe {
            ptr::write(ptr.cast::<usize>(), vec.len());
        }

        let buf_ptr = unsafe { ptr.cast::<usize>().offset(1).cast::<u8>() };

        // offset from `buf_ptr` to the empty place
        let mut offset = indices * mem::size_of::<usize>();
        for (idx, s) in vec.into_iter().enumerate() {
            let str = s.borrow();


            unsafe {
                buf_ptr
                    .cast::<usize>()
                    .add(idx)
                    .write(offset);
            }

            // ## Safety
            unsafe {
                ptr::copy_nonoverlapping(
                    str.as_bytes().as_ptr(),
                    buf_ptr.add(offset),
                    str.len(), // TODO: check that we didn't exceed limit
                )
            }

            offset += str.len();
        }

        unsafe {
            buf_ptr
                .cast::<usize>()
                .add(indices - 1)
                .write(offset);
        }

        unsafe {
            let ptr = core::slice::from_raw_parts_mut(
                ptr as *mut (),
                dbg!(size + indices * mem::size_of::<usize>())
            ) as *mut [()] as *mut Strs;
            Box::from_raw(ptr)
        }
    }
}

impl Strs {
    /// Empty [`Strs`] e.i. [`Strs`] that doesn't contain any strings
    ///
    /// ```
    /// use strs::Strs;
    ///
    /// assert_eq!(Strs::EMPTY.len(), 0);
    /// assert_eq!(Strs::EMPTY.get(0), None);
    /// assert_eq!(Strs::EMPTY.as_str(), "");
    /// ```
    pub const EMPTY: &'static Strs = {
        let coerce: &Coerce<[u8]> = &Coerce {
            len: 0,
            buf: 0usize.to_ne_bytes()
        };

        unsafe {
            // Not using `coerce_unchecked` here because `transmute` is still unstable in `const fn`
            transmute::<&Coerce<[u8]>, &Strs>(coerce)
        }
    };

    ///
    ///
    /// ## Example
    ///
    /// ```
    /// use strs::{Coerce, Strs};
    ///
    /// // Size can vary on different architectures
    /// const SIZE: usize = core::mem::size_of::<usize>();
    /// let mut buf = [0; SIZE * 3 + 11];
    ///
    /// buf[..SIZE].copy_from_slice(&(SIZE * 3).to_ne_bytes());
    /// buf[SIZE..SIZE * 2].copy_from_slice(&(SIZE * 3 + 6).to_ne_bytes());
    /// buf[SIZE * 2..SIZE * 3].copy_from_slice(&(SIZE * 3 + 11).to_ne_bytes());
    ///
    /// //                                       ______*--- second part
    /// buf[SIZE * 3..].copy_from_slice(b"Hello world");
    /// //                                 ^^^^^^*--- first part
    ///
    /// let val = &Coerce { len: 2, buf };
    /// let strs = Strs::coerce(val).unwrap();
    /// assert_eq!(&strs[0], "Hello ");
    /// assert_eq!(&strs[1], "world");
    /// ```
    pub fn coerce(val: &Coerce<[u8]>) -> Result<&Self, CoerceError> {
        let Coerce { len: len, buf } = val;

        // Invariant check: beginning of the `buf` is valid `[usize]`
        let (a, indices, b) = unsafe {
            buf[..(len + 1) * core::mem::size_of::<usize>()].align_to::<usize>()
        };

        if !a.is_empty() || indices.len() != *len + 1 || !b.is_empty() {
            return Err(CoerceError::BufStartIsNotProperlyAligned);
        }

        // Result<&str, Utf8Error> -> Result<(), CoerceError>
        let utferr = |res: Result<&str, Utf8Error>| res.map(drop).map_err(|err| CoerceError::NonUtf8 {
            offset: len * core::mem::size_of::<usize>(),
            inner: err,
        });

        // Check invariant: end of the buf is valid utf8 string
        utferr(core::str::from_utf8(&buf[(len + 1) * core::mem::size_of::<usize>()..]))?;

        // Check invariants:
        // - indices are sorted
        // - all parts are valid utf8 strings
        indices
            .windows(2)
            .try_for_each(|s| match s {
                [a, b] if a > b => Err(CoerceError::IndicesOrder),
                &[a, b] => utferr(core::str::from_utf8(&buf[a..b])),
                _ => Ok(()), // Unreachable
            })?;

        // ## Safety
        //
        // We've just checked all invariants
        unsafe {
            Ok(Self::coerce_unchecked(val))
        }
    }

    pub unsafe fn coerce_unchecked(val: &Coerce<[u8]>) -> &Self {
        transmute::<&Coerce<[u8]>, &Strs>(val)
    }

    pub fn as_str(&self) -> &str {
        // if self.len() == 0 {
        //     return "";
        // }

        unsafe {
            // let from = *self
            //     .buf
            //     .as_ptr()
            //     .cast::<usize>();
            let from = mem::size_of::<usize>() * self.len + mem::size_of::<usize>();

            let bytes = &self.buf[from..];
            core::str::from_utf8_unchecked(bytes)
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn get(&self, idx: usize) -> Option<&str> {
        if self.len() > idx {
            Some(unsafe { self.get_unchecked(idx) })
        } else {
            None
        }
    }

    pub unsafe fn get_unchecked(&self, idx: usize) -> &str {
        let (from, to) = self.get_bounds(idx);
        let bytes = &self.buf[from..to];

        core::str::from_utf8_unchecked(bytes)
    }

    unsafe fn get_bounds(&self, idx: usize) -> (usize, usize) {
        let ptr = self
            .buf
            .as_ptr()
            .cast::<usize>()
            .add(idx);

        (*ptr, *ptr.add(1))
    }
}


impl std::ops::Index<usize> for Strs {
    type Output = str;

    // TODO: #[track_caller]
    fn index(&self, idx: usize) -> &str {
        match self.get(idx) {
            Some(ret) => ret,
            None => panic!("index out of bounds: the len is {} but the index is {}", self.len(), idx),
        }
    }
}

#[macro_export]
macro_rules! strs {
    ($out:ident = $( $lits:literal ),*) => {
        let val = &{
            const SIZE: usize = ::core::mem::size_of::<usize>();
            const count: usize = $({
                let _ = $lits;
                1
            } + )* 1;

            const len: usize = $( str::len($lits) + )* 0;

            let mut buf = [0; SIZE * count + len];

            {
                let mut offset = 0;
                let mut acc = SIZE * count;
                $(
                    buf[SIZE * offset..SIZE * (offset + 1)].copy_from_slice(&acc.to_ne_bytes());
                    buf[acc..acc + str::len($lits)].copy_from_slice(str::as_bytes($lits));
                    acc += str::len($lits);
                    offset += 1;
                )*

                buf[SIZE * offset..SIZE * (offset + 1)].copy_from_slice(&acc.to_ne_bytes());
            }

            $crate::Coerce { len: (count - 1), buf }
        };

        let $out: &$crate::Strs = $crate::Strs::coerce(val).unwrap();
    }
}

#[test]
fn miri() {
    strs![strs = "nWCrGwqS8D64upzLfC4b", "6PVhjouHIDefr6jIyWqy"];

    assert_eq!(&strs[0], "nWCrGwqS8D64upzLfC4b");
    assert_eq!(&strs[1], "6PVhjouHIDefr6jIyWqy");
    assert_eq!(strs.get(2), None);
    assert_eq!(strs.len(), 2);
}