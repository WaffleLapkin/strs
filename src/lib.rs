//! TODO: crate docs
#![cfg_attr(feature = "nightly", feature(exact_size_is_empty))]
//#![deny(missing_docs)] // TODO
use core::{
    cmp::Ordering,
    convert::AsRef,
    fmt,
    hash::{Hash, Hasher},
    iter,
    mem::{self, transmute, MaybeUninit},
    ops::DerefMut,
    ptr, slice,
    str::Utf8Error,
};
use std::{error::Error, process::abort, rc::Rc, sync::Arc};

/// Collection of strings.
///
/// More precisely it's dynamic (you can create one at runtime) immutable slice of strings those
/// are stored (among with indices) in a continuous chunk of memory.
///
/// `Strs` is [dst] (dynamically sized type) so you can't store it on stack (well, not really but
/// we'll cover that later), instead you would need to create a `Strs` wrapped in [`Box<_>`][box],
/// [`Arc<_>`][arc], [`Rc<_>`][rc] or common ref `&_`, see [`boxed`], [`arced`], [`rced`].
///
/// The main use case for this struct is collections of strings those are once created and then
/// don't change. This allows to occupy less space in comparison with `[String]` (slice of strings
/// takes 3 words per string of overhead whereas `Strs` _currently_ takes 2 words per string +
/// 1 word in any case). Additionally this removes 1 level of indirection because all string are
/// contained inline.
///
/// `Rc` and `Arc` wrapped `Strs`es are particularly useful due to cheap cloning.
///
/// [dst]: https://doc.rust-lang.org/nomicon/exotic-sizes.html#dynamically-sized-types-dsts
/// [`boxed`]: Strs::boxed
/// [`arced`]: Strs::arced
/// [`rced`]: Strs::rced
///
///
/// ## Examples
///
///
///
/// ## Benchmarks
///
/// // TODO
///
/// ## When not to use / Alternatives
///
/// - If you need to mutate strings [`Strs`] is not your shoice either, you probably need something
///   like `Vec<String>`
/// - If your strings known at compile time you can simply use `&'static [&'static str]`
/// - If borrowed variant is ok for you, you can use `Vec<&str>` and/or `&[&str]`
/// - If you know what strings you'll have at compile-time then you'd better use simply
///   [`Box`][box]/[`Arc`][arc]/[`Rc`][rc]`<str>` + indices. E.g.:
///   ```
///   use std::sync::Arc;
///   pub struct Config {
///       strs: Arc<str>,
///       name: usize,
///       thing: usize
///   }
///
///   impl Config {
///       pub fn aaa(&self) -> &str {
///           &self.strs[..self.name]
///       }
///
///       pub fn name(&self) -> &str {
///           &self.strs[self.name..self.thing]
///       }
///
///       pub fn thing(&self) -> &str {
///           &self.strs[self.thing..]
///       }
///   }
///   ```
///
/// [box]: Box
/// [arc]: std::sync::Arc
/// [rc]: std::rc::Rc
#[repr(C)]
pub struct Strs {
    /// Number of strings contained
    len: usize,
    /// ## Invariants
    ///
    /// 1. `buf[..(len + 1) * size_of::<usize>()]` is valid `[usize]` i.e
    ///   ```ignore
    ///   buf[..(len + 1) * core::mem::size_of::<usize>()].align_to::<usize>()
    ///   ```
    ///   Returns `([], xs, [])` where `xs` is a `usize` slice
    ///   - `xs` must be sorted such that `xs[n] <= xs[n + 1]`
    /// 2. `buf[(len + 1) * size_of::<usize>()..]` is valid utf-8 string
    /// 3. `buf[x..y]` such that `x, y ∈ indices` and `x <= y` is also valid utf-8 string
    buf: [u8],
}

#[derive(Debug)]
pub enum FromSliceError {
    IndicesOrder,
    NonUtf8 { offset: usize, inner: Utf8Error },
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
        #[repr(C)]
        struct Coerce<T: ?Sized> {
            len: usize,
            buf: T,
        }

        let coerce: &Coerce<[u8]> = &Coerce {
            len: 0,
            buf: 0usize.to_ne_bytes(),
        };

        // TODO: safety
        #[allow(clippy::transmute_ptr_to_ptr)] // you can't dereference ptr in consts :|
        unsafe {
            transmute::<&Coerce<[u8]>, &Strs>(coerce)
        }
    };

    /// Creates `Arc<Strs>` from `&[S: AsRef<str>]`.
    ///
    /// ## Examples
    ///
    /// ```
    /// use std::sync::Arc;
    /// use strs::Strs;
    ///
    /// let arced: Arc<Strs> = Strs::arced(&["Hello,", " ", "world", "!"]);
    /// let clone = Arc::clone(&arced);
    ///
    /// assert_eq!(arced.as_str(), "Hello, world!");
    /// assert_eq!(clone.as_str(), "Hello, world!");
    ///
    /// assert_eq!(&arced[0], "Hello,");
    /// assert_eq!(&clone[2], "world");
    /// ```
    pub fn arced<S: AsRef<str>>(slice: &[S]) -> Arc<Self> {
        let req = Strs::required_words_for(slice);

        // Allocate required memory
        //
        // This is stable analogous to the `Arc::new_uninit_slice(len)` method. Tests[^0] show that
        // this is as performant, as the nightly methods
        //
        // [^0]: https://godbolt.org/z/43b8Kz
        let mut arc: Arc<[MaybeUninit<usize>]> =
            iter::repeat(MaybeUninit::uninit()).take(req).collect();

        let target = Arc::get_mut(&mut arc).expect("just created, not cloned");

        // Initialize `Strs` in place
        let strs = Strs::init_from_slice(slice, target) as *mut Strs;

        Arc::into_raw(arc);

        // TODO: fix safety notes for this and next
        // ## Safety
        //
        // `init_from_slice` guarantees that it has written `Strs` to the target.
        //
        // `Strs` has the same layout as `[usize]` (e.i.: it won't be UB to dealloc memory with
        // `Strs` layout if it was allocated with `[usize]` layout)
        unsafe { Arc::from_raw(strs) }
    }

    /// Creates `Rc<Strs>` from `&[S: AsRef<str>]`.
    ///
    /// ## Examples
    ///
    /// ```
    /// use std::rc::Rc;
    /// use strs::Strs;
    ///
    /// let rced: Rc<Strs> = Strs::rced(&["Hello,", " ", "world", "!"]);
    /// let clone = Rc::clone(&rced);
    ///
    /// assert_eq!(rced.as_str(), "Hello, world!");
    /// assert_eq!(clone.as_str(), "Hello, world!");
    ///
    /// assert_eq!(&rced[0], "Hello,");
    /// assert_eq!(&clone[2], "world");
    /// ```
    ///
    /// ## See also
    ///
    ///
    pub fn rced<S: AsRef<str>>(slice: &[S]) -> Rc<Self> {
        let req = Strs::required_words_for(slice);

        // Allocate required memory
        //
        // This is stable analogous to the `Arc::new_uninit_slice(len)` method. Tests[^0] show that
        // this is as performant, as the nightly methods
        //
        // [^0]: https://godbolt.org/z/43b8Kz
        let mut rc: Rc<[MaybeUninit<usize>]> =
            iter::repeat(MaybeUninit::uninit()).take(req).collect();

        let target = Rc::get_mut(&mut rc).expect("just created, not cloned");

        // Initialize `Strs` in place
        let strs = Strs::init_from_slice(slice, target) as *mut Strs;

        let _ = Rc::into_raw(rc);
        // ## Safety
        //
        // `init_from_slice` guarantees that it has written `Strs` to the target.
        //
        // `Strs` has the same layout as `[usize]` (e.i.: it won't be UB to dealloc memory with
        // `Strs` layout if it was allocated with `[usize]` layout)
        unsafe { Rc::from_raw(strs) }
    }

    /// Creates `Box<Strs>` from `&[S: AsRef<str>]`.
    ///
    /// ## Examples
    ///
    /// ```
    /// use std::boxed::Box;
    /// use strs::Strs;
    ///
    /// let boxed: Box<Strs> = Strs::boxed(&["Hello,", " ", "world", "!"]);
    ///
    /// assert_eq!(boxed.as_str(), "Hello, world!");
    /// assert_eq!(&boxed[0], "Hello,");
    /// ```
    ///
    /// ## See also
    ///
    ///
    pub fn boxed<S: AsRef<str>>(slice: &[S]) -> Box<Self> {
        let req = Strs::required_words_for(slice);

        // Allocate required memory
        //
        // This is stable analogous to the `Box::new_uninit_slice(len)` method. Tests[^0] show that
        // this is as performant, as the nightly methods
        //
        // [^0]: https://godbolt.org/z/43b8Kz
        let mut boxed: Box<[MaybeUninit<usize>]> =
            iter::repeat(MaybeUninit::uninit()).take(req).collect();

        // Initialize `Strs` in place
        let strs = Strs::init_from_slice(slice, boxed.deref_mut()) as *mut Strs;

        // Forget
        Box::into_raw(boxed);

        // ## Safety
        //
        // `init_from_slice` guarantees that returned reference is the same as input
        // (i.e.: it's the same allocation, etc).
        //
        // `Strs` has the same layout as `[usize]` (e.i.: it won't be UB to dealloc memory with
        // `Strs` layout if it was allocated with `[usize]` layout)
        unsafe { Box::from_raw(strs) }
    }

    ///
    ///
    /// ## Example
    ///
    /// ```ignore
    /// // TODO: understand how to properly create stack allocated Strs with safe code
    /// use strs::Strs;
    ///
    /// // Size can vary on different architectures
    /// const SIZE: usize = core::mem::size_of::<usize>();
    /// let mut buf = [0; SIZE * 4 + 11];
    ///
    /// // write len
    /// buf[..SIZE].copy_from_slice(&2usize.to_ne_bytes());
    ///
    /// // write indices
    /// buf[SIZE..SIZE * 2].copy_from_slice(&(SIZE * 3).to_ne_bytes());
    /// buf[SIZE * 2..SIZE * 3].copy_from_slice(&(SIZE * 3 + 6).to_ne_bytes());
    /// buf[SIZE * 3..SIZE * 4].copy_from_slice(&(SIZE * 3 + 11).to_ne_bytes());
    ///
    /// // Write the strings themselfs          _____*--- second part
    /// buf[SIZE * 4..].copy_from_slice(b"Hello world");
    /// //                                ^^^^^^*--- first part
    ///
    /// let strs = Strs::from_slice(&buf[..]).unwrap();
    /// assert_eq!(&strs[0], "Hello ");
    /// assert_eq!(&strs[1], "world");
    /// ```
    pub fn from_slice(slice: &[usize]) -> Result<&Self, FromSliceError> {
        let len = slice[0];
        let (_, buf, _) = unsafe { slice[1..].align_to::<u8>() };
        let indices = &slice[1..len + 2];

        // Result<&str, Utf8Error> -> Result<(), CoerceError>
        let utferr = |res: Result<&str, Utf8Error>| {
            res.map(drop).map_err(|err| FromSliceError::NonUtf8 {
                offset: len * core::mem::size_of::<usize>(),
                inner: err,
            })
        };

        // Check invariant: end of the buf is valid utf8 string
        utferr(core::str::from_utf8(
            &buf[(len + 1) * core::mem::size_of::<usize>()..],
        ))?;

        // Check invariants:
        // - indices are sorted
        // - all parts are valid utf8 strings
        indices.windows(2).try_for_each(|s| match s {
            [a, b] if a > b => Err(FromSliceError::IndicesOrder),
            &[a, b] => utferr(core::str::from_utf8(&buf[a..b])),
            _ => Ok(()), // Unreachable
        })?;

        // ## Safety
        //
        // We've just checked all invariants
        unsafe { Ok(Self::from_slice_unchecked(slice)) }
    }

    /// Returns an iterator over the strings in `Strs`.
    ///
    /// ## Examples
    ///
    /// ```
    /// use strs::Strs;
    ///
    /// let strs = Strs::boxed(&["aaaa", "hahaha", "AAAAAAA"]);
    /// let mut iter = strs.iter();
    ///
    /// assert_eq!(iter.next(), Some("aaaa"));
    /// assert_eq!(iter.next_back(), Some("AAAAAAA"));
    /// assert_eq!(iter.next(), Some("hahaha"));
    /// assert_eq!(iter.next(), None);
    /// ```
    pub fn iter(&self) -> Iter {
        Iter {
            strs: self,
            forth_idx: 0,
            back_idx: self.len(),
        }
    }

    /// View entire underling buffer as `&str`
    ///
    /// ## Examples
    ///
    /// ```
    /// use strs::Strs;
    ///
    /// let s = Box::<Strs>::from(vec!["Hello,", " ", "world", "!"]);
    ///
    /// assert_eq!(s.as_str(), "Hello, world!");
    /// ```
    pub fn as_str(&self) -> &str {
        unsafe {
            // At the start of the buf there are `len+1` usizes, so the strings start from
            let lower = mem::size_of::<usize>() * self.len + mem::size_of::<usize>();

            let bytes = &self.buf[lower..];
            core::str::from_utf8_unchecked(bytes)
        }
    }

    /// Returns length (number of strings) of the collection.
    ///
    /// ## Examples
    ///
    /// ```
    /// use strs::Strs;
    ///
    /// let s = Box::<Strs>::from(vec!["", "abs", "x"]);
    ///
    /// assert_eq!(s.len(), 3);
    /// assert_eq!(Strs::EMPTY.len(), 0);
    /// ```
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the strs contains no strings.
    ///
    /// ## Examples
    ///
    /// ```
    /// use strs::Strs;
    ///
    /// let s = Box::<Strs>::from(vec![""]);
    ///
    /// assert_eq!(s.is_empty(), false);
    /// assert_eq!(Strs::EMPTY.is_empty(), true);
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns a reference to an `idx`th element string.
    ///
    /// If the index is out of bounds returns [`None`].
    ///
    /// ## Examples
    ///
    /// ```
    /// use strs::Strs;
    ///
    /// let s = Box::<Strs>::from(vec!["abs", "x"]);
    ///
    /// assert_eq!(s.get(0), Some("abs"));
    /// assert_eq!(s.get(1), Some("x"));
    /// assert_eq!(s.get(2), None);
    /// ```
    pub fn get(&self, idx: usize) -> Option<&str> {
        if self.len() > idx {
            // ## Safety
            //
            // We've just ensured that index is in bounds
            Some(unsafe { self.get_unchecked(idx) })
        } else {
            None
        }
    }

    /// Returns a reference to an element string, without doing bounds checking.
    ///
    /// This is generally not recommended, use with caution! Calling this method with an
    /// out-of-bounds index is [undefined behavior][ub] even if the resulting reference
    /// is not used. For a safe alternative see [`get`] or [`index`].
    ///
    /// [ub]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    /// [`get`]: Strs::get
    /// [`index`]: Strs#impl-Index<usize>
    ///
    /// ## Safety
    ///
    /// caller must ensure that `idx` is in-bounds that is - `.len() > idx`
    ///
    /// ## Examples
    ///
    /// ```
    /// use strs::Strs;
    ///
    /// let s = Box::<Strs>::from(vec!["abs", "x"]);
    ///
    /// unsafe { assert_eq!(s.get_unchecked(1), "x"); }
    /// ```
    ///
    /// Example of **wrong** usage that causes [ub]:
    ///
    /// ```no_run
    /// use strs::Strs;
    ///
    /// let s = Box::<Strs>::from(vec!["abs", "x"]);
    ///
    /// // The output is not used, but it's UB anyway
    /// unsafe { s.get_unchecked(2); }
    /// ```
    pub unsafe fn get_unchecked(&self, idx: usize) -> &str {
        // ## Safety
        //
        // caller guarantees that idx is inbounds
        let (lower, upper) = self.get_bounds(idx);

        // ## Safety
        //
        // Struct invariants guarantee that for any two indices `&self.buf[lower..upper]`
        // contains valid utf-8 string and that lower/upper indices are in bounds
        #[allow(unused_unsafe)]
        unsafe {
            let bytes = &self.buf.get_unchecked(lower..upper);
            core::str::from_utf8_unchecked(bytes)
        }
    }

    /// ## Safety
    ///
    /// No
    pub unsafe fn from_slice_unchecked(slice: &[usize]) -> &Self {
        let len = slice[0];
        let size = slice[len + 1] - slice[1]; // TODO: check if this works for empty

        Self::from_raw_parts(slice.as_ptr(), size)
    }

    /// ## Safety
    ///
    /// No
    pub unsafe fn from_slice_unchecked_mut(slice: &mut [usize]) -> &mut Self {
        let len = slice[0];
        let size = slice[len + 1] - slice[1]; // TODO: check if this works for empty

        Self::from_raw_parts_mut(slice.as_mut_ptr(), size)
    }

    /// Return space required for creating `Strs` from the given slice in **words**.
    ///
    /// That's it - to create `Strs` from `slice` you need a `&[usize]`-slice with
    /// `.len() == Strs::required_words_for(slice)`
    pub fn required_words_for<T: AsRef<str>>(slice: &[T]) -> usize {
        Self::required_words_for_and_size(slice).0
    }

    fn required_words_for_and_size<T: AsRef<str>>(slice: &[T]) -> (usize, usize) {
        let size: usize = slice.iter().map(|s| s.as_ref().len()).sum();
        let len = slice.len();

        // `len` field + indices + payload
        let req = 1 + (len + 1) + ceiling_div(size, mem::size_of::<usize>());
        (req, size)
    }

    /// Writes content of `vec` into target in a pretty compact way.
    ///
    /// This function is low-level,so if you just want to create [`Strs`] consider using [`arced`],
    /// [`boxed`] or [`rced`].
    ///
    /// [`arced`]: Self::arced
    /// [`boxed`]: Self::boxed
    /// [`rced`]: Self::rced
    ///
    /// ## Guarantees
    ///
    /// This function **guarantee** that
    /// - `data` part (not `size` though) of the output reference points to the same location as
    ///   `target`, this means that e.g. if `target` was received from `Box` you can recreate
    ///   `Box<Strs>` from output of this function
    /// - After calling it `target` will be fully initialized e.i.: calling
    ///   `MaybeUninit::slice_get_mut(target)` **won't** be UB
    ///
    /// ## Panics
    ///
    ///
    #[track_caller]
    pub fn init_from_slice<'t, S: AsRef<str>>(
        slice: &[S],
        target: &'t mut [MaybeUninit<usize>],
    ) -> &'t mut Self {
        let (required_words, size) = Self::required_words_for_and_size(slice);
        let len = slice.len();
        let indices = len + 1;

        // Check that size of target is exactly equal to required
        assert_eq!(
            required_words,
            target.len(),
            "`target` is bigger or smaller that required for this operation"
        );

        // write `len` field
        target[0] = MaybeUninit::new(len);

        let buf_ptr = unsafe { target.as_mut_ptr().offset(1).cast::<u8>() };
        let target_buf_bytes = (target.len() - 1) * mem::size_of::<usize>();

        // offset from `buf_ptr` to the empty place
        let mut offset = indices * mem::size_of::<usize>();
        for (idx, s) in slice.iter().enumerate() {
            let str = s.as_ref();

            // Double check that we didn't run out of space because `S::as_ref` may be
            // malicious and return different strings upon calls (I wish it was pure...)
            if offset + str.len() > target_buf_bytes {
                malicious_as_ref(0)
            }

            unsafe {
                buf_ptr.cast::<usize>().add(idx).write(offset);
            }

            // ## Safety
            unsafe {
                ptr::copy_nonoverlapping(str.as_bytes().as_ptr(), buf_ptr.add(offset), str.len())
            }

            offset += str.len();
        }

        unsafe {
            buf_ptr.cast::<usize>().add(indices - 1).write(offset);
        }

        unsafe {
            let tail = target_buf_bytes - offset;

            // Double check that we've spent all space except last (usize-1)
            // (yet again malicious `S::as_ref`)
            if tail >= mem::size_of::<usize>() {
                malicious_as_ref(1)
            }

            // Fill/initialize the tail to guarantee that `target` is fully initialized
            ptr::write_bytes(buf_ptr.add(offset), 0, tail);
        }

        unsafe { Self::from_raw_parts_mut(target.as_mut_ptr().cast::<usize>(), size) }
    }
}

impl std::ops::Index<usize> for Strs {
    type Output = str;

    #[track_caller]
    fn index(&self, idx: usize) -> &str {
        match self.get(idx) {
            Some(ret) => ret,
            None => panic!(
                "index out of bounds: the len is {} but the index is {}",
                self.len(),
                idx
            ),
        }
    }
}

impl<S: AsRef<str>> From<Vec<S>> for Box<Strs> {
    fn from(vec: Vec<S>) -> Self {
        vec.as_slice().into()
    }
}

impl<S: AsRef<str>> From<&[S]> for Box<Strs> {
    fn from(slice: &[S]) -> Self {
        Strs::boxed(slice)
    }
}

impl fmt::Debug for Strs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        struct Helper<'a>(&'a Strs);

        impl fmt::Debug for Helper<'_> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.debug_list().entries(self.0.iter()).finish()
            }
        }

        f.debug_struct("Strs")
            .field("len", &self.len)
            .field("strs", &Helper(self))
            .finish()
    }
}

/// Note: this borrows the whole underling string, same as [`Strs::as_str`]
impl AsRef<str> for Strs {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl PartialEq for Strs {
    fn eq(&self, other: &Self) -> bool {
        // We use `as_raw` instead of `as_str` to compare indices too
        //
        // Also, we store the `len` at the beginning, so if lens are not equal this will fast-exit
        self.as_raw() == other.as_raw()
    }
}

impl Eq for Strs {}

impl PartialOrd for Strs {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Strs {
    fn cmp(&self, other: &Self) -> Ordering {
        self.iter().cmp(other.iter())
    }
}

impl Hash for Strs {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Note: `as_raw` is used instead of `as_str` to hash indices and len too
        self.as_raw().hash(state)
    }
}

impl Clone for Box<Strs> {
    fn clone(&self) -> Self {
        let raw = self.as_raw();

        // Allocate memory for copy
        let mut boxed: Box<[usize]> = iter::repeat(0).take(raw.len()).collect();

        // make bitwise copy
        boxed.copy_from_slice(raw);

        let ptr = unsafe { Strs::from_slice_unchecked_mut(&mut *boxed) as *mut Strs };
        let _ = Box::into_raw(boxed);
        unsafe { Box::from_raw(ptr) }
    }
}

impl fmt::Display for FromSliceError {
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
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::IndicesOrder => None,
            FromSliceError::NonUtf8 { offset: _, inner } => Some(inner),
        }
    }
}

impl Strs {
    /// Creates `Strs` from raw parts.
    ///
    /// `ptr` - pointer to the start of the struct, `len` - number of strings contained,
    /// `size` - total size of all strings combined.
    ///
    /// ## Safety
    ///
    /// - Caller need to ensure invariants of [`slice::from_raw_parts`][slice_parts] on
    ///   len: `size + (len + 1) * mem::size_of::<usize>()`
    /// - It's [ub] to not ensure invariants described in [`from_slice`].
    ///
    /// [slice_parts]: core::slice::from_raw_parts
    /// [ub]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    unsafe fn from_raw_parts<'a>(ptr: *const usize, size: usize) -> &'a Self {
        let len = *ptr;

        let length = size + (len + 1) * mem::size_of::<usize>();
        let ptr = slice::from_raw_parts(ptr.cast::<()>(), length) as *const [()] as *const Strs;

        &*ptr
    }

    /// Performs the same functionality as [`from_raw_parts`], except that a mutable `Strs` is
    /// returned.
    ///
    /// ## Safety
    ///
    /// - Caller need to ensure invariants of [`slice::from_raw_parts_mut`][slice_parts] on
    ///   len: `size + (len + 1) * mem::size_of::<usize>()`
    /// - It's [ub] to not ensure invariants described in [`from_slice`].
    ///
    /// [slice_parts]: core::slice::from_raw_parts_mut
    /// [ub]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    unsafe fn from_raw_parts_mut<'a>(ptr: *mut usize, size: usize) -> &'a mut Self {
        let len = *ptr;

        let length = size + (len + 1) * mem::size_of::<usize>();
        let ptr = slice::from_raw_parts_mut(ptr.cast::<()>(), length) as *mut [()] as *mut Strs;

        &mut *ptr
    }

    /// Get bounds of the `idx`th element.
    ///
    /// E.g. `strs[1]` would be equal to
    /// ```ignore
    /// let (lower, upper) = strs.get_bounds(1);
    /// strs.as_str()[lower..upper]
    /// ```
    ///
    /// ## Safety
    ///
    /// The caller must ensure that `len < idx`
    unsafe fn get_bounds(&self, idx: usize) -> (usize, usize) {
        let ptr = self
            .buf
            .as_ptr()
            .cast::<usize>()
            // Safety guaranteed by the caller
            .add(idx);

        // ## Safety
        //
        // Invariants of the `Strs` guarantee that `buf` is filled from start with `len+1`
        // `usize`-indices, the caller guarantees that `idx < len`.
        (*ptr, *ptr.add(1))
    }

    fn as_raw(&self) -> &[usize] {
        let length = 1 + ceiling_div(self.buf.len(), mem::size_of::<usize>());
        unsafe { slice::from_raw_parts(self as *const Strs as *const usize, length) }
    }
}

/// Iterator over strings in [`Strs`]
///
/// See [`Strs::iter`]
#[derive(Debug)]
pub struct Iter<'a> {
    strs: &'a Strs,
    // TODO: probably it would be better to implement this with pointers to idx buffer
    forth_idx: usize,
    back_idx: usize,
}

impl<'a> Iterator for Iter<'a> {
    type Item = &'a str;

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
            self.forth_idx += 1;
            res
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }
}

impl<'a> DoubleEndedIterator for Iter<'a> {
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
            self.back_idx -= 1;
            Some(self.strs.get_unchecked(self.back_idx))
        }
    }
}

impl ExactSizeIterator for Iter<'_> {
    fn len(&self) -> usize {
        self.back_idx - self.forth_idx
    }

    #[cfg(feature = "nightly")]
    fn is_empty(&self) -> bool {
        self.back_idx == self.forth_idx
    }
}

impl<'a> IntoIterator for &'a Strs {
    type Item = &'a str;
    type IntoIter = Iter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

pub(crate) fn ceiling_div(n: usize, d: usize) -> usize {
    (n / d) + ((n % d != 0) as usize)
}

#[track_caller]
#[cfg_attr(test, allow(unreachable_code))]
pub(crate) fn malicious_as_ref(_n: u8) -> ! {
    // Panic in tests to test that we actually detect malicious `as_ref`s
    #[cfg(test)]
    panic!("malicious S::as_ref ({})", _n);

    // aborting because who the fuck are writing malicious `as_ref`s???
    abort()
}

#[cfg(test)]
mod tests {
    use crate::Strs;
    use core::mem::{self, MaybeUninit};
    use std::cell::Cell;
    use std::convert::AsRef;

    #[test]
    fn from_vec() {
        let strings = ["a", "36dx1LOPnHgobIhUF3Ik", "", "", "fffff", "NWUzxiexni48"];

        fn assertions(s: &Strs) {
            assert_eq!(s.len(), 6);
            assert_eq!(&s[0], "a");
            assert_eq!(&s[1], "36dx1LOPnHgobIhUF3Ik");
            assert_eq!(&s[2], "");
            assert_eq!(&s[3], "");
            assert_eq!(&s[4], "fffff");
            assert_eq!(&s[5], "NWUzxiexni48");
            assert_eq!(s.get(6), None);
            assert_eq!(s.as_str(), "a36dx1LOPnHgobIhUF3IkfffffNWUzxiexni48");
        }

        assertions(Strs::boxed(&strings).as_ref());
        assertions(Strs::arced(&strings).as_ref());
        assertions(Strs::rced(&strings).as_ref());
    }

    #[test]
    fn tail_is_initialized() {
        let mut uninit = box_new_uninit_slice(Strs::required_words_for(&["x"]));

        let _ = Strs::init_from_slice(&["x"], &mut *uninit);

        let mut tail = [0; mem::size_of::<usize>()];
        tail[0] = b'x';
        let buf = [
            1,
            mem::size_of::<usize>() * 2,
            mem::size_of::<usize>() * 2 + 1,
            usize::from_ne_bytes(tail),
        ];

        unsafe {
            assert_eq!(&*box_assume_init(uninit), &buf);
        }
    }

    #[test]
    fn from_empty_str() {
        let _ = Strs::boxed(&[""]);
    }

    #[test]
    fn eq_neq() {
        assert_eq!(Strs::boxed(&["axd", "FF"]), Strs::boxed(&["axd", "FF"]));
        assert_ne!(Strs::boxed(&["X", ""]), Strs::boxed(&["", "X"]));
    }

    #[test]
    fn iter() {
        assert_eq!(Strs::EMPTY.iter().collect::<Vec<_>>(), Vec::<&str>::new());
        assert_eq!(Strs::EMPTY.iter().len(), 0);

        assert_eq!(
            Strs::boxed(&["a", "b'", "ccc"]).iter().collect::<Vec<_>>(),
            vec!["a", "b'", "ccc"]
        );
        assert_eq!(
            Strs::boxed(&["x", "y", "z"])
                .iter()
                .rev()
                .collect::<Vec<_>>(),
            vec!["z", "y", "x"]
        );

        let strs = Strs::boxed(&["x", "y", "z", "a", "b"]);
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

    #[test]
    fn debug() {
        assert_eq!(format!("{:?}", Strs::EMPTY), "Strs { len: 0, strs: [] }");
        assert_eq!(
            format!("{:?}", Strs::boxed(&["42", "xir"])),
            "Strs { len: 2, strs: [\"42\", \"xir\"] }"
        );
    }

    #[test]
    fn clone() {
        let boxed = Strs::boxed(&["a"]);
        assert_eq!(boxed.clone(), boxed);

        let boxed = Strs::boxed::<&str>(&[]);
        assert_eq!(boxed.clone(), boxed);
    }

    #[test]
    #[should_panic(expected = "assertion failed: `(left == right)`
  left: `4`,
 right: `3`: `target` is bigger or smaller that required for this operation")]
    fn not_enough_space() {
        let mut uninit = box_new_uninit_slice(Strs::required_words_for(&["x"]) - 1);
        let _ = Strs::init_from_slice(&["x"], &mut *uninit);
    }

    #[test]
    #[should_panic(expected = "assertion failed: `(left == right)`
  left: `4`,
 right: `5`: `target` is bigger or smaller that required for this operation")]
    fn too_much_space() {
        let mut uninit = box_new_uninit_slice(Strs::required_words_for(&["x"]) + 1);
        let _ = Strs::init_from_slice(&["x"], &mut *uninit);
    }

    #[test]
    #[should_panic(expected = "malicious S::as_ref (0)")]
    fn malicious_as_ref_greater() {
        let badarr = [MaliciousAsRef {
            n: Cell::new(0),
            // lengths should differ such that
            // ceiling_div(l.len(), size_of::<usize>()) != ceiling_div(g.len(), size_of::<usize>())
            l: "a",
            g: "bbbbbbbbb",
        }];
        let _ = Strs::boxed(&badarr);
    }

    #[test]
    #[should_panic(expected = "malicious S::as_ref (1)")]
    fn malicious_as_ref_less() {
        let badarr = [MaliciousAsRef {
            n: Cell::new(0),
            // lengths should differ such that
            // ceiling_div(l.len(), size_of::<usize>()) != ceiling_div(g.len(), size_of::<usize>())
            l: "aaaaaaaaa",
            g: "b",
        }];
        let _ = Strs::boxed(&badarr);
    }

    /// A test util structure that returns `self.l` upon first 2 calls to `.as_ref()`
    /// and self.g from 3-rd.
    struct MaliciousAsRef {
        n: Cell<u8>,
        l: &'static str,
        g: &'static str,
    }

    impl AsRef<str> for MaliciousAsRef {
        fn as_ref(&self) -> &str {
            let val = self.n.get();
            // `required_words_for_and_size` is called in both `Strs::boxed`
            // and Strs::init_from_slice
            if val < 2 {
                self.n.set(val + 1);
                self.l
            } else {
                self.g
            }
        }
    }

    // stable analog for Box::new_uninit_slice
    fn box_new_uninit_slice(req: usize) -> Box<[MaybeUninit<usize>]> {
        core::iter::repeat(MaybeUninit::uninit())
            .take(req)
            .collect()
    }

    unsafe fn box_assume_init(this: Box<[MaybeUninit<usize>]>) -> Box<[usize]> {
        Box::from_raw(Box::into_raw(this) as *mut [usize])
    }
}
