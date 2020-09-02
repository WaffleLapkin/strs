use std::{
    convert::TryInto,
    mem::transmute,
    ops::{Add, Sub},
};

use crate::Strs;

/// Represents a type that can be used in [`Strs`](crate::Strs) as `Idx`.
///
/// ## Safety
///
/// Ugh, just don't implement this trait
pub unsafe trait TrustedIdx:
    Copy + Eq + Ord + Add<Output = Self> + Sub<Output = Self> + private::Sealed + 'static
{
    /// Zero (`0`)
    const ZERO: Self;

    /// One (`1`)
    const ONE: Self;

    /// Helper for implementing `Strs::EMPTY`
    /// (it's required because we can't use generics in consts, see
    /// [rust-lang/rfcs/pull/2944#issuecomment-685087090][comment])
    ///
    /// [comment]: https://github.com/rust-lang/rfcs/pull/2944#issuecomment-685087090
    const EMPTY_STRS: &'static Strs<Self>;

    /// Lossless conversion to usize
    fn as_usize(self) -> usize;

    /// Try to convert `usize` to `Self`
    ///
    /// If `us` is bigger than maximum representable in `Self` integer, then this method must panic
    fn from_usize(us: usize) -> Self;

    /// Increment `self` by one (`1`)
    ///
    /// I.e. `self += 1`
    #[inline]
    fn inc(&mut self) {
        *self = *self + Self::ONE
    }

    /// Decrement `self` by one (`1`)
    ///
    /// I.e. `self -= 1`
    #[inline]
    fn dec(&mut self) {
        *self = *self - Self::ONE
    }
}

/// Helper struct to create `Strs` in const ctx
#[repr(C)]
struct Coerce<Idx: TrustedIdx, T: ?Sized> {
    len: Idx,
    buf: T,
}

unsafe impl TrustedIdx for usize {
    const ZERO: Self = 0;
    const ONE: Self = 1;
    const EMPTY_STRS: &'static Strs<Self> = {
        let coerce: &Coerce<Self, [u8]> = &Coerce {
            len: 0,
            buf: Self::ZERO.to_ne_bytes(),
        };

        // TODO: safety
        #[allow(clippy::transmute_ptr_to_ptr)] // you can't dereference ptr in consts :|
        unsafe {
            transmute::<&Coerce<Self, [u8]>, &Strs<Self>>(coerce)
        }
    };

    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn as_usize(self) -> usize {
        self
    }

    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn from_usize(us: usize) -> Self {
        us
    }
}

macro_rules! impls {
    (unsafe { $( $( #[$($meta:tt)+] )? $ty:ty )* }) => {
        $(
            $( #[$($meta)+] )?
            unsafe impl TrustedIdx for $ty {
                const ZERO: Self = 0;
                const ONE: Self = 1;
                const EMPTY_STRS: &'static Strs<Self> = {
                    let coerce: &Coerce<Self, [u8]> = &Coerce {
                        len: 0,
                        buf: Self::ZERO.to_ne_bytes(),
                    };

                    // TODO: safety
                    #[allow(clippy::transmute_ptr_to_ptr)] // you can't dereference ptr in consts :|
                    unsafe {
                        transmute::<&Coerce<Self, [u8]>, &Strs<Self>>(coerce)
                    }
                };

                #[inline]
                fn as_usize(self) -> usize {
                    self as _
                }

                #[inline]
                fn from_usize(us: usize) -> Self {
                    us.try_into().expect("TODO") // TODO
                }
            }
        )*
    };
}

impls!(unsafe {
    u8
    u16
    // Idx should not be bigger that usize
    #[cfg(not(target_pointer_width = "16"))]
    u32
});

mod private {
    pub trait Sealed {}

    impl Sealed for u8 {}
    impl Sealed for u16 {}
    impl Sealed for u32 {}
    impl Sealed for usize {}
}
