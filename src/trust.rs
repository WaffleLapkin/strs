use std::{borrow::Cow, rc::Rc, sync::Arc};

use crate::{int::Source, trusted_idx::TrustedIdx};

/// A type which `<_ as AsRef<str>>::as_ref` method trusted to be is pure
///
/// ## Safety
///
/// `<_ as AsRef<str>>::as_ref` method **must** be _pure_. I.e.: it **must**
/// return the same strings between calls on the same value. Also it **must
/// not** cause any side effects.
pub unsafe trait TrustedAsRef: AsRef<str> {}

/// A [`Source`] which generic type `S`'s `AsRef<str>` impl is
/// [trusted to be pure](TrustedAsRef#Safety)
pub struct TrustedSource<'a, S, Idx: TrustedIdx>(Source<'a, S, Idx>);

impl<'a, S, Idx: TrustedIdx> TrustedSource<'a, S, Idx> {
    /// (unsafely) [trust] type `S` and create a [`TrustedSource`] from a [`Source`]
    ///
    /// [trust]: TrustedAsRef#Safety
    ///
    /// ## Safety
    ///
    /// See [`trust`]
    #[inline]
    pub unsafe fn new_unchecked(inner: Source<'a, S, Idx>) -> Self
    where
        S: AsRef<str>,
    {
        Self(inner)
    }

    /// Get a ref to the inner [`Source`] value
    ///
    /// ## Guarantees (which `unsafe` code can use)
    ///
    /// `S` satisfies guarantees mentioned in
    /// [`TrustedAsRef`](TrustedAsRef#Safety).
    #[inline]
    pub fn inner(&self) -> &Source<'a, S, Idx> {
        &self.0
    }

    /// Get the inner [`Source`] value
    ///
    /// ## Guarantees (which `unsafe` code can use)
    ///
    /// `S` satisfies guarantees mentioned in
    /// [`TrustedAsRef`](TrustedAsRef#Safety).
    #[inline]
    pub fn into_inner(self) -> Source<'a, S, Idx> {
        self.0
    }
}

impl<'a, S: TrustedAsRef, Idx: TrustedIdx> From<Source<'a, S, Idx>> for TrustedSource<'a, S, Idx> {
    #[inline]
    fn from(inner: Source<'a, S, Idx>) -> Self {
        // ## Safety
        //
        // `S` implements `TrustedAsRef` and thus must uphold it's guarantees.
        unsafe { Self::new_unchecked(inner) }
    }
}

/// (unsafely) [trust] type `S` and create a [`TrustedSource`] from a [`Source`]
///
/// Note: if `S` implements [`TrustedAsRef`] then you should use
/// [`TrustedSource::from`].
///
/// This method should only be used if `S` is not owned
/// by you (i.e.: declared in a different crate) **and** it doesn't implement
/// [`TrustedAsRef`] **and** you are totally sure it's `AsRef` impl is pure.
///
/// [trust]: TrustedAsRef#Safety
///
/// ## Safety
///
/// `S` must satisfy guarantees mentioned in [`TrustedAsRef`](TrustedAsRef#Safety).
#[inline]
pub unsafe fn trust<S: AsRef<str>, Idx: TrustedIdx>(
    slice: Source<S, Idx>,
) -> TrustedSource<S, Idx> {
    // ## Safety
    //
    // Guaranteed by the caller
    TrustedSource::new_unchecked(slice)
}

/// If this function return `true`, then `T`'s implementation of `AsRef<str>`
/// is trusted — `T` implements [`TrustedAsRef`].
///
/// **Note**: this function does it's job on the "best effort" basis —
/// it may return `false` even if `T` implements [`TrustedAsRef`].
///
/// More precisely, it always return `false` if `nightly` feature is **not**
/// enabled.
// Returns constant
#[allow(clippy::inline_always)]
#[inline(always)]
pub(crate) fn is_trusted_as_ref<T: AsRef<str>>() -> bool {
    #[cfg(feature = "nightly")]
    // Returns constant
    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn impl_<T: AsRef<str>>() -> bool {
        trait Is {
            const TRUSTED: bool;
        }

        impl<T> Is for T {
            default const TRUSTED: bool = false;
        }

        impl<T: TrustedAsRef> Is for T {
            const TRUSTED: bool = true;
        }

        <T as Is>::TRUSTED
    }

    #[cfg(not(feature = "nightly"))]
    // Returns constant
    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn impl_<T: AsRef<str>>() -> bool {
        false
    }

    impl_::<T>()
}

// ## Safety
//
// All std types implement `AsRef` in a reasonable way
unsafe impl TrustedAsRef for String {}
unsafe impl TrustedAsRef for str {}
unsafe impl TrustedAsRef for Cow<'_, str> {}
unsafe impl TrustedAsRef for Rc<str> {}
unsafe impl TrustedAsRef for Arc<str> {}
unsafe impl TrustedAsRef for Box<str> {}
unsafe impl<T: ?Sized + TrustedAsRef> TrustedAsRef for &T {}
unsafe impl<T: ?Sized + TrustedAsRef> TrustedAsRef for &mut T {}
