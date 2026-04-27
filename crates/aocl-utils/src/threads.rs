//! Thread-pinning helpers from AOCL-Utils.
//!
//! These are very thin wrappers around the AOCL routines that take a
//! list of OS-level thread handles (`pthread_t` on Unix, `HANDLE` on
//! Windows) and pin them to physical or logical cores. Most callers
//! collect `pthread_t` / `HANDLE` values from their thread library
//! (`std::thread::current().handle()`-style accessors via `os`-specific
//! crates) before invoking these.
//!
//! The functions are `unsafe` because passing a thread handle that no
//! longer refers to a live thread is undefined behavior on the AOCL
//! side.

use aocl_utils_sys as sys;

/// OS-native thread handle.
pub type ThreadHandle = sys::pthread_t;

/// Pinning strategy for a list of thread handles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PinStrategy {
    /// Pin each thread to a unique physical core (one thread per core,
    /// preferring physical core 0, 1, 2, …).
    Core,
    /// Pin each thread to a unique logical processor (SMT thread).
    Logical,
    /// Spread threads across cores to maximize per-thread cache, mapping
    /// each thread to a different L3 / NUMA region when possible.
    Spread,
}

/// Pin the given thread handles using the chosen strategy.
///
/// # Safety
///
/// Each entry of `threads` must be a valid OS-level handle to a still-live
/// thread. Passing a stale handle (e.g. for a thread that has exited or
/// been detached) is undefined behavior. The slice's length must fit in
/// a C `size_t`.
pub unsafe fn pin_threads(strategy: PinStrategy, threads: &mut [ThreadHandle]) {
    let n = threads.len();
    let p = threads.as_mut_ptr();
    match strategy {
        PinStrategy::Core => unsafe { sys::au_pin_threads_core(p, n) },
        PinStrategy::Logical => unsafe { sys::au_pin_threads_logical(p, n) },
        PinStrategy::Spread => unsafe { sys::au_pin_threads_spread(p, n) },
    }
}

/// Pin threads with a custom mapping: `affinity_vector[i]` is the logical
/// processor index thread `i` should bind to.
///
/// # Safety
///
/// Same as [`pin_threads`]; in addition, the affinity values must
/// reference logical processors that exist on the host.
pub unsafe fn pin_threads_custom(threads: &mut [ThreadHandle], affinity_vector: &[i32]) {
    let n = threads.len();
    if n == 0 || affinity_vector.is_empty() {
        return;
    }
    unsafe {
        sys::au_pin_threads_custom(
            threads.as_mut_ptr(),
            n,
            affinity_vector.as_ptr() as *mut std::os::raw::c_int,
            affinity_vector.len(),
        );
    }
}
