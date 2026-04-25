//! AOCL-Utils — CPU identification and threading helpers.
//!
//! Currently exposes the [`cpuid`] sub-module wrapping `au_cpuid_*` from
//! amd-utils. Threading-pinning and version queries will follow.

pub mod cpuid;
