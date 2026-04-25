//! Safe wrappers around the AMD AOCL-Utils CPU identification API.
//!
//! All AOCL `au_cpuid_*` routines that take a `cpu_num` migrate the calling
//! thread to that core if `cpu_num != AU_CURRENT_CPU_NUM`. The [`Cpu`] enum
//! makes this distinction explicit so callers cannot accidentally pin
//! themselves to core 0 by passing a literal `0u32`.

use aocl_sys::utils as sys;

/// Sentinel passed to AOCL to mean "use whatever core the calling thread is
/// already running on, do not migrate". Equal to `UINT32_MAX`.
const AU_CURRENT_CPU_NUM: u32 = u32::MAX;

/// Which CPU to query.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Cpu {
    /// The current thread's CPU. Does not cause thread migration.
    Current,
    /// A specific core by 0-based index. **Calling AOCL with a specific core
    /// will migrate the current thread to that core** for the duration of the
    /// call.
    Specific(u32),
}

impl Cpu {
    fn as_raw(self) -> u32 {
        match self {
            Cpu::Current => AU_CURRENT_CPU_NUM,
            Cpu::Specific(n) => n,
        }
    }
}

/// AMD Zen sub-architectures recognized by AOCL.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ZenArch {
    Zen,
    ZenPlus,
    Zen2,
    Zen3,
    Zen4,
    Zen5,
}

/// x86-64 microarchitecture levels (the System V psABI levels).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[non_exhaustive]
pub enum X86_64Level {
    V2,
    V3,
    V4,
}

/// CPU vendor + family/model details, as reported by `au_cpuid_get_vendor`.
///
/// AOCL returns these as newline-separated string fields:
/// `VendorID\nFamilyID\nModelID\nSteppingID\nUarchID`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VendorInfo {
    pub vendor_id: String,
    pub family_id: String,
    pub model_id: String,
    pub stepping_id: String,
    pub uarch_id: String,
}

/// Returns `true` if the queried CPU is manufactured by AMD.
pub fn is_amd(cpu: Cpu) -> bool {
    // SAFETY: `au_cpuid_is_amd` is a pure CPUID query with no preconditions
    // beyond a valid `cpu_num`; `AU_CURRENT_CPU_NUM` is always valid and any
    // `u32` is accepted (out-of-range cores cause AOCL to fall back to the
    // current core per the AOCL docs).
    unsafe { sys::au_cpuid_is_amd(cpu.as_raw()) }
}

/// Returns `true` if the queried CPU is in the Zen family (Zen, Zen+, Zen2,
/// Zen3, Zen4, Zen5; not Zen6+).
pub fn is_zen_family(cpu: Cpu) -> bool {
    unsafe { sys::au_cpuid_arch_is_zen_family(cpu.as_raw()) }
}

/// Returns the Zen sub-architecture of the queried CPU, if any.
pub fn zen_arch(cpu: Cpu) -> Option<ZenArch> {
    let raw = cpu.as_raw();
    // Order matters: AOCL's `arch_is_zenN` returns true for ZenN *and later*,
    // so we test from the newest backwards to get the most specific answer.
    unsafe {
        if sys::au_cpuid_arch_is_zen5(raw) {
            Some(ZenArch::Zen5)
        } else if sys::au_cpuid_arch_is_zen4(raw) {
            Some(ZenArch::Zen4)
        } else if sys::au_cpuid_arch_is_zen3(raw) {
            Some(ZenArch::Zen3)
        } else if sys::au_cpuid_arch_is_zen2(raw) {
            Some(ZenArch::Zen2)
        } else if sys::au_cpuid_arch_is_zenplus(raw) {
            Some(ZenArch::ZenPlus)
        } else if sys::au_cpuid_arch_is_zen(raw) {
            Some(ZenArch::Zen)
        } else {
            None
        }
    }
}

/// Returns the highest x86-64 microarchitecture level supported by the
/// queried CPU, or `None` if it does not even meet x86-64-v2.
pub fn x86_64_level(cpu: Cpu) -> Option<X86_64Level> {
    let raw = cpu.as_raw();
    unsafe {
        if sys::au_cpuid_arch_is_x86_64v4(raw) {
            Some(X86_64Level::V4)
        } else if sys::au_cpuid_arch_is_x86_64v3(raw) {
            Some(X86_64Level::V3)
        } else if sys::au_cpuid_arch_is_x86_64v2(raw) {
            Some(X86_64Level::V2)
        } else {
            None
        }
    }
}

/// Fetch vendor / family / model / stepping / uarch identifiers for the
/// queried CPU.
pub fn vendor_info(cpu: Cpu) -> VendorInfo {
    // AOCL specifies a buffer of >= 16 bytes; in practice the response is
    // ~64 bytes max (5 short fields, newline-separated). 256 is plenty.
    let mut buf = [0u8; 256];
    // SAFETY: We pass a valid pointer + length pair; AOCL writes a
    // NUL-terminated string of at most `len` bytes.
    unsafe {
        sys::au_cpuid_get_vendor(
            cpu.as_raw(),
            buf.as_mut_ptr() as *mut std::os::raw::c_char,
            buf.len(),
        );
    }
    let nul = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    let raw = std::str::from_utf8(&buf[..nul]).unwrap_or("");
    let mut it = raw.split('\n');
    VendorInfo {
        vendor_id: it.next().unwrap_or("").to_string(),
        family_id: it.next().unwrap_or("").to_string(),
        model_id: it.next().unwrap_or("").to_string(),
        stepping_id: it.next().unwrap_or("").to_string(),
        uarch_id: it.next().unwrap_or("").to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn current_cpu_does_not_panic() {
        // We can't assert specific values (varies by host), but the calls
        // must complete and return a coherent VendorInfo with a non-empty
        // vendor_id on any reasonable x86-64 CPU.
        let info = vendor_info(Cpu::Current);
        assert!(!info.vendor_id.is_empty(), "vendor_id should be populated");
        let _ = is_amd(Cpu::Current);
        let _ = is_zen_family(Cpu::Current);
        let _ = zen_arch(Cpu::Current);
        let _ = x86_64_level(Cpu::Current);
    }
}
