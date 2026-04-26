# aocl-utils

Safe wrappers for AOCL-Utils — AMD CPU identification (vendor, Zen sub-architecture, x86-64 microarchitecture levels) and threading helpers.

Built on top of [`aocl-utils-sys`](../aocl-utils-sys/).

## Coverage

- `cpuid::is_amd`, `is_zen_family`, `zen_arch`, `x86_64_level`, `vendor_info`.

Threading-pinning and version queries are next.

Dual-licensed under MIT or Apache-2.0.
