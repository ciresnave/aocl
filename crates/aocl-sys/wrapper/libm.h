/* AOCL-LibM umbrella header.
 *
 * Pulls in only the scalar API (`amdlibm.h`). The vector API in
 * `amdlibm_vec.h` is gated behind AMD_LIBM_VEC_EXPERIMENTAL by AMD and
 * pulls in x86 SIMD intrinsics that vary by toolchain — opt in to it
 * via a future `libm-vec` feature when stabilized.
 */
#include <amdlibm.h>
