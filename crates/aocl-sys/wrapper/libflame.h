/* AOCL-LAPACK (libFLAME) umbrella header.
 *
 * Pulls in the standard LAPACK Fortran-symbol declarations and the LAPACKE
 * C interface. The libFLAME-native API in `FLAME.h` is intentionally not
 * included here because it redeclares many of the same Fortran symbols
 * with conflicting signatures; users who specifically need it can include
 * it directly via `aocl_sys::libflame` after the safe layer matures.
 */
#include <lapack.h>
#include <lapacke.h>
