/* AOCL-LAPACK (libFLAME) umbrella header.
 *
 * Pulls in the standard LAPACK Fortran-symbol declarations and the LAPACKE
 * C interface. The libFLAME-native API in `FLAME.h` is intentionally not
 * included because it redeclares many Fortran symbols with conflicting
 * signatures vs `lapack.h` / `lapacke.h`.
 */
#include <lapack.h>
#include <lapacke.h>
