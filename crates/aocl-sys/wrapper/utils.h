/* AOCL-Utils umbrella header.
 *
 * Pulls in the public C interfaces of amd-utils:
 *   - alci/alci_c.h transitively includes Capi/au/cpuid/cpuid.h, providing
 *     the CPU identification API (vendor, microarchitecture, feature flags).
 *   - The Capi/au top-level header brings in Au C-ABI type/error definitions.
 */
#include <alci/alci.h>
#include <alci/alci_c.h>
#include <alci/arch.h>
#include <Capi/au/au.h>
#include <Capi/au/cpuid/cpuid.h>
#include <Capi/au/error.h>
#include <Capi/au/threadpinning.h>
#include <Capi/au/version.h>
