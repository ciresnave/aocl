/* AOCL-Cryptography (ALCP) umbrella header.
 *
 * Pulls in the public ALCP C API. Individual subsystems (cipher, digest,
 * MAC, RSA, EC, RNG) live under <alcp/>. The umbrella alcp/alcp.h does
 * not include rsa.h, so we pull it in explicitly for asymmetric crypto.
 */
#include <alcp/alcp.h>
#include <alcp/rsa.h>
