/* AOCL-Compression umbrella header.
 *
 * The unified API entrypoint is aocl_compression.h; per-algorithm headers
 * (lz4.h, lz4hc.h, lz4frame.h, snappy-c.h, bzlib.h, LzmaEnc.h, LzmaDec.h)
 * are also pulled in for direct algorithm-specific access.
 */
#include <aocl_compression.h>
#include <lz4.h>
#include <lz4hc.h>
#include <lz4frame.h>
#include <snappy-c.h>
#include <bzlib.h>
#include <LzmaEnc.h>
#include <LzmaDec.h>
