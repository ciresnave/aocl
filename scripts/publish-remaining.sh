#!/usr/bin/env bash
# One-shot publisher for the remaining tiers, with rate-limit retry.
# After Tier 1 (aocl-error, aocl-types, aocl-build) and the first three
# *-sys (aocl-utils-sys, aocl-securerng-sys, aocl-math-sys, plus
# whatever else has already gone live), this script picks up the rest
# in dependency order.
set -u

publish_one() {
  local pkg=$1
  echo "=== $(date -u +%H:%M:%S) publishing $pkg ==="
  while true; do
    out=$(cargo publish -p "$pkg" 2>&1)
    if echo "$out" | grep -q "Published $pkg" || echo "$out" | grep -q "already exists"; then
      echo "PUB OK: $pkg"
      return 0
    fi
    if echo "$out" | grep -q "crate version .* is already uploaded"; then
      echo "ALREADY: $pkg"
      return 0
    fi
    if echo "$out" | grep -q "429"; then
      retry=$(echo "$out" | grep -oE 'after [A-Za-z0-9, :]+GMT' | head -1)
      echo "RATE LIMITED on $pkg ($retry); waiting 90s"
      sleep 90
      continue
    fi
    echo "FAIL on $pkg:"
    echo "$out" | tail -15
    return 1
  done
}

# Remaining Tier 2 (*-sys) — fft already kicked off separately, but we
# guard against duplicates.
TIER2=(
  aocl-blas-sys
  aocl-lapack-sys
  aocl-sparse-sys
  aocl-rng-sys
  aocl-compression-sys
  aocl-crypto-sys
  aocl-data-analytics-sys
  aocl-scalapack-sys
)

# Tier 3 — safe wrappers (each depends on its *-sys + foundation).
TIER3=(
  aocl-utils
  aocl-securerng
  aocl-math
  aocl-fft
  aocl-blas
  aocl-lapack
  aocl-sparse
  aocl-rng
  aocl-compression
  aocl-crypto
  aocl-data-analytics
  aocl-scalapack
)

# Tier 4 — umbrellas. aocl-sys re-exports *-sys; aocl re-exports safe.
TIER4=(
  aocl-sys
  aocl
)

for p in "${TIER2[@]}" "${TIER3[@]}" "${TIER4[@]}"; do
  publish_one "$p" || { echo "ABORTING at $p"; exit 1; }
done

echo "ALL TIERS DONE"
