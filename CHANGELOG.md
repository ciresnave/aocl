# Changelog

## 0.1.7

- Docs only: ROADMAP row 7 (`aocl-scalapack`) is scheduled for when an MPI consumer exists, per PM ruling 2026-11-01. No code change.

## 0.1.6

- Docs only: ROADMAP rows 4 and 5 record that `aocl-blas-sys` regenerates with bindgen 0.73.2 against AOCL 5.3.2 headers, and that the vendored docs.rs bindings do not match that version (265 names only in the vendored file, 568 only in the regenerated one, 30 constant values differ). No code change.

## 0.1.5

- Dependencies to latest: `thiserror` 1 to 2, `bindgen` 0.69 to 0.73. No public API change; `libc` stays on 0.2 (crates.io has only 1.0.0-alpha).

## 0.1.4

- Add `deny.toml` and a `cargo deny` CI job (advisories, licences, bans, sources); no code change.

## 0.1.3

- Add `ROADMAP.md` (docs only; no code change). Earlier history: see git log (0.1.0 published 2026-04-27; 0.1.1 and 0.1.2 not yet published).
