# aocl-error

Shared `Error` and `Result` types for the `aocl-*` family of safe wrappers around AMD's [AOCL](https://www.amd.com/en/developer/aocl.html) libraries.

Pulled out into its own crate so every safe `aocl-*` crate (`aocl-blas`, `aocl-lapack`, `aocl-sparse`, …) returns the *same* error type — there's no need to manually convert between per-library error types when chaining calls across libraries.

Dual-licensed under MIT or Apache-2.0.
