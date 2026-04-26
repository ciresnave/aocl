//! Build-script helpers shared across the `aocl-*-sys` crates.
//!
//! Each `aocl-foo-sys` crate's `build.rs` calls into this crate to
//! locate the AOCL install, run `bindgen` against a per-component umbrella
//! header, and emit the `cargo:rustc-link-*` directives needed to link the
//! corresponding native library.

use std::env;
use std::path::{Path, PathBuf};

/// Description of a single AOCL component for `build.rs`.
#[derive(Debug, Clone)]
pub struct Component<'a> {
    /// Subdirectory under the AOCL install root, e.g. `"amd-blis"`.
    pub component_dir: &'a str,
    /// Path (relative to the calling crate root) to the umbrella C header.
    pub wrapper: &'a str,
    /// Module / file name to write the bindgen output to inside `OUT_DIR`,
    /// e.g. `"blas"` produces `$OUT_DIR/blas.rs`.
    pub module: &'a str,
    /// Whether this component splits headers / libs into `LP64/` and
    /// `ILP64/` subdirectories.
    pub has_int_subdir: bool,
    /// True if the component only ships an LP64 build (e.g. AOCL-SecureRNG).
    pub only_lp64: bool,
    /// Native lib base names for static Windows linking (no extension, no
    /// `lib` prefix; e.g. `"AOCL-LibBlis-Win-MT"`).
    pub win_static: &'a [&'a str],
    /// Native lib base names for dynamic Windows linking (the matching
    /// `*-dll` import lib).
    pub win_dynamic: &'a [&'a str],
    /// Native lib names for static Unix linking.
    pub unix_static: &'a [&'a str],
    /// Native lib names for dynamic Unix linking.
    pub unix_dynamic: &'a [&'a str],
    /// Additional `-I` includes to pass to bindgen's clang.
    pub extra_includes: &'a [&'a str],
}

impl Component<'_> {
    /// Locate this component under `root`, emit link directives, and run
    /// `bindgen` against [`Self::wrapper`]. Reads `CARGO_FEATURE_ILP64` and
    /// `CARGO_FEATURE_STATIC_LINK` to choose layout.
    ///
    /// Panics on misconfiguration (missing component dir, missing wrapper,
    /// bindgen failure) so build-script breakage surfaces clearly.
    pub fn build(&self, root: &Path) {
        let static_link = env::var_os("CARGO_FEATURE_STATIC_LINK").is_some();
        let ilp64 = env::var_os("CARGO_FEATURE_ILP64").is_some();
        let target_windows =
            env::var("CARGO_CFG_TARGET_OS").unwrap_or_default() == "windows";
        let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR"));

        let comp_dir = root.join(self.component_dir);
        if !comp_dir.exists() {
            panic!(
                "AOCL component '{}' not found at {}. Either install it or \
                 disable the corresponding Cargo feature in your downstream crate.",
                self.component_dir,
                comp_dir.display()
            );
        }

        // Pick include + lib subdirectories.
        let int_sub: &str = if self.only_lp64 {
            "LP64"
        } else if ilp64 {
            "ILP64"
        } else {
            "LP64"
        };
        let (inc_dir, lib_dir) = if self.has_int_subdir {
            (
                comp_dir.join("include").join(int_sub),
                comp_dir.join("lib").join(int_sub),
            )
        } else {
            (comp_dir.join("include"), comp_dir.join("lib"))
        };

        // Some components split libs into shared/ vs. static/ sub-trees.
        let final_lib_dir = {
            let split = if static_link {
                lib_dir.join("static")
            } else {
                lib_dir.join("shared")
            };
            if split.exists() {
                split
            } else {
                lib_dir.clone()
            }
        };

        if final_lib_dir.exists() {
            println!(
                "cargo:rustc-link-search=native={}",
                final_lib_dir.display()
            );
        } else {
            println!(
                "cargo:warning=lib dir not found for '{}': {}",
                self.component_dir,
                final_lib_dir.display()
            );
        }

        // Pick lib names per OS + linkage.
        let names: &[&str] = match (target_windows, static_link) {
            (true, true) => self.win_static,
            (true, false) => self.win_dynamic,
            (false, true) => self.unix_static,
            (false, false) => self.unix_dynamic,
        };
        let kind = if static_link { "static" } else { "dylib" };
        for n in names {
            println!("cargo:rustc-link-lib={kind}={n}");
        }

        // Run bindgen.
        if self.wrapper.is_empty() {
            // Link-only component (e.g. ScaLAPACK has no public C headers).
            return;
        }
        let wrapper_path = PathBuf::from(self.wrapper);
        if !wrapper_path.exists() {
            panic!(
                "Missing wrapper header for component '{}': {}",
                self.component_dir,
                wrapper_path.display()
            );
        }
        println!("cargo:rerun-if-changed={}", wrapper_path.display());

        let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();

        let mut builder = bindgen::Builder::default()
            .header(wrapper_path.to_string_lossy())
            .clang_arg(format!("-I{}", inc_dir.display()))
            .clang_arg(format!("-I{}", comp_dir.join("include").display()))
            .layout_tests(false)
            .derive_default(true)
            .derive_debug(true)
            .generate_comments(false)
            // Restrict bindings to definitions inside this AOCL component.
            // Keeps transitively-included system headers (Windows.h etc.)
            // out of the generated bindings.
            .allowlist_file(format!(".*{}.*", self.component_dir))
            .allowlist_file(format!(".*{}.*", self.wrapper))
            .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()));

        for inc in self.extra_includes {
            builder = builder.clang_arg(format!("-I{inc}"));
        }

        // x86 SIMD intrinsic baseline.
        if target_arch == "x86_64" || target_arch == "x86" {
            builder = builder
                .clang_arg("-msse4.2")
                .clang_arg("-mavx2")
                .clang_arg("-mfma");
        }

        if ilp64 {
            builder = builder
                .clang_arg("-DAOCL_ILP64")
                .clang_arg("-DBLIS_ENABLE_ILP64")
                .clang_arg("-DLAPACK_ILP64");
        }

        let bindings = builder.generate().unwrap_or_else(|e| {
            panic!(
                "bindgen failed for component '{}' (header {}): {e}",
                self.component_dir,
                wrapper_path.display()
            )
        });

        let out_path = out_dir.join(format!("{}.rs", self.module));
        bindings.write_to_file(&out_path).unwrap_or_else(|e| {
            panic!("failed to write {}: {e}", out_path.display())
        });
    }
}

/// Locate the AOCL install root.
///
/// Resolution order:
///   1. `AOCL_ROOT` environment variable.
///   2. Windows: `C:\Program Files\AMD\AOCL-Windows`.
///   3. Linux: any directory matching `/opt/AMD/aocl/aocl-linux-*`.
///
/// Panics if no install can be found, with instructions on how to set
/// `AOCL_ROOT`.
pub fn locate_aocl_root() -> PathBuf {
    if let Ok(v) = env::var("AOCL_ROOT") {
        let p = PathBuf::from(v);
        if p.exists() {
            return p;
        }
        panic!("AOCL_ROOT is set but does not exist: {}", p.display());
    }
    if cfg!(target_os = "windows") {
        let p = PathBuf::from(r"C:\Program Files\AMD\AOCL-Windows");
        if p.exists() {
            return p;
        }
    } else if let Ok(entries) = std::fs::read_dir("/opt/AMD/aocl") {
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir()
                && p.file_name()
                    .and_then(|s| s.to_str())
                    .is_some_and(|n| n.starts_with("aocl-linux-"))
            {
                return p;
            }
        }
    }
    panic!(
        "Could not locate AOCL. Install AOCL from \
         https://www.amd.com/en/developer/aocl.html and set AOCL_ROOT \
         to the install root."
    );
}

/// On Windows, if neither `LIBCLANG_PATH` nor `LLVM_CONFIG_PATH` is set,
/// auto-detect a standard LLVM install. Without this, `clang-sys` may pick
/// up an unrelated `libclang.dll` from `PATH` (e.g. one bundled with the
/// Swift toolchain) that mishandles AOCL's SIMD intrinsic headers.
pub fn ensure_libclang_path() {
    if env::var_os("LIBCLANG_PATH").is_some()
        || env::var_os("LLVM_CONFIG_PATH").is_some()
    {
        return;
    }
    if cfg!(target_os = "windows") {
        for candidate in [
            r"C:\Program Files\LLVM\bin",
            r"C:\Program Files (x86)\LLVM\bin",
        ] {
            let p = PathBuf::from(candidate);
            if p.join("libclang.dll").exists() {
                println!(
                    "cargo:warning=auto-detected libclang at {}",
                    p.display()
                );
                env::set_var("LIBCLANG_PATH", &p);
                return;
            }
        }
    }
}
