//! Build script for `aocl-sys`.
//!
//! Per enabled Cargo feature, this script:
//!   1. Locates the corresponding AOCL component under `$AOCL_ROOT`.
//!   2. Emits `cargo:rustc-link-search` / `cargo:rustc-link-lib` directives.
//!   3. Runs `bindgen` against a per-component umbrella header in `wrapper/`,
//!      writing the generated bindings to `$OUT_DIR/<module>.rs`. `lib.rs`
//!      then `include!`s the generated file inside the matching module.

use std::env;
use std::path::PathBuf;

#[derive(Debug, Clone, Copy)]
struct LibSpec {
    feature: &'static str,
    component: &'static str,
    wrapper: &'static str,
    module: &'static str,
    has_int_subdir: bool,
    only_lp64: bool,
    win_static: &'static [&'static str],
    win_dynamic: &'static [&'static str],
    unix_static: &'static [&'static str],
    unix_dynamic: &'static [&'static str],
    extra_includes: &'static [&'static str],
}

const LIBRARIES: &[LibSpec] = &[
    LibSpec {
        feature: "blis",
        component: "amd-blis",
        wrapper: "blis.h",
        module: "blis",
        has_int_subdir: true,
        only_lp64: false,
        win_static: &["AOCL-LibBlis-Win-MT"],
        win_dynamic: &["AOCL-LibBlis-Win-MT-dll"],
        unix_static: &["blis-mt"],
        unix_dynamic: &["blis-mt"],
        extra_includes: &[],
    },
    LibSpec {
        feature: "libflame",
        component: "amd-libflame",
        wrapper: "libflame.h",
        module: "libflame",
        has_int_subdir: true,
        only_lp64: false,
        win_static: &["AOCL-LibFlame-Win-MT"],
        win_dynamic: &["AOCL-LibFlame-Win-MT-dll"],
        unix_static: &["flame"],
        unix_dynamic: &["flame"],
        extra_includes: &[],
    },
    LibSpec {
        feature: "utils",
        component: "amd-utils",
        wrapper: "utils.h",
        module: "utils",
        has_int_subdir: false,
        only_lp64: false,
        win_static: &["libaoclutils_static", "au_cpuid_static"],
        win_dynamic: &["libaoclutils", "au_cpuid"],
        unix_static: &["aoclutils", "au_cpuid"],
        unix_dynamic: &["aoclutils", "au_cpuid"],
        extra_includes: &[],
    },
    LibSpec {
        feature: "libm",
        component: "amd-libm",
        wrapper: "libm.h",
        module: "libm",
        has_int_subdir: false,
        only_lp64: false,
        win_static: &["libalm-static"],
        win_dynamic: &["libalm"],
        unix_static: &["alm"],
        unix_dynamic: &["alm"],
        extra_includes: &[],
    },
    LibSpec {
        feature: "fftw",
        component: "amd-fftw",
        wrapper: "fftw.h",
        module: "fftw",
        has_int_subdir: false,
        only_lp64: false,
        // FFTW filenames vary by precision (libfftw3, libfftw3f, libfftw3l).
        // For v0 we link the double-precision variant; precision-specific
        // features can be added later.
        win_static: &["libfftw3"],
        win_dynamic: &["libfftw3"],
        unix_static: &["fftw3"],
        unix_dynamic: &["fftw3"],
        extra_includes: &[],
    },
    LibSpec {
        feature: "sparse",
        component: "amd-sparse",
        wrapper: "sparse.h",
        module: "sparse",
        has_int_subdir: true,
        only_lp64: false,
        win_static: &["aoclsparse-static"],
        win_dynamic: &["aoclsparse"],
        unix_static: &["aoclsparse"],
        unix_dynamic: &["aoclsparse"],
        extra_includes: &[],
    },
    LibSpec {
        feature: "rng",
        component: "amd-rng",
        wrapper: "rng.h",
        module: "rng",
        has_int_subdir: true,
        only_lp64: false,
        win_static: &["rng_amd-static"],
        win_dynamic: &["rng_amd"],
        unix_static: &["amdrng"],
        unix_dynamic: &["amdrng"],
        extra_includes: &[],
    },
    LibSpec {
        feature: "securerng",
        component: "amd-securerng",
        wrapper: "securerng.h",
        module: "securerng",
        has_int_subdir: true,
        only_lp64: true,
        win_static: &["amdsecrng-static"],
        win_dynamic: &["amdsecrng"],
        unix_static: &["secrng"],
        unix_dynamic: &["secrng"],
        extra_includes: &[],
    },
    LibSpec {
        feature: "compression",
        component: "amd-compression",
        wrapper: "compression.h",
        module: "compression",
        has_int_subdir: false,
        only_lp64: false,
        win_static: &["aocl_compression-static"],
        win_dynamic: &["aocl_compression"],
        unix_static: &["aocl_compression"],
        unix_dynamic: &["aocl_compression"],
        extra_includes: &[],
    },
    LibSpec {
        feature: "crypto",
        component: "amd-crypto",
        wrapper: "crypto.h",
        module: "crypto",
        has_int_subdir: false,
        only_lp64: false,
        win_static: &["alcp_static"],
        win_dynamic: &["alcp"],
        unix_static: &["alcp"],
        unix_dynamic: &["alcp"],
        extra_includes: &[],
    },
    LibSpec {
        feature: "data-analytics",
        component: "amd-da",
        wrapper: "data_analytics.h",
        module: "data_analytics",
        has_int_subdir: true,
        only_lp64: false,
        win_static: &["aocl-da"],
        win_dynamic: &["aocl-da"],
        unix_static: &["aocl-da"],
        unix_dynamic: &["aocl-da"],
        extra_includes: &[],
    },
    LibSpec {
        feature: "scalapack",
        component: "amd-scalapack",
        // ScaLAPACK is a Fortran/MPI library; no public C headers ship with it.
        // Users call into it via their MPI-Fortran toolchain. We only emit
        // link directives for completeness.
        wrapper: "",
        module: "scalapack",
        has_int_subdir: true,
        only_lp64: false,
        win_static: &["scalapack"],
        win_dynamic: &["scalapack"],
        unix_static: &["scalapack"],
        unix_dynamic: &["scalapack"],
        extra_includes: &[],
    },
];

fn feature_enabled(feature: &str) -> bool {
    let var = format!(
        "CARGO_FEATURE_{}",
        feature.replace('-', "_").to_uppercase()
    );
    env::var_os(&var).is_some()
}

fn locate_aocl_root() -> PathBuf {
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

fn int_subdir(only_lp64: bool) -> &'static str {
    if only_lp64 {
        "LP64"
    } else if feature_enabled("ilp64") {
        "ILP64"
    } else {
        "LP64"
    }
}

/// On Windows, if neither `LIBCLANG_PATH` nor `LLVM_CONFIG_PATH` is set, try
/// the standard LLVM install locations. Without this, on systems where
/// another libclang appears earlier on `PATH` (e.g. the one bundled with the
/// Swift toolchain) `clang-sys` can pick that up, which breaks parsing of
/// the x86 intrinsic headers AOCL pulls in.
fn ensure_libclang_path() {
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
            let p = std::path::PathBuf::from(candidate);
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

fn main() {
    println!("cargo:rerun-if-env-changed=AOCL_ROOT");
    println!("cargo:rerun-if-env-changed=LIBCLANG_PATH");
    println!("cargo:rerun-if-changed=wrapper");
    println!("cargo:rerun-if-changed=build.rs");

    ensure_libclang_path();

    let static_link = feature_enabled("static-link");
    let ilp64 = feature_enabled("ilp64");
    let target_windows =
        env::var("CARGO_CFG_TARGET_OS").unwrap_or_default() == "windows";
    let root = locate_aocl_root();
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").unwrap());

    println!("cargo:warning=AOCL root: {}", root.display());
    println!(
        "cargo:warning=Indexing: {} | Linkage: {}",
        if ilp64 { "ILP64" } else { "LP64" },
        if static_link { "static" } else { "dynamic" }
    );

    for lib in LIBRARIES {
        if !feature_enabled(lib.feature) {
            continue;
        }

        let comp_dir = root.join(lib.component);
        if !comp_dir.exists() {
            panic!(
                "AOCL component '{}' (feature '{}') not found at {}. \
                 Either install it or disable the feature.",
                lib.component,
                lib.feature,
                comp_dir.display()
            );
        }

        let (inc_dir, lib_dir) = if lib.has_int_subdir {
            let sub = int_subdir(lib.only_lp64);
            (
                comp_dir.join("include").join(sub),
                comp_dir.join("lib").join(sub),
            )
        } else {
            (comp_dir.join("include"), comp_dir.join("lib"))
        };

        // Some components (sparse, fftw, compression) split their libs
        // between `shared/` and `static/` subdirectories. Pick whichever
        // matches the requested linkage.
        let final_lib_dir = {
            let split = if static_link {
                lib_dir.join("static")
            } else {
                lib_dir.join("shared")
            };
            if split.exists() { split } else { lib_dir.clone() }
        };

        if final_lib_dir.exists() {
            println!(
                "cargo:rustc-link-search=native={}",
                final_lib_dir.display()
            );
        } else {
            println!(
                "cargo:warning=lib dir not found for '{}': {}",
                lib.feature,
                final_lib_dir.display()
            );
        }

        let names: &[&str] = match (target_windows, static_link) {
            (true, true) => lib.win_static,
            (true, false) => lib.win_dynamic,
            (false, true) => lib.unix_static,
            (false, false) => lib.unix_dynamic,
        };
        let kind = if static_link { "static" } else { "dylib" };
        for n in names {
            println!("cargo:rustc-link-lib={kind}={n}");
        }

        if lib.wrapper.is_empty() {
            continue; // e.g. scalapack: link-only, no public C headers
        }

        let wrapper_path = PathBuf::from("wrapper").join(lib.wrapper);
        if !wrapper_path.exists() {
            panic!(
                "Missing wrapper header: {} (expected for feature '{}')",
                wrapper_path.display(),
                lib.feature
            );
        }
        println!("cargo:rerun-if-changed={}", wrapper_path.display());

        let target_arch =
            env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();

        let mut builder = bindgen::Builder::default()
            .header(wrapper_path.to_string_lossy())
            // Primary include for this component's headers.
            .clang_arg(format!("-I{}", inc_dir.display()))
            // Some components nest internal headers under the parent include
            // (e.g. amd-utils references "Capi/au/cpuid/cpuid.h").
            .clang_arg(format!("-I{}", comp_dir.join("include").display()))
            .layout_tests(false)
            .derive_default(true)
            .derive_debug(true)
            .generate_comments(false)
            // Restrict bindings to declarations defined inside the AOCL
            // component directory itself. Without this, transitively-included
            // system headers (Windows.h, etc.) bleed thousands of irrelevant
            // declarations into the bindings — many of which violate our
            // `derive_default` request (large arrays etc.).
            .allowlist_file(format!(".*{}.*", lib.component))
            // Also allow the wrapper header itself (so the entry-point
            // doesn't get filtered out).
            .allowlist_file(format!(".*{}.*", lib.wrapper))
            .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()));

        for inc in lib.extra_includes {
            builder = builder.clang_arg(format!("-I{inc}"));
        }

        // Several AOCL components include x86 SIMD intrinsic headers
        // (mmintrin.h, immintrin.h, ...). Without explicit target-feature
        // flags, clang refuses to parse the vector-type definitions.
        // Enabling SSE4.2 + AVX2 covers everything AOCL ships and is a
        // safe baseline for any modern AMD CPU.
        if target_arch == "x86_64" || target_arch == "x86" {
            builder = builder
                .clang_arg("-msse4.2")
                .clang_arg("-mavx2")
                .clang_arg("-mfma");
        }

        if ilp64 {
            // Several AOCL libraries gate ILP64 typedefs on these macros.
            builder = builder
                .clang_arg("-DAOCL_ILP64")
                .clang_arg("-DBLIS_ENABLE_ILP64")
                .clang_arg("-DLAPACK_ILP64");
        }

        let bindings = builder.generate().unwrap_or_else(|e| {
            panic!(
                "bindgen failed for feature '{}' (header {}): {e}",
                lib.feature,
                wrapper_path.display()
            )
        });

        let out_path = out_dir.join(format!("{}.rs", lib.module));
        bindings.write_to_file(&out_path).unwrap_or_else(|e| {
            panic!("failed to write {}: {e}", out_path.display())
        });
    }
}
