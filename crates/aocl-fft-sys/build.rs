use aocl_build::{ensure_libclang_path, locate_aocl_root, Component};

fn main() {
    println!("cargo:rerun-if-env-changed=AOCL_ROOT");
    println!("cargo:rerun-if-env-changed=LIBCLANG_PATH");
    println!("cargo:rerun-if-changed=wrapper");
    println!("cargo:rerun-if-changed=build.rs");

    ensure_libclang_path();
    let root = locate_aocl_root();

    Component {
        component_dir: "amd-fftw",
        wrapper: "wrapper/fft.h",
        module: "bindings",
        has_int_subdir: false,
        only_lp64: false,
        // FFTW filenames vary by precision (libfftw3 = double, libfftw3f =
        // float, libfftw3l = long double). The double-precision build is
        // exposed here; precision-specific features can be added later.
        win_static: &["libfftw3"],
        win_dynamic: &["libfftw3"],
        unix_static: &["fftw3"],
        unix_dynamic: &["fftw3"],
        extra_includes: &[],
    }
    .build(&root);
}
