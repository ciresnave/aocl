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
        // FFTW filenames vary by precision: libfftw3 is double, libfftw3f
        // is single, libfftw3l is long double. We link both single and
        // double; long-double is not yet exposed by the safe wrapper.
        win_static: &["libfftw3", "libfftw3f"],
        win_dynamic: &["libfftw3", "libfftw3f"],
        unix_static: &["fftw3", "fftw3f"],
        unix_dynamic: &["fftw3", "fftw3f"],
        extra_includes: &[],
    }
    .build(&root);
}
